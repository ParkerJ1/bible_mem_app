"""Session Manager — orchestrates a full Bible memorisation practice session.

Responsibilities
----------------
* prepare_verse(passage_ref)
    Called once when a user adds a verse to their list.  Fetches the passage
    text, synthesises audio via the TTS provider, runs forced alignment, and
    stores all results in the database so they can be reused in every future
    session without repeating the expensive steps.

* get_segments(user_id, passage_ref) -> list[str]
    Loads the user's current proficiency level and uses the Segmenter to
    divide the passage text into appropriately-sized text segments.

* get_segment_audio(user_id, passage_ref, segment_idx) -> Path
    Slices the precomputed full-passage audio file at the word-level
    timestamps that correspond to the requested segment.

* score_attempt(user_id, passage_ref, segment_idx, user_audio_path) -> SegmentResult
    Runs speech recognition on the user's recorded audio, then scores the
    transcript against the expected segment text.

* finish_session(user_id, passage_ref, segment_results, attempt_date) -> SessionResult
    Averages the per-segment scores, records the session in the database, and
    updates the user's proficiency level via the progression engine.

Audio slicing note
------------------
Full-passage audio is sliced with a small silence pad on each side so segment
boundaries do not feel abrupt.  The sliced files are cached in the directory
supplied at construction time (default: segment_cache/).  They are never
re-sliced once the cached file exists.

REFERENCE_ONLY level
--------------------
At this level the user recites the full passage from memory without hearing
any audio prompt.  get_segment_audio() returns None; score_attempt() scores
the user's recording against the full passage text.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, ContextManager

from src.aligner.base import Aligner, WordTimestamp
from src.data.models import PreparedPassage, SessionRecord
from src.progression.engine import ProgressionEngine
from src.progression.levels import LEVEL_TARGET_WORDS, Level
from src.scorer.base import DiffToken, Scorer
from src.segmenter.segmenter import PunctuationSegmenter
from src.sr.base import SpeechRecogniser
from src.text.base import TextProvider
from src.tts.base import TTSProvider

_SEGMENTER = PunctuationSegmenter()
_SEGMENT_PAD_MS = 150  # silence padding added to each side of a sliced segment


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass
class SegmentResult:
    """The outcome of a single segment attempt."""

    segment_idx: int
    expected: str
    score: float
    diff: list[DiffToken] = field(default_factory=list)


@dataclass
class SessionResult:
    """The complete result of a finished practice session."""

    passage_ref: str
    overall_score: float
    segment_results: list[SegmentResult]
    level_before: Level
    level_after: Level
    passed: bool


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Orchestrates a single Bible memorisation practice session.

    All external dependencies are injected so that each component can be
    swapped or mocked independently.

    Args:
        text_provider:   Fetches verse text.
        tts_provider:    Synthesises full-passage audio.
        aligner:         Produces word-level timestamps from audio + transcript.
        recogniser:      Transcribes the user's recorded audio.
        scorer:          Scores a transcript against expected text.
        progression:     Tracks and updates per-user, per-verse proficiency.
        db_factory:      Zero-argument callable returning a DB session context
                         manager (i.e. ``get_session`` from database.py).
        segments_dir:    Directory where sliced segment audio files are cached.
                         Created automatically if it does not exist.
    """

    def __init__(
        self,
        text_provider: TextProvider,
        tts_provider: TTSProvider,
        aligner: Aligner,
        recogniser: SpeechRecogniser,
        scorer: Scorer,
        progression: ProgressionEngine,
        db_factory: Callable[[], ContextManager],
        segments_dir: Path = Path("segment_cache"),
    ) -> None:
        self._text = text_provider
        self._tts = tts_provider
        self._aligner = aligner
        self._recogniser = recogniser
        self._scorer = scorer
        self._progression = progression
        self._db = db_factory
        self._segments_dir = segments_dir
        self._segments_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare_verse(self, passage_ref: str) -> None:
        """Fetch text, synthesise audio, and align — store results in DB.

        Idempotent: if the passage has already been prepared (row exists in
        prepared_passages), this method returns immediately without repeating
        any work.

        Args:
            passage_ref: A reference string such as "John 3:16" or
                         "Romans 8:28-30".
        """
        with self._db() as session:
            existing = (
                session.query(PreparedPassage)
                .filter_by(passage_ref=passage_ref)
                .first()
            )
            if existing:
                return

        book, chapter, verse_start, verse_end = _parse_passage_ref(passage_ref)
        verses = self._text.get_verses(book, chapter, verse_start, verse_end)
        full_text = " ".join(v.text for v in verses)

        audio_path = self._tts.synthesise(full_text)
        timestamps = self._aligner.align(audio_path, full_text)
        timestamps_json = json.dumps(
            [{"word": t.word, "start": t.start, "end": t.end} for t in timestamps]
        )

        with self._db() as session:
            session.add(
                PreparedPassage(
                    passage_ref=passage_ref,
                    full_text=full_text,
                    audio_path=str(audio_path),
                    timestamps_json=timestamps_json,
                )
            )

    # ------------------------------------------------------------------
    # Level query
    # ------------------------------------------------------------------

    def get_level(self, user_id: int, passage_ref: str) -> Level:
        """Return the user's current proficiency level for this passage."""
        return self._progression.get_level(user_id, passage_ref)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def get_segments(self, user_id: int, passage_ref: str) -> list[str]:
        """Return the text segments for the user's current proficiency level.

        Args:
            user_id:     The user's database id.
            passage_ref: The passage reference string.

        Returns:
            Ordered list of text segment strings.  At REFERENCE_ONLY level a
            single-element list containing the passage_ref itself is returned.
        """
        level = self._progression.get_level(user_id, passage_ref)
        target = LEVEL_TARGET_WORDS[level]

        if target is None:
            # REFERENCE_ONLY: user recites entirely from memory
            return [passage_ref]

        full_text = self._load_full_text(passage_ref)
        return _SEGMENTER.segment(full_text, target)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def get_segment_audio(
        self, user_id: int, passage_ref: str, segment_idx: int
    ) -> Path | None:
        """Return the path to the sliced audio for a single segment.

        The file is created on first call and cached for subsequent requests.

        Args:
            user_id:      The user's database id.
            passage_ref:  The passage reference string.
            segment_idx:  Zero-based index into get_segments().

        Returns:
            Path to the MP3 audio file, or None at REFERENCE_ONLY level.
        """
        level = self._progression.get_level(user_id, passage_ref)
        target = LEVEL_TARGET_WORDS[level]

        if target is None:
            return None

        with self._db() as session:
            pp = (
                session.query(PreparedPassage)
                .filter_by(passage_ref=passage_ref)
                .one()
            )
            full_text = pp.full_text
            audio_path = Path(pp.audio_path)
            timestamps = _deserialise_timestamps(pp.timestamps_json)

        segments = _SEGMENTER.segment(full_text, target)
        if segment_idx >= len(segments):
            raise IndexError(
                f"segment_idx {segment_idx} out of range "
                f"(passage has {len(segments)} segments at level {level.name})"
            )

        word_start, word_end = _segment_word_range(segments, segment_idx)
        dest = self._segments_dir / f"{_ref_slug(passage_ref)}_{level.value}_{segment_idx}.mp3"

        if dest.exists():
            return dest

        return _slice_audio(audio_path, timestamps, word_start, word_end, dest)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_attempt(
        self,
        user_id: int,
        passage_ref: str,
        segment_idx: int,
        user_audio_path: Path,
    ) -> SegmentResult:
        """Transcribe the user's recording and score it against the expected text.

        Args:
            user_id:          The user's database id.
            passage_ref:      The passage reference string.
            segment_idx:      Zero-based segment index (ignored at REFERENCE_ONLY
                              level — the full passage text is used instead).
            user_audio_path:  Path to the user's recorded audio file.

        Returns:
            SegmentResult with the expected text, score, and word-level diff.
        """
        level = self._progression.get_level(user_id, passage_ref)
        target = LEVEL_TARGET_WORDS[level]

        full_text = self._load_full_text(passage_ref)

        if target is None:
            expected = full_text
        else:
            segments = _SEGMENTER.segment(full_text, target)
            expected = segments[segment_idx]

        transcript = self._recogniser.recognise(user_audio_path)
        result = self._scorer.score(expected, transcript.text)

        return SegmentResult(
            segment_idx=segment_idx,
            expected=expected,
            score=result.score,
            diff=result.diff,
        )

    # ------------------------------------------------------------------
    # Finishing
    # ------------------------------------------------------------------

    def finish_session(
        self,
        user_id: int,
        passage_ref: str,
        segment_results: list[SegmentResult],
        attempt_date: date | None = None,
    ) -> SessionResult:
        """Record the session result and update the user's proficiency level.

        The overall score is the arithmetic mean of all per-segment scores.

        Args:
            user_id:         The user's database id.
            passage_ref:     The passage reference string.
            segment_results: List of SegmentResult objects, one per segment.
            attempt_date:    Calendar date of the session (defaults to today).

        Returns:
            SessionResult with the overall score and updated proficiency level.
        """
        if not segment_results:
            raise ValueError("segment_results must not be empty")

        today = attempt_date or date.today()
        overall_score = sum(r.score for r in segment_results) / len(segment_results)

        outcome = self._progression.record_attempt(
            user_id, passage_ref, overall_score, today
        )

        with self._db() as session:
            session.add(
                SessionRecord(
                    user_id=user_id,
                    verse_ref=passage_ref,
                    score=overall_score,
                    level_before=outcome.level_before.value,
                    level_after=outcome.level_after.value,
                    attempt_date=today,
                )
            )

        return SessionResult(
            passage_ref=passage_ref,
            overall_score=round(overall_score, 4),
            segment_results=segment_results,
            level_before=outcome.level_before,
            level_after=outcome.level_after,
            passed=outcome.passed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_full_text(self, passage_ref: str) -> str:
        """Return the stored full text for a prepared passage."""
        with self._db() as session:
            pp = (
                session.query(PreparedPassage)
                .filter_by(passage_ref=passage_ref)
                .one()
            )
            return pp.full_text


# ---------------------------------------------------------------------------
# Module-level helpers (independently testable)
# ---------------------------------------------------------------------------


def _parse_passage_ref(passage_ref: str) -> tuple[str, int, int, int]:
    """Parse a passage reference string into its components.

    Supports single-verse ("John 3:16") and verse-range ("Romans 8:28-30")
    formats.

    Returns:
        (book, chapter, verse_start, verse_end) — verse_end equals verse_start
        for single-verse references.

    Raises:
        ValueError: If the string does not match the expected format.
    """
    m = re.match(r"^(.+?)\s+(\d+):(\d+)(?:-(\d+))?$", passage_ref.strip())
    if not m:
        raise ValueError(f"Cannot parse passage reference: {passage_ref!r}")
    book = m.group(1)
    chapter = int(m.group(2))
    verse_start = int(m.group(3))
    verse_end = int(m.group(4)) if m.group(4) else verse_start
    return book, chapter, verse_start, verse_end


def _deserialise_timestamps(timestamps_json: str) -> list[WordTimestamp]:
    """Reconstruct a list of WordTimestamp objects from stored JSON."""
    raw = json.loads(timestamps_json)
    return [WordTimestamp(word=w["word"], start=w["start"], end=w["end"]) for w in raw]


def _segment_word_range(segments: list[str], segment_idx: int) -> tuple[int, int]:
    """Return the (word_start, word_end) indices of segment_idx in the full passage.

    Indices are inclusive, zero-based counts of words across the entire passage.
    """
    words_before = sum(len(s.split()) for s in segments[:segment_idx])
    segment_word_count = len(segments[segment_idx].split())
    return words_before, words_before + segment_word_count - 1


def _slice_audio(
    source_path: Path,
    timestamps: list[WordTimestamp],
    word_start: int,
    word_end: int,
    dest_path: Path,
) -> Path:
    """Slice a segment from source_path using word-level timestamps.

    The slice boundaries are looked up in *timestamps* by word index.  If
    the timestamps list is shorter than expected (some words were unaligned),
    the nearest available timestamp is used as a fallback.

    A small silence pad (``_SEGMENT_PAD_MS``) is added to each side so that
    segment boundaries do not feel abrupt during playback.

    The slice is performed by ffmpeg (already required by whisperx) using
    stream-copy mode, so no re-encoding happens.  A small silence pad is added
    to each boundary so the segment does not feel clipped.

    Args:
        source_path: Full-passage MP3 file produced by the TTS provider.
        timestamps:  Word-level alignment from the Aligner.
        word_start:  Inclusive start word index within the full passage.
        word_end:    Inclusive end word index within the full passage.
        dest_path:   Where the sliced audio will be written.

    Returns:
        dest_path (the sliced audio file).
    """
    if dest_path.exists():
        return dest_path

    if not timestamps:
        raise ValueError("Cannot slice audio: no timestamps available for this passage.")

    pad_s = _SEGMENT_PAD_MS / 1000.0

    # Clamp indices to the available timestamp range
    max_idx = len(timestamps) - 1
    t_start = max(0.0, timestamps[min(word_start, max_idx)].start - pad_s)
    t_end = timestamps[min(word_end, max_idx)].end + pad_s

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(source_path),
            "-ss", f"{t_start:.3f}",
            "-to", f"{t_end:.3f}",
            "-c", "copy",
            str(dest_path),
        ],
        check=True,
        capture_output=True,
    )
    return dest_path


def _ref_slug(passage_ref: str) -> str:
    """Convert a passage ref to a safe filename component, e.g. 'john_3_16'."""
    return re.sub(r"[^a-z0-9]+", "_", passage_ref.lower()).strip("_")
