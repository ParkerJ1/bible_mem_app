"""Tests for SessionManager.

All external dependencies (TextProvider, TTSProvider, Aligner, SpeechRecogniser,
Scorer) are replaced with lightweight fakes.  The progression engine and database
use real in-memory implementations so the interaction between those layers is
also exercised.
"""

import json
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.aligner.base import WordTimestamp
from src.data.models import Base, PreparedPassage, SessionRecord
from src.progression.engine import DefaultProgressionEngine
from src.progression.levels import Level
from src.scorer.base import DiffToken, ScorerResult
from src.session.session_manager import (
    SessionManager,
    SessionResult,
    _deserialise_timestamps,
    _parse_passage_ref,
    _ref_slug,
    _segment_word_range,
    _slice_audio,
)
from src.shared import Verse
from src.sr.base import TranscriptResult


# ---------------------------------------------------------------------------
# Fakes and fixtures
# ---------------------------------------------------------------------------


def _make_verse(text: str, verse_num: int = 16) -> Verse:
    return Verse(book="John", chapter=3, verse=verse_num, text=text)


class FakeTextProvider:
    """Returns a configurable list of verses."""

    def __init__(self, verses: list[Verse]) -> None:
        self._verses = verses

    def get_verses(self, book, chapter, verse_start, verse_end=None):
        return self._verses


class FakeTTSProvider:
    """Writes empty bytes and returns a fixed path."""

    def __init__(self, audio_path: Path) -> None:
        self._path = audio_path
        self._path.write_bytes(b"FAKE_AUDIO")

    def synthesise(self, text: str) -> Path:
        return self._path


class FakeAligner:
    """Returns a configurable list of WordTimestamps."""

    def __init__(self, timestamps: list[WordTimestamp]) -> None:
        self._timestamps = timestamps

    def align(self, audio_path: Path, transcript: str) -> list[WordTimestamp]:
        return self._timestamps


class FakeRecogniser:
    """Returns a fixed transcript."""

    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    def recognise(self, audio_path: Path) -> TranscriptResult:
        return TranscriptResult(text=self._transcript)


class FakeScorer:
    """Returns a configurable ScorerResult."""

    def __init__(self, result: ScorerResult) -> None:
        self._result = result

    def score(self, expected: str, got: str) -> ScorerResult:
        return self._result


def _make_timestamps(words: list[str]) -> list[WordTimestamp]:
    """Build evenly-spaced 1-second-per-word timestamps."""
    return [
        WordTimestamp(word=w, start=float(i), end=float(i + 1))
        for i, w in enumerate(words)
    ]


@pytest.fixture()
def tmp_audio(tmp_path):
    """A temporary MP3 file with fake audio bytes."""
    p = tmp_path / "passage.mp3"
    p.write_bytes(b"FAKE_AUDIO")
    return p


@pytest.fixture()
def db_factory(tmp_path):
    """Returns a get_session-compatible context-manager factory backed by SQLite."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    @contextmanager
    def factory():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return factory


def _make_manager(
    tmp_path,
    db_factory,
    *,
    verse_text: str = "For God so loved the world that he gave his only Son",
    transcript: str = "For God so loved the world",
    score: float = 1.0,
    diff: list[DiffToken] | None = None,
    extra_verses: list[Verse] | None = None,
) -> SessionManager:
    """Build a SessionManager with controllable fake dependencies."""
    audio_path = tmp_path / "passage.mp3"
    audio_path.write_bytes(b"FAKE_AUDIO")

    words = verse_text.split()
    timestamps = _make_timestamps(words)

    verses = extra_verses or [_make_verse(verse_text)]

    return SessionManager(
        text_provider=FakeTextProvider(verses),
        tts_provider=FakeTTSProvider(audio_path),
        aligner=FakeAligner(timestamps),
        recogniser=FakeRecogniser(transcript),
        scorer=FakeScorer(ScorerResult(score=score, expected=verse_text, got=transcript, diff=diff or [])),
        progression=DefaultProgressionEngine(pass_threshold=0.9, drop_days=3),
        db_factory=db_factory,
        segments_dir=tmp_path / "segments",
    )


# ---------------------------------------------------------------------------
# _parse_passage_ref
# ---------------------------------------------------------------------------


class TestParsePassageRef:
    def test_single_verse(self):
        assert _parse_passage_ref("John 3:16") == ("John", 3, 16, 16)

    def test_verse_range(self):
        assert _parse_passage_ref("Romans 8:28-30") == ("Romans", 8, 28, 30)

    def test_multi_word_book(self):
        assert _parse_passage_ref("1 Corinthians 13:4-7") == ("1 Corinthians", 13, 4, 7)

    def test_trailing_whitespace_stripped(self):
        assert _parse_passage_ref("  Psalm 23:1  ") == ("Psalm", 23, 1, 1)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_passage_ref("not a reference")


# ---------------------------------------------------------------------------
# _segment_word_range
# ---------------------------------------------------------------------------


class TestSegmentWordRange:
    def test_first_segment(self):
        segs = ["For God so", "loved the world"]
        assert _segment_word_range(segs, 0) == (0, 2)

    def test_second_segment(self):
        segs = ["For God so", "loved the world"]
        assert _segment_word_range(segs, 1) == (3, 5)

    def test_single_segment(self):
        segs = ["For God so loved the world"]
        assert _segment_word_range(segs, 0) == (0, 5)


# ---------------------------------------------------------------------------
# _deserialise_timestamps
# ---------------------------------------------------------------------------


class TestDeserialiseTimestamps:
    def test_round_trips(self):
        original = [WordTimestamp("For", 0.0, 1.0), WordTimestamp("God", 1.0, 2.0)]
        serialised = json.dumps(
            [{"word": t.word, "start": t.start, "end": t.end} for t in original]
        )
        restored = _deserialise_timestamps(serialised)
        assert restored == original

    def test_empty_list(self):
        assert _deserialise_timestamps("[]") == []


# ---------------------------------------------------------------------------
# _ref_slug
# ---------------------------------------------------------------------------


class TestRefSlug:
    def test_basic(self):
        assert _ref_slug("John 3:16") == "john_3_16"

    def test_range(self):
        assert _ref_slug("Romans 8:28-30") == "romans_8_28_30"


# ---------------------------------------------------------------------------
# prepare_verse
# ---------------------------------------------------------------------------


class TestPrepareVerse:
    def test_stores_passage_in_db(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory, verse_text="For God so loved the world")
        mgr.prepare_verse("John 3:16")

        with db_factory() as session:
            pp = session.query(PreparedPassage).filter_by(passage_ref="John 3:16").one()
            assert pp.full_text == "For God so loved the world"
            assert Path(pp.audio_path).exists()
            ts = _deserialise_timestamps(pp.timestamps_json)
            assert len(ts) == 6  # one per word

    def test_idempotent(self, tmp_path, db_factory):
        """Calling prepare_verse twice must not raise or create duplicate rows."""
        mgr = _make_manager(tmp_path, db_factory, verse_text="For God so loved the world")
        mgr.prepare_verse("John 3:16")
        mgr.prepare_verse("John 3:16")  # second call should be a no-op

        with db_factory() as session:
            count = session.query(PreparedPassage).filter_by(passage_ref="John 3:16").count()
            assert count == 1

    def test_multi_verse_passage_joins_text(self, tmp_path, db_factory):
        verses = [
            Verse("John", 3, 16, "For God so loved the world,"),
            Verse("John", 3, 17, "that he gave his only Son."),
        ]
        mgr = _make_manager(tmp_path, db_factory, extra_verses=verses)
        mgr.prepare_verse("John 3:16-17")

        with db_factory() as session:
            pp = session.query(PreparedPassage).filter_by(passage_ref="John 3:16-17").one()
            assert "For God so loved" in pp.full_text
            assert "that he gave his only Son" in pp.full_text

    def test_calls_tts_and_aligner(self, tmp_path, db_factory):
        audio_path = tmp_path / "passage.mp3"
        audio_path.write_bytes(b"FAKE")

        tts = MagicMock()
        tts.synthesise.return_value = audio_path

        aligner = MagicMock()
        aligner.align.return_value = [WordTimestamp("For", 0.0, 1.0)]

        mgr = SessionManager(
            text_provider=FakeTextProvider([_make_verse("For God")]),
            tts_provider=tts,
            aligner=aligner,
            recogniser=FakeRecogniser("For God"),
            scorer=FakeScorer(ScorerResult(score=1.0, expected="For God", got="For God")),
            progression=DefaultProgressionEngine(),
            db_factory=db_factory,
            segments_dir=tmp_path / "segments",
        )
        mgr.prepare_verse("John 3:16")

        tts.synthesise.assert_called_once_with("For God")
        aligner.align.assert_called_once()


# ---------------------------------------------------------------------------
# get_segments
# ---------------------------------------------------------------------------


class TestGetSegments:
    def test_returns_segments_at_words_3_level(self, tmp_path, db_factory):
        """Default level is WORDS_3; passage splits into ~3-word chunks."""
        text = "For God so loved the world that he gave his only Son"
        mgr = _make_manager(tmp_path, db_factory, verse_text=text)
        mgr.prepare_verse("John 3:16")

        segments = mgr.get_segments(user_id=1, passage_ref="John 3:16")

        assert len(segments) > 1
        for seg in segments:
            assert len(seg.split()) <= 5  # tolerance of ±2 above target of 3

    def test_reference_only_returns_passage_ref(self, tmp_path, db_factory):
        """At REFERENCE_ONLY level, the 'segment' is the passage reference itself."""
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")

        # Advance progression to REFERENCE_ONLY (level 6) manually
        engine = DefaultProgressionEngine(pass_threshold=0.0, drop_days=999)
        engine._states[(1, "John 3:16")] = _state_at_level(Level.REFERENCE_ONLY)
        mgr._progression = engine

        segments = mgr.get_segments(user_id=1, passage_ref="John 3:16")
        assert segments == ["John 3:16"]

    def test_full_verse_level_returns_one_segment(self, tmp_path, db_factory):
        """At FULL_VERSE level, the entire passage is a single segment."""
        text = "For God so loved the world that he gave his only Son"
        mgr = _make_manager(tmp_path, db_factory, verse_text=text)
        mgr.prepare_verse("John 3:16")

        engine = DefaultProgressionEngine()
        engine._states[(1, "John 3:16")] = _state_at_level(Level.FULL_VERSE)
        mgr._progression = engine

        segments = mgr.get_segments(user_id=1, passage_ref="John 3:16")
        assert len(segments) == 1
        assert segments[0] == text


# ---------------------------------------------------------------------------
# get_segment_audio
# ---------------------------------------------------------------------------


class TestGetSegmentAudio:
    def test_returns_none_at_reference_only_level(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")

        engine = DefaultProgressionEngine()
        engine._states[(1, "John 3:16")] = _state_at_level(Level.REFERENCE_ONLY)
        mgr._progression = engine

        result = mgr.get_segment_audio(user_id=1, passage_ref="John 3:16", segment_idx=0)
        assert result is None

    def test_raises_for_out_of_range_segment(self, tmp_path, db_factory):
        text = "For God so loved the world"
        mgr = _make_manager(tmp_path, db_factory, verse_text=text)
        mgr.prepare_verse("John 3:16")

        with patch("src.session.session_manager._slice_audio") as mock_slice:
            mock_slice.return_value = tmp_path / "seg.mp3"
            with pytest.raises(IndexError, match="out of range"):
                mgr.get_segment_audio(user_id=1, passage_ref="John 3:16", segment_idx=999)

    def test_calls_slice_audio_with_correct_word_range(self, tmp_path, db_factory):
        """Verify that _slice_audio receives the right word indices for the segment."""
        text = "For God so loved the world"
        mgr = _make_manager(tmp_path, db_factory, verse_text=text)
        mgr.prepare_verse("John 3:16")

        captured = {}

        def fake_slice(source, timestamps, word_start, word_end, dest):
            captured["word_start"] = word_start
            captured["word_end"] = word_end
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"SLICED")
            return dest

        with patch("src.session.session_manager._slice_audio", side_effect=fake_slice):
            mgr.get_segment_audio(user_id=1, passage_ref="John 3:16", segment_idx=0)

        assert captured["word_start"] == 0
        assert captured["word_end"] >= 0

    def test_segment_audio_cached_on_second_call(self, tmp_path, db_factory):
        """_slice_audio must not be called a second time when the dest file exists.

        The caching guard lives in get_segment_audio (above the _slice_audio call).
        """
        text = "For God so loved the world"
        mgr = _make_manager(tmp_path, db_factory, verse_text=text)
        mgr.prepare_verse("John 3:16")

        call_count = {"n": 0}

        def fake_slice(source, timestamps, word_start, word_end, dest):
            call_count["n"] += 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"SLICED")
            return dest

        with patch("src.session.session_manager._slice_audio", side_effect=fake_slice):
            mgr.get_segment_audio(user_id=1, passage_ref="John 3:16", segment_idx=0)
            mgr.get_segment_audio(user_id=1, passage_ref="John 3:16", segment_idx=0)

        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# score_attempt
# ---------------------------------------------------------------------------


class TestScoreAttempt:
    def test_returns_segment_result_with_correct_index(self, tmp_path, db_factory):
        mgr = _make_manager(
            tmp_path, db_factory,
            verse_text="For God so loved the world",
            transcript="For God so loved the world",
            score=1.0,
        )
        mgr.prepare_verse("John 3:16")

        result = mgr.score_attempt(
            user_id=1,
            passage_ref="John 3:16",
            segment_idx=0,
            user_audio_path=tmp_path / "user.wav",
        )

        assert result.segment_idx == 0
        assert result.score == pytest.approx(1.0)
        assert result.expected != ""

    def test_uses_full_text_at_reference_only_level(self, tmp_path, db_factory):
        """At REFERENCE_ONLY the expected text is the full passage, not a segment."""
        text = "For God so loved the world"
        mgr = _make_manager(tmp_path, db_factory, verse_text=text, score=0.8)
        mgr.prepare_verse("John 3:16")

        engine = DefaultProgressionEngine()
        engine._states[(1, "John 3:16")] = _state_at_level(Level.REFERENCE_ONLY)
        mgr._progression = engine

        result = mgr.score_attempt(
            user_id=1,
            passage_ref="John 3:16",
            segment_idx=0,
            user_audio_path=tmp_path / "user.wav",
        )

        assert result.expected == text

    def test_diff_is_forwarded(self, tmp_path, db_factory):
        diff = [DiffToken("For", "correct"), DiffToken("God", "missing")]
        mgr = _make_manager(tmp_path, db_factory, score=0.5, diff=diff)
        mgr.prepare_verse("John 3:16")

        result = mgr.score_attempt(
            user_id=1,
            passage_ref="John 3:16",
            segment_idx=0,
            user_audio_path=tmp_path / "user.wav",
        )

        assert result.diff == diff


# ---------------------------------------------------------------------------
# finish_session
# ---------------------------------------------------------------------------


class TestFinishSession:
    def _segment_results(self, scores: list[float]) -> list:
        from src.session.session_manager import SegmentResult
        return [
            SegmentResult(segment_idx=i, expected="text", score=s)
            for i, s in enumerate(scores)
        ]

    def test_overall_score_is_mean_of_segments(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")

        result = mgr.finish_session(
            user_id=1,
            passage_ref="John 3:16",
            segment_results=self._segment_results([1.0, 0.5, 0.75]),
            attempt_date=date(2026, 3, 22),
        )

        assert result.overall_score == pytest.approx(0.75, abs=1e-4)

    def test_session_record_written_to_db(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")

        mgr.finish_session(
            user_id=1,
            passage_ref="John 3:16",
            segment_results=self._segment_results([1.0]),
            attempt_date=date(2026, 3, 22),
        )

        with db_factory() as session:
            rec = session.query(SessionRecord).filter_by(verse_ref="John 3:16").one()
            assert rec.score == pytest.approx(1.0)
            assert rec.attempt_date == date(2026, 3, 22)

    def test_level_advances_after_two_consecutive_passes(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")

        mgr.finish_session(
            user_id=1,
            passage_ref="John 3:16",
            segment_results=self._segment_results([1.0]),
            attempt_date=date(2026, 3, 21),
        )
        result = mgr.finish_session(
            user_id=1,
            passage_ref="John 3:16",
            segment_results=self._segment_results([1.0]),
            attempt_date=date(2026, 3, 22),
        )

        assert result.passed is True
        assert result.level_after > result.level_before

    def test_level_does_not_advance_on_single_pass(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")

        result = mgr.finish_session(
            user_id=1,
            passage_ref="John 3:16",
            segment_results=self._segment_results([1.0]),
            attempt_date=date(2026, 3, 22),
        )

        assert result.level_before == result.level_after

    def test_level_drops_after_three_consecutive_fails(self, tmp_path, db_factory):
        """Three consecutive daily failures should drop the level."""
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")

        # Advance to level 1 first via two consecutive passes
        mgr.finish_session(
            user_id=1, passage_ref="John 3:16",
            segment_results=self._segment_results([1.0]),
            attempt_date=date(2026, 3, 19),
        )
        mgr.finish_session(
            user_id=1, passage_ref="John 3:16",
            segment_results=self._segment_results([1.0]),
            attempt_date=date(2026, 3, 20),
        )

        # Three consecutive failures
        for day in [21, 22, 23]:
            result = mgr.finish_session(
                user_id=1, passage_ref="John 3:16",
                segment_results=self._segment_results([0.0]),
                attempt_date=date(2026, 3, day),
            )

        assert result.level_after < result.level_before

    def test_raises_on_empty_segment_results(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory)
        with pytest.raises(ValueError, match="empty"):
            mgr.finish_session(
                user_id=1, passage_ref="John 3:16", segment_results=[]
            )

    def test_session_result_contains_all_segment_results(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")
        segs = self._segment_results([0.8, 0.9])

        result = mgr.finish_session(
            user_id=1, passage_ref="John 3:16",
            segment_results=segs,
            attempt_date=date(2026, 3, 22),
        )

        assert len(result.segment_results) == 2
        assert result.passage_ref == "John 3:16"

    def test_multiple_sessions_recorded(self, tmp_path, db_factory):
        mgr = _make_manager(tmp_path, db_factory)
        mgr.prepare_verse("John 3:16")

        for day in [20, 21, 22]:
            mgr.finish_session(
                user_id=1, passage_ref="John 3:16",
                segment_results=self._segment_results([0.9]),
                attempt_date=date(2026, 3, day),
            )

        with db_factory() as session:
            count = session.query(SessionRecord).filter_by(verse_ref="John 3:16").count()
            assert count == 3


# ---------------------------------------------------------------------------
# _slice_audio (unit-tested with pydub mocked)
# ---------------------------------------------------------------------------


class TestSliceAudio:
    def test_returns_dest_immediately_if_exists(self, tmp_path):
        dest = tmp_path / "seg.mp3"
        dest.write_bytes(b"CACHED")
        source = tmp_path / "full.mp3"
        timestamps = _make_timestamps(["For", "God"])

        result = _slice_audio(source, timestamps, 0, 1, dest)

        assert result == dest
        # source was never read — file doesn't even need to exist

    def test_raises_when_no_timestamps(self, tmp_path):
        dest = tmp_path / "seg.mp3"
        source = tmp_path / "full.mp3"
        source.write_bytes(b"AUDIO")

        with pytest.raises(ValueError, match="no timestamps"):
            _slice_audio(source, [], 0, 0, dest)

    def test_slices_correct_range(self, tmp_path):
        """Verify ffmpeg is called with the correct -ss/-to timestamps."""
        source = tmp_path / "full.mp3"
        source.write_bytes(b"AUDIO")
        dest = tmp_path / "seg.mp3"

        # 10 words, 1 second each: w0=0–1s, w1=1–2s, …, w9=9–10s
        timestamps = _make_timestamps([f"w{i}" for i in range(10)])

        with patch("src.session.session_manager.subprocess.run") as mock_run:
            _slice_audio(source, timestamps, 2, 4, dest)

        args = mock_run.call_args[0][0]  # the command list
        ss = float(args[args.index("-ss") + 1])
        to = float(args[args.index("-to") + 1])

        pad = 0.150  # _SEGMENT_PAD_MS / 1000
        assert ss == pytest.approx(max(0.0, 2.0 - pad), abs=1e-3)  # word 2 start − pad
        assert to == pytest.approx(5.0 + pad, abs=1e-3)             # word 4 end + pad

    def test_index_clamped_when_timestamps_shorter_than_expected(self, tmp_path):
        """If word_end exceeds timestamp list length, use the last timestamp."""
        source = tmp_path / "full.mp3"
        source.write_bytes(b"AUDIO")
        dest = tmp_path / "seg.mp3"

        timestamps = _make_timestamps(["For", "God"])  # only 2 words

        with patch("src.session.session_manager.subprocess.run") as mock_run:
            # word_end=99 exceeds timestamps — must not raise
            _slice_audio(source, timestamps, 0, 99, dest)

        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Helpers used by multiple test classes
# ---------------------------------------------------------------------------


def _state_at_level(level: Level):
    """Return a _DayState with the given level and reset streaks."""
    from src.progression.engine import _DayState
    return _DayState(level=level, pass_streak=0, fail_streak=0)
