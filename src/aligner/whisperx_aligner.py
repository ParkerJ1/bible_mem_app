"""WhisperXAligner — forced alignment using WhisperX / wav2vec2.

WhisperX aligns a known transcript to an audio file using a wav2vec2
forced-alignment model.  The Session Manager uses the resulting word-level
timestamps to slice the full-verse audio into short segments without
ever re-generating or re-fetching audio.
"""

from pathlib import Path

import torch
import whisperx

from src.aligner.base import Aligner, WordTimestamp

# WhisperX loads audio at 16 kHz internally
_SAMPLE_RATE = 16_000
_LANGUAGE = "en"


class WhisperXAligner(Aligner):
    """Forced alignment via WhisperX (wav2vec2 back-end).

    The alignment model is loaded once at construction time and reused for
    every align() call.  CUDA is used automatically when available; otherwise
    the aligner falls back to CPU.

    Words that WhisperX cannot confidently place (no start/end returned) are
    omitted from the result rather than being returned with sentinel values.
    """

    def __init__(self) -> None:
        """Load the WhisperX alignment model for English."""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._metadata = whisperx.load_align_model(
            language_code=_LANGUAGE,
            device=self._device,
        )

    def align(self, audio_path: Path, transcript: str) -> list[WordTimestamp]:
        """Return word-level timestamps for every alignable word in transcript.

        Constructs a single segment spanning the full audio duration and runs
        WhisperX forced alignment.  Words that lack start or end times in the
        alignment output are silently dropped.

        Args:
            audio_path: Path to the full-verse audio file.
            transcript: The verse text exactly as spoken in the audio.

        Returns:
            Ordered list of WordTimestamp; one entry per aligned word.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not transcript.strip():
            return []

        audio = whisperx.load_audio(str(audio_path))
        duration = len(audio) / _SAMPLE_RATE

        # WhisperX forced alignment: provide the transcript as a pre-formed
        # segment so the model aligns the known text rather than transcribing.
        segments = [{"text": transcript, "start": 0.0, "end": duration}]

        result = whisperx.align(
            segments,
            self._model,
            self._metadata,
            audio,
            self._device,
            return_char_alignments=False,
        )

        return _extract_timestamps(result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_timestamps(result: dict) -> list[WordTimestamp]:
    """Pull word-level timestamps out of a WhisperX alignment result dict.

    Words missing start or end keys are omitted — this can happen when
    wav2vec2 cannot confidently place a word (e.g. very quiet audio or
    uncommon proper nouns).
    """
    word_segments = result.get("word_segments", [])
    timestamps: list[WordTimestamp] = []

    for w in word_segments:
        start = w.get("start")
        end = w.get("end")
        word = w.get("word", "").strip()

        if word and start is not None and end is not None:
            timestamps.append(WordTimestamp(word=word, start=float(start), end=float(end)))

    return timestamps
