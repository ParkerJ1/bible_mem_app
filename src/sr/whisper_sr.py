import math
import os
from pathlib import Path

import whisper
from dotenv import load_dotenv

from src.sr.base import SpeechRecogniser, TranscriptResult

load_dotenv()

_DEFAULT_MODEL = "base"


class WhisperSpeechRecogniser(SpeechRecogniser):
    """Speech recogniser backed by OpenAI Whisper, running fully on-device.

    The Whisper model size is controlled by the WHISPER_MODEL environment
    variable (default: "base"). Valid values are: tiny, base, small, medium,
    large.

    Confidence is derived from the per-segment avg_logprob values returned by
    Whisper. Each segment's log-probability is exponentiated and the scores
    are averaged across all segments, giving a value in (0.0, 1.0].
    A result with no segments (e.g. silence) returns confidence 0.0.
    """

    def __init__(self) -> None:
        """Load the Whisper model specified by WHISPER_MODEL."""
        model_size = os.getenv("WHISPER_MODEL", _DEFAULT_MODEL)
        self._model = whisper.load_model(model_size)

    def recognise(self, audio_path: Path) -> TranscriptResult:
        """Transcribe audio_path and return the transcript with a confidence score.

        Args:
            audio_path: Path to the audio file. Whisper accepts MP3, WAV,
                        M4A, and other common formats.

        Returns:
            TranscriptResult with stripped transcript text and a confidence
            score averaged over all recognised segments.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result = self._model.transcribe(str(audio_path))

        text = result["text"].strip()
        confidence = _mean_confidence(result.get("segments", []))

        return TranscriptResult(text=text, confidence=confidence)


def _mean_confidence(segments: list[dict]) -> float:
    """Convert Whisper segment avg_logprob values to a mean confidence score.

    avg_logprob is a negative log-probability; exp() maps it back to (0, 1].
    Returns 0.0 when there are no segments.
    """
    if not segments:
        return 0.0
    avg_logprob = sum(s["avg_logprob"] for s in segments) / len(segments)
    return float(min(1.0, math.exp(avg_logprob)))
