from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptResult:
    """The output of a speech recognition pass."""

    text: str
    confidence: float = -1.0 # 0.0 (no confidence) – 1.0 (certain)


class SpeechRecogniser(ABC):
    """Abstract base class for speech recognition providers.

    Implementations accept a path to an audio file, run recognition,
    and return the transcript alongside a normalised confidence score.
    """

    @abstractmethod
    def recognise(self, audio_path: Path) -> TranscriptResult:
        """Transcribe the audio file at audio_path.

        Args:
            audio_path: Path to the audio file to transcribe (e.g. MP3, WAV).

        Returns:
            TranscriptResult containing the transcript text and a confidence
            score in the range [0.0, 1.0].
        """
