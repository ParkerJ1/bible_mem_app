from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WordTimestamp:
    """The alignment result for a single word."""

    word: str
    start: float  # seconds from the beginning of the audio
    end: float    # seconds from the beginning of the audio


class Aligner(ABC):
    """Abstract base class for forced-alignment providers.

    An Aligner takes a full-verse audio file and its transcript and returns
    word-level timestamps. It is core infrastructure used by every audio path
    (TTS-generated or ESV recorded). The Session Manager uses these timestamps
    to slice the audio into segments of the appropriate proficiency-level
    length; the Aligner itself knows nothing about segment lengths.
    """

    @abstractmethod
    def align(self, audio_path: Path, transcript: str) -> list[WordTimestamp]:
        """Align transcript words to their positions in the audio file.

        Args:
            audio_path: Path to the full-verse audio file (MP3, WAV, etc.).
            transcript: The full text of the verse as it is spoken in the
                        audio. Word order must match the audio.

        Returns:
            An ordered list of WordTimestamp objects, one per word in the
            transcript. Words that cannot be aligned are omitted.
        """
