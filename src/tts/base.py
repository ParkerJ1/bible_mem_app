from abc import ABC, abstractmethod
from pathlib import Path


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers.

    Implementations must synthesise a full text string into a single audio
    file saved to disk and return the path to that file. Providers always
    operate on complete verse or passage text; segment slicing is handled
    downstream by the Session Manager.
    """

    @abstractmethod
    def synthesise(self, text: str) -> Path:
        """Convert text to speech and save the result as an audio file.

        Args:
            text: The full text to synthesise. Must be non-empty.

        Returns:
            Path to the saved audio file on disk.
        """
