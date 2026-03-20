from abc import ABC, abstractmethod

from src.shared import Verse


class TextProvider(ABC):
    """Abstract base class for Bible text providers."""

    @abstractmethod
    def get_verses(
        self,
        book: str,
        chapter: int,
        verse_start: int,
        verse_end: int = None,
    ) -> list[Verse]:
        """Fetch one or more verses from the provider.

        Given a book name, chapter number, and an inclusive verse range,
        return an ordered list of Verse objects. Implementations are
        responsible for fetching, caching, and returning the text.
        """
