from dataclasses import dataclass


@dataclass
class Verse:
    """A single Bible verse with its reference and text."""

    book: str
    chapter: int
    verse: int
    text: str
