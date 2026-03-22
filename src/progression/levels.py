"""Proficiency levels for the progression engine."""

from enum import IntEnum


class Level(IntEnum):
    """Ordered proficiency levels, from easiest to hardest.

    The integer value reflects the rank — higher is more advanced.
    Levels map directly to segmenter target word counts, with the exception
    of FULL_VERSE, FULL_PASSAGE (target=-1), and REFERENCE_ONLY (special).
    """

    WORDS_3 = 0
    WORDS_5 = 1
    WORDS_8 = 2
    WORDS_12 = 3
    FULL_VERSE = 4
    FULL_PASSAGE = 5
    REFERENCE_ONLY = 6


# Maps each level to the segmenter target word count.
# -1 means "whole text" (passed to PunctuationSegmenter as target=-1).
# None means the session handles it specially (no segmentation needed).
LEVEL_TARGET_WORDS: dict[Level, int | None] = {
    Level.WORDS_3: 3,
    Level.WORDS_5: 5,
    Level.WORDS_8: 8,
    Level.WORDS_12: 12,
    Level.FULL_VERSE: -1,
    Level.FULL_PASSAGE: -1,
    Level.REFERENCE_ONLY: None,
}


def advance(level: Level) -> Level:
    """Return the next level up, or the same level if already at the top."""
    next_value = level.value + 1
    if next_value > Level.REFERENCE_ONLY.value:
        return level
    return Level(next_value)


def drop(level: Level) -> Level:
    """Return the next level down, or the same level if already at the bottom."""
    prev_value = level.value - 1
    if prev_value < Level.WORDS_3.value:
        return level
    return Level(prev_value)
