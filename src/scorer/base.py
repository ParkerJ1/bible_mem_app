from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


DiffStatus = Literal["correct", "missing", "inserted"]


@dataclass
class DiffToken:
    """A single token in the word-level diff between expected and got."""

    word: str
    status: DiffStatus


@dataclass
class ScorerResult:
    """The output of a scoring pass."""

    score: float                    # 0.0 – 1.0; fraction of expected words correct
    expected: str                   # normalised expected text
    got: str                        # normalised user transcript
    diff: list[DiffToken] = field(default_factory=list)


class Scorer(ABC):
    """Abstract base class for verse-recitation scorers.

    Implementations compare the canonical verse text against a user's
    transcribed attempt and return a structured result including a numeric
    score and a word-level diff.
    """

    @abstractmethod
    def score(self, expected: str, got: str) -> ScorerResult:
        """Compare expected verse text against the user's transcript.

        Args:
            expected: The canonical verse text.
            got:      The user's transcribed attempt.

        Returns:
            ScorerResult with score, normalised strings, and word-level diff.
        """
