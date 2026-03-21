import re
from abc import ABC, abstractmethod


class Segmenter(ABC):
    """Abstract base class for text segmenters."""

    @abstractmethod
    def segment(self, text: str, target: int) -> list[str]:
        """Segment text into chunks of approximately target words.

        Args:
            text: The input text to segment.
            target: The desired number of words per segment.
                    Pass -1 to return the entire text as a single segment.

        Returns:
            An ordered list of text segments.
        """


class WordSegmenter(Segmenter):
    """Segments text by strict word count.

    Each segment contains exactly target words, except the last. If the final
    segment would be shorter than half the target word count, it is merged into
    the preceding segment instead.
    """

    def segment(self, text: str, target: int) -> list[str]:
        """Divide text into fixed-size word chunks, merging a short tail segment.

        Pass target=-1 to return the entire text as a single segment.
        """
        words = text.split()
        if not words:
            return []

        if target == -1:
            return [" ".join(words)]

        chunks: list[list[str]] = []
        for i in range(0, len(words), target):
            chunks.append(words[i : i + target])

        # Merge trailing segment if it is less than half the target length
        if len(chunks) > 1 and len(chunks[-1]) < target / 2:
            chunks[-2].extend(chunks.pop())

        return [" ".join(chunk) for chunk in chunks]


def _ends_with_pause_punctuation(word: str) -> bool:
    """Return True if the word ends with a mid-sentence pause punctuation mark.

    Recognised marks: comma, semicolon, colon, hyphen, en-dash, em-dash.
    """
    return bool(re.search(r"[,;:\-\u2013\u2014]$", word))


class PunctuationSegmenter(Segmenter):
    """Segments text by word count, preferring natural punctuation break points.

    Rules applied in order for each segment boundary:
    1. Target word count with a tolerance of ±2.
    2. Prefer a punctuation boundary (comma, semicolon, colon, dash) within
       that window, choosing the latest one to maximise segment length.
    3. Fall back to the exact target word count if no punctuation is found.
    4. The final segment returns whatever words remain, regardless of length.
    """

    _TOLERANCE = 2

    def segment(self, text: str, target: int) -> list[str]:
        """Segment text guided by punctuation boundaries near the target count.

        Pass target=-1 to return the entire text as a single segment.
        """
        words = text.split()
        if not words:
            return []

        if target == -1:
            return [" ".join(words)]

        segments: list[str] = []
        i = 0

        while i < len(words):
            remaining = words[i:]

            # Last segment: take everything that is left
            if len(remaining) <= target:
                segments.append(" ".join(remaining))
                break

            # Search window: [target - tolerance, target + tolerance]
            low = max(1, target - self._TOLERANCE)
            high = min(target + self._TOLERANCE, len(remaining) - 1)

            # Scan from the top of the window downward; take the latest hit
            cut: int | None = None
            for j in range(high, low - 1, -1):
                if _ends_with_pause_punctuation(remaining[j]):
                    cut = j + 1  # exclusive upper bound
                    break

            if cut is None:
                cut = target

            segments.append(" ".join(remaining[:cut]))
            i += cut

        return segments
