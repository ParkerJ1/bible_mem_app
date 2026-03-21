"""Tests for WordSegmenter and PunctuationSegmenter."""

import pytest

from src.segmenter.segmenter import PunctuationSegmenter, WordSegmenter

# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------
# Each entry: (id, text)
# Deliberately varied: no punctuation, dense punctuation, sparse punctuation,
# short gaps between marks, long gaps between marks, and edge cases.

NO_PUNCT_SHORT = "In the beginning God created the heavens and the earth"
# 10 words, no punctuation

NO_PUNCT_LONG = (
    "The Lord is my shepherd I shall not want he makes me lie down "
    "in green pastures he leads me beside still waters he restores my soul"
)
# 28 words, no punctuation

DENSE_PUNCT = "Grace, mercy, and peace, from God our Father, and Christ Jesus our Lord."
# Commas every 2-3 words

SPARSE_PUNCT = (
    "For I know the plans I have for you declares the Lord "
    "plans to prosper you and not to harm you, "
    "plans to give you hope and a future."
)
# One comma near the middle, 34 words total

SHORT_GAPS = "Faith, hope, love; these three, but the greatest, is love."
# Punctuation every 1-3 words

LONG_GAPS_BETWEEN = (
    "In the beginning was the Word and the Word was with God "
    "and the Word was God; he was with God in the beginning "
    "through him all things were made and without him nothing was made "
    "that has been made."
)
# Semicolon roughly in the middle, ~42 words

COLON_AND_DASH = (
    "There are three things that endure: faith, hope, and love — "
    "and the greatest of these is love."
)
# Colon and em-dash as potential boundaries

SINGLE_WORD = "Amen."
# Edge case: one word

EXACT_TARGET = "For God so loved the world"
# 6 words — tests target matching exactly

TRAILING_SHORT = (
    "Blessed are the poor in spirit for theirs is the kingdom of heaven blessed are those"
)
# 17 words — tests WordSegmenter tail-merge with various targets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in text."""
    return len(text.split())


def rejoined(segments: list[str], original: str) -> bool:
    """Return True if joining segments reproduces the original word sequence."""
    return " ".join(" ".join(s.split()) for s in segments) == " ".join(original.split())


# ---------------------------------------------------------------------------
# WordSegmenter
# ---------------------------------------------------------------------------

class TestWordSegmenter:
    """Tests for WordSegmenter."""

    @pytest.fixture
    def seg(self) -> WordSegmenter:
        """Return a WordSegmenter instance."""
        return WordSegmenter()

    # --- target=-1 (whole string) ---

    def test_whole_string_no_punct(self, seg: WordSegmenter) -> None:
        result = seg.segment(NO_PUNCT_SHORT, -1)
        assert result == [NO_PUNCT_SHORT]

    def test_whole_string_with_punct(self, seg: WordSegmenter) -> None:
        result = seg.segment(DENSE_PUNCT, -1)
        assert len(result) == 1
        assert rejoined(result, DENSE_PUNCT)

    def test_whole_string_single_word(self, seg: WordSegmenter) -> None:
        result = seg.segment(SINGLE_WORD, -1)
        assert result == [SINGLE_WORD]

    # --- strict word counts ---

    def test_no_punct_target_3(self, seg: WordSegmenter) -> None:
        """10-word string with target=3: expect 3+3+4 (tail 1 < 1.5, merged)."""
        result = seg.segment(NO_PUNCT_SHORT, 3)
        assert rejoined(result, NO_PUNCT_SHORT)
        # tail would be 1 word (< 3/2=1.5) → merged into chunk 3
        for s in result[:-1]:
            assert word_count(s) == 3
        # last chunk carries the merged tail
        assert word_count(result[-1]) == 4

    def test_no_punct_target_5(self, seg: WordSegmenter) -> None:
        """28-word string with target=5: last chunk is 3 (≥ 2.5) so kept separate."""
        result = seg.segment(NO_PUNCT_LONG, 5)
        assert rejoined(result, NO_PUNCT_LONG)
        assert word_count(result[-1]) >= 3

    def test_exact_target_match(self, seg: WordSegmenter) -> None:
        """Text with exactly target words should return one segment."""
        result = seg.segment(EXACT_TARGET, 6)
        assert result == [EXACT_TARGET]

    def test_tail_merge_happens(self, seg: WordSegmenter) -> None:
        """16-word text with target=5: last chunk is 1 word (< 2.5) and gets merged."""
        result = seg.segment(TRAILING_SHORT, 5)
        assert rejoined(result, TRAILING_SHORT)
        # chunks before merge: 5,5,5,1 → last 1 merged into preceding 5 → 5,5,6
        assert word_count(result[-1]) == 6

    def test_tail_merge_does_not_happen(self, seg: WordSegmenter) -> None:
        """16-word text with target=6: last chunk is 4 words (≥ 3) so kept separate."""
        result = seg.segment(TRAILING_SHORT, 6)
        assert rejoined(result, TRAILING_SHORT)
        # 6,6,4 → 4 ≥ 3 (6/2), no merge
        assert len(result) == 3
        assert word_count(result[-1]) == 4

    def test_single_word_input(self, seg: WordSegmenter) -> None:
        result = seg.segment(SINGLE_WORD, 5)
        assert result == [SINGLE_WORD]

    def test_empty_string(self, seg: WordSegmenter) -> None:
        assert seg.segment("", 5) == []

    def test_punctuation_ignored_for_boundaries(self, seg: WordSegmenter) -> None:
        """WordSegmenter must ignore punctuation and cut purely on word count."""
        result = seg.segment(DENSE_PUNCT, 3)
        assert rejoined(result, DENSE_PUNCT)
        for s in result[:-1]:
            assert word_count(s) == 3


# ---------------------------------------------------------------------------
# PunctuationSegmenter
# ---------------------------------------------------------------------------

class TestPunctuationSegmenter:
    """Tests for PunctuationSegmenter."""

    @pytest.fixture
    def seg(self) -> PunctuationSegmenter:
        """Return a PunctuationSegmenter instance."""
        return PunctuationSegmenter()

    # --- target=-1 (whole string) ---

    def test_whole_string_no_punct(self, seg: PunctuationSegmenter) -> None:
        result = seg.segment(NO_PUNCT_LONG, -1)
        assert len(result) == 1
        assert rejoined(result, NO_PUNCT_LONG)

    def test_whole_string_dense_punct(self, seg: PunctuationSegmenter) -> None:
        result = seg.segment(DENSE_PUNCT, -1)
        assert result == [DENSE_PUNCT]

    # --- no punctuation: falls back to word count ---

    def test_no_punct_falls_back_to_target(self, seg: PunctuationSegmenter) -> None:
        """Without punctuation, each segment should be exactly target words."""
        result = seg.segment(NO_PUNCT_SHORT, 5)
        assert rejoined(result, NO_PUNCT_SHORT)
        assert word_count(result[0]) == 5

    def test_no_punct_long_target_8(self, seg: PunctuationSegmenter) -> None:
        result = seg.segment(NO_PUNCT_LONG, 8)
        assert rejoined(result, NO_PUNCT_LONG)
        # first segment must be exactly 8 (no punctuation to deviate)
        assert word_count(result[0]) == 8

    # --- punctuation guides the boundary ---

    def test_dense_punct_target_5_snaps_to_comma(self, seg: PunctuationSegmenter) -> None:
        """With commas every 2-3 words, target=5 should always snap to a comma."""
        result = seg.segment(DENSE_PUNCT, 5)
        assert rejoined(result, DENSE_PUNCT)
        for s in result[:-1]:
            # Each non-final segment should end with a punctuation-bearing word
            assert s.rstrip()[-1] in {",", ";", ":", "-", "\u2013", "\u2014"}

    def test_sparse_punct_target_5(self, seg: PunctuationSegmenter) -> None:
        """Sparse punctuation: only the segment that contains the comma snaps to it."""
        result = seg.segment(SPARSE_PUNCT, 5)
        assert rejoined(result, SPARSE_PUNCT)
        # At least one segment boundary should land on the comma
        comma_segments = [s for s in result if s.endswith(",")]
        assert len(comma_segments) >= 1

    def test_short_gaps_target_5(self, seg: PunctuationSegmenter) -> None:
        """Text with punctuation every 1-3 words: target=5 snaps to nearest mark."""
        result = seg.segment(SHORT_GAPS, 5)
        assert rejoined(result, SHORT_GAPS)

    def test_long_gaps_target_8_uses_semicolon(self, seg: PunctuationSegmenter) -> None:
        """Semicolon roughly mid-text should be used as a boundary near target=8."""
        result = seg.segment(LONG_GAPS_BETWEEN, 8)
        assert rejoined(result, LONG_GAPS_BETWEEN)
        semicolon_segments = [s for s in result if s.rstrip().endswith(";")]
        assert len(semicolon_segments) >= 1

    def test_colon_and_dash_target_6(self, seg: PunctuationSegmenter) -> None:
        """Colon and em-dash should both be valid snap points."""
        result = seg.segment(COLON_AND_DASH, 6)
        assert rejoined(result, COLON_AND_DASH)

    # --- last segment behaviour ---

    def test_last_segment_can_be_short(self, seg: PunctuationSegmenter) -> None:
        """Unlike WordSegmenter, PunctuationSegmenter never merges the last segment."""
        # 10 words, target=4 → 4, 4, 2 (last 2 kept as-is, no merge)
        result = seg.segment(NO_PUNCT_SHORT, 4)
        assert rejoined(result, NO_PUNCT_SHORT)
        assert len(result) == 3
        assert word_count(result[-1]) == 2

    def test_single_word_input(self, seg: PunctuationSegmenter) -> None:
        result = seg.segment(SINGLE_WORD, 5)
        assert result == [SINGLE_WORD]

    def test_empty_string(self, seg: PunctuationSegmenter) -> None:
        assert seg.segment("", 5) == []

    # --- segments cover entire input ---

    @pytest.mark.parametrize("text,target", [
        (NO_PUNCT_SHORT, 3),
        (NO_PUNCT_LONG, 6),
        (DENSE_PUNCT, 4),
        (SPARSE_PUNCT, 7),
        (LONG_GAPS_BETWEEN, 10),
        (COLON_AND_DASH, 5),
        (TRAILING_SHORT, 5),
    ])
    def test_no_words_lost(self, seg: PunctuationSegmenter, text: str, target: int) -> None:
        """Joining all segments must reproduce the original word sequence."""
        result = seg.segment(text, target)
        assert rejoined(result, text)
