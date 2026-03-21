"""Tests for SequenceAligner scorer.

Covers: normalisation, strict mode, lenient mode, diff token statuses,
edge cases, and the ScorerResult structure.
"""

import pytest

from src.scorer.base import DiffToken, Scorer, ScorerResult
from src.scorer.sequence_aligner import SequenceAligner, _lemmatize, _normalise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strict() -> SequenceAligner:
    return SequenceAligner(mode="strict")


@pytest.fixture
def lenient() -> SequenceAligner:
    return SequenceAligner(mode="lenient")


# ---------------------------------------------------------------------------
# _normalise helper
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_lowercases(self) -> None:
        assert _normalise("For GOD So Loved") == "for god so loved"

    def test_strips_punctuation(self) -> None:
        assert _normalise("hello, world!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert _normalise("  too   many   spaces  ") == "too many spaces"

    def test_strips_common_verse_punctuation(self) -> None:
        result = _normalise("Grace, mercy; and peace: from God—our Father.")
        assert "," not in result
        assert ";" not in result
        assert "." not in result

    def test_empty_string(self) -> None:
        assert _normalise("") == ""


# ---------------------------------------------------------------------------
# _lemmatize helper
# ---------------------------------------------------------------------------

class TestLemmatize:
    def test_verb_tense_loved(self) -> None:
        assert _lemmatize("loved") == "love"

    def test_verb_tense_created(self) -> None:
        assert _lemmatize("created") == "create"

    def test_plural_noun_worlds(self) -> None:
        assert _lemmatize("worlds") == "world"

    def test_base_form_unchanged(self) -> None:
        assert _lemmatize("love") == "love"

    def test_common_word_unchanged(self) -> None:
        assert _lemmatize("god") == "god"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_implements_scorer_abc(self, strict: SequenceAligner) -> None:
        assert isinstance(strict, Scorer)

    def test_mode_strict(self, strict: SequenceAligner) -> None:
        assert strict.mode == "strict"

    def test_mode_lenient(self, lenient: SequenceAligner) -> None:
        assert lenient.mode == "lenient"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scoring mode"):
            SequenceAligner(mode="fuzzy")

    def test_reads_config_when_no_mode_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.config.SCORING_MODE", "lenient")
        sa = SequenceAligner()
        assert sa.mode == "lenient"


# ---------------------------------------------------------------------------
# ScorerResult structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_returns_scorer_result(self, strict: SequenceAligner) -> None:
        result = strict.score("for god so loved the world", "for god so loved the world")
        assert isinstance(result, ScorerResult)

    def test_result_has_all_fields(self, strict: SequenceAligner) -> None:
        result = strict.score("hello world", "hello world")
        assert hasattr(result, "score")
        assert hasattr(result, "expected")
        assert hasattr(result, "got")
        assert hasattr(result, "diff")

    def test_expected_and_got_are_normalised(self, strict: SequenceAligner) -> None:
        result = strict.score("Hello, World!", "HELLO WORLD")
        assert result.expected == "hello world"
        assert result.got == "hello world"

    def test_diff_is_list_of_diff_tokens(self, strict: SequenceAligner) -> None:
        result = strict.score("hello world", "hello world")
        assert isinstance(result.diff, list)
        assert all(isinstance(t, DiffToken) for t in result.diff)

    def test_score_rounded_to_4dp(self, strict: SequenceAligner) -> None:
        result = strict.score("one two three", "one two")
        assert result.score == round(result.score, 4)


# ---------------------------------------------------------------------------
# Strict mode — scoring
# ---------------------------------------------------------------------------

class TestStrictScoring:
    def test_perfect_match_scores_1(self, strict: SequenceAligner) -> None:
        result = strict.score(
            "For God so loved the world",
            "For God so loved the world",
        )
        assert result.score == pytest.approx(1.0)

    def test_completely_wrong_scores_0(self, strict: SequenceAligner) -> None:
        result = strict.score("alpha beta gamma", "one two three")
        assert result.score == pytest.approx(0.0)

    def test_half_correct(self, strict: SequenceAligner) -> None:
        # "one two" correct out of "one two three four"
        result = strict.score("one two three four", "one two")
        assert result.score == pytest.approx(0.5)

    def test_extra_inserted_words_do_not_lower_score(self, strict: SequenceAligner) -> None:
        # User said correct words plus extras; score should reflect expected words only
        result = strict.score("God loved the world", "God really loved the world so much")
        assert result.score == pytest.approx(1.0)

    def test_tense_difference_penalised_in_strict(self, strict: SequenceAligner) -> None:
        result = strict.score("God loved the world", "God love the world")
        assert result.score < 1.0

    def test_plural_difference_penalised_in_strict(self, strict: SequenceAligner) -> None:
        result = strict.score("the heavens and the earth", "the heaven and the earth")
        assert result.score < 1.0

    def test_empty_expected_returns_1_when_got_also_empty(self, strict: SequenceAligner) -> None:
        result = strict.score("", "")
        assert result.score == pytest.approx(1.0)

    def test_empty_got_scores_0(self, strict: SequenceAligner) -> None:
        result = strict.score("God loved the world", "")
        assert result.score == pytest.approx(0.0)

    def test_score_between_0_and_1(self, strict: SequenceAligner) -> None:
        result = strict.score("one two three four five", "one two four")
        assert 0.0 <= result.score <= 1.0

    def test_mid_sentence_insertion_does_not_wreck_later_words(self, strict: SequenceAligner) -> None:
        """LCS alignment: inserted word in middle must not mark later correct words as wrong."""
        result = strict.score(
            "the lord is my shepherd",
            "the lord definitely is my shepherd",
        )
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Lenient mode — scoring
# ---------------------------------------------------------------------------

class TestLenientScoring:
    def test_perfect_match_scores_1(self, lenient: SequenceAligner) -> None:
        result = lenient.score("God loved the world", "God loved the world")
        assert result.score == pytest.approx(1.0)

    def test_tense_difference_not_penalised(self, lenient: SequenceAligner) -> None:
        result = lenient.score("God loved the world", "God love the world")
        assert result.score == pytest.approx(1.0)

    def test_plural_difference_not_penalised(self, lenient: SequenceAligner) -> None:
        result = lenient.score("the heavens and the earth", "the heaven and the earth")
        assert result.score == pytest.approx(1.0)

    def test_completely_wrong_still_scores_0(self, lenient: SequenceAligner) -> None:
        result = lenient.score("alpha beta gamma", "one two three")
        assert result.score == pytest.approx(0.0)

    def test_strict_penalises_where_lenient_forgives(
        self, strict: SequenceAligner, lenient: SequenceAligner
    ) -> None:
        expected = "he creates all things"
        got = "he created all things"
        assert strict.score(expected, got).score < lenient.score(expected, got).score


# ---------------------------------------------------------------------------
# Diff token statuses
# ---------------------------------------------------------------------------

class TestDiffTokens:
    def test_all_correct(self, strict: SequenceAligner) -> None:
        result = strict.score("hello world", "hello world")
        statuses = [t.status for t in result.diff]
        assert all(s == "correct" for s in statuses)

    def test_missing_word_marked(self, strict: SequenceAligner) -> None:
        result = strict.score("hello beautiful world", "hello world")
        missing = [t.word for t in result.diff if t.status == "missing"]
        assert "beautiful" in missing

    def test_inserted_word_marked(self, strict: SequenceAligner) -> None:
        result = strict.score("hello world", "hello beautiful world")
        inserted = [t.word for t in result.diff if t.status == "inserted"]
        assert "beautiful" in inserted

    def test_diff_contains_only_valid_statuses(self, strict: SequenceAligner) -> None:
        result = strict.score("one two three", "one four three")
        valid = {"correct", "missing", "inserted"}
        assert all(t.status in valid for t in result.diff)

    def test_correct_words_in_diff_match_expected(self, strict: SequenceAligner) -> None:
        result = strict.score("for god so loved", "for god so loved")
        correct_words = [t.word for t in result.diff if t.status == "correct"]
        assert correct_words == ["for", "god", "so", "loved"]

    def test_empty_diff_for_empty_inputs(self, strict: SequenceAligner) -> None:
        result = strict.score("", "")
        assert result.diff == []

    def test_all_missing_when_got_is_empty(self, strict: SequenceAligner) -> None:
        result = strict.score("hello world", "")
        assert all(t.status == "missing" for t in result.diff)

    def test_all_inserted_when_expected_is_empty(self, strict: SequenceAligner) -> None:
        result = strict.score("", "hello world")
        assert all(t.status == "inserted" for t in result.diff)

    def test_middle_insertion_shows_as_inserted(self, strict: SequenceAligner) -> None:
        result = strict.score("the lord is my shepherd", "the lord definitely is my shepherd")
        inserted = [t.word for t in result.diff if t.status == "inserted"]
        assert "definitely" in inserted
        correct = [t.word for t in result.diff if t.status == "correct"]
        assert "shepherd" in correct

    def test_replace_emits_missing_then_inserted(self, strict: SequenceAligner) -> None:
        result = strict.score("hello world", "hello earth")
        missing = [t.word for t in result.diff if t.status == "missing"]
        inserted = [t.word for t in result.diff if t.status == "inserted"]
        assert "world" in missing
        assert "earth" in inserted

    def test_lenient_diff_uses_original_words_not_lemmas(self, lenient: SequenceAligner) -> None:
        """Even in lenient mode, diff tokens should show the original surface form."""
        result = lenient.score("God loved the world", "God love the world")
        correct_words = [t.word for t in result.diff if t.status == "correct"]
        # "loved" matched "love" via lemma — original form should appear
        assert "loved" in correct_words
