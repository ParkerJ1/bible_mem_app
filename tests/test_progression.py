"""Tests for the progression engine and level utilities."""

from datetime import date, timedelta

import pytest

from src.progression.engine import (
    AttemptOutcome,
    DefaultProgressionEngine,
    ProgressionEngine,
    _apply_attempt,
    _DayState,
)
from src.progression.levels import Level, LEVEL_TARGET_WORDS, advance, drop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine(pass_threshold: float = 0.9, drop_days: int = 3) -> DefaultProgressionEngine:
    return DefaultProgressionEngine(pass_threshold=pass_threshold, drop_days=drop_days)


def _day(offset: int = 0) -> date:
    """Return a fixed base date plus offset days (avoids dependency on real clock)."""
    return date(2026, 1, 1) + timedelta(days=offset)


PASS = 0.95
FAIL = 0.5
USER = 1
VERSE = "John 3:16"


# ---------------------------------------------------------------------------
# Level enum and utilities
# ---------------------------------------------------------------------------

class TestLevels:
    def test_levels_are_ordered(self) -> None:
        assert Level.WORDS_3 < Level.WORDS_5 < Level.WORDS_8 < Level.WORDS_12
        assert Level.WORDS_12 < Level.FULL_VERSE < Level.FULL_PASSAGE < Level.REFERENCE_ONLY

    def test_advance_moves_up_one(self) -> None:
        assert advance(Level.WORDS_3) == Level.WORDS_5
        assert advance(Level.WORDS_5) == Level.WORDS_8

    def test_advance_at_top_stays(self) -> None:
        assert advance(Level.REFERENCE_ONLY) == Level.REFERENCE_ONLY

    def test_drop_moves_down_one(self) -> None:
        assert drop(Level.WORDS_8) == Level.WORDS_5
        assert drop(Level.WORDS_5) == Level.WORDS_3

    def test_drop_at_bottom_stays(self) -> None:
        assert drop(Level.WORDS_3) == Level.WORDS_3

    def test_all_levels_have_target_word_entry(self) -> None:
        for level in Level:
            assert level in LEVEL_TARGET_WORDS

    def test_target_words_correct_values(self) -> None:
        assert LEVEL_TARGET_WORDS[Level.WORDS_3] == 3
        assert LEVEL_TARGET_WORDS[Level.WORDS_12] == 12
        assert LEVEL_TARGET_WORDS[Level.FULL_VERSE] == -1
        assert LEVEL_TARGET_WORDS[Level.REFERENCE_ONLY] is None


# ---------------------------------------------------------------------------
# DefaultProgressionEngine — initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_implements_abstract_interface(self) -> None:
        assert isinstance(_engine(), ProgressionEngine)

    def test_new_verse_starts_at_words_3(self) -> None:
        eng = _engine()
        assert eng.get_level(USER, VERSE) == Level.WORDS_3

    def test_separate_verses_are_independent(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        eng.record_attempt(USER, VERSE, PASS, _day(1))  # advance John 3:16
        assert eng.get_level(USER, "Psalm 23:1") == Level.WORDS_3

    def test_separate_users_are_independent(self) -> None:
        eng = _engine()
        eng.record_attempt(1, VERSE, PASS, _day(0))
        eng.record_attempt(1, VERSE, PASS, _day(1))  # advance user 1
        assert eng.get_level(2, VERSE) == Level.WORDS_3

    def test_pass_threshold_configurable(self) -> None:
        eng = _engine(pass_threshold=0.5)
        outcome = eng.record_attempt(USER, VERSE, 0.6, _day(0))
        assert outcome.passed is True

    def test_drop_days_configurable(self) -> None:
        eng = _engine(drop_days=2)
        eng.record_attempt(USER, VERSE, FAIL, _day(0))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(1))
        assert outcome.level_after == Level.WORDS_3  # already at floor, no drop
        # bump to WORDS_5 first, then drop with drop_days=2
        eng2 = _engine(drop_days=2)
        eng2.record_attempt(USER, VERSE, PASS, _day(0))
        eng2.record_attempt(USER, VERSE, PASS, _day(1))  # now at WORDS_5
        eng2.record_attempt(USER, VERSE, FAIL, _day(2))
        outcome2 = eng2.record_attempt(USER, VERSE, FAIL, _day(3))
        assert outcome2.level_after == Level.WORDS_3


# ---------------------------------------------------------------------------
# Advancing
# ---------------------------------------------------------------------------

class TestAdvancing:
    def test_two_consecutive_passes_advance(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(1))
        assert outcome.level_after == Level.WORDS_5

    def test_single_pass_does_not_advance(self) -> None:
        eng = _engine()
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(0))
        assert outcome.level_after == Level.WORDS_3

    def test_non_consecutive_passes_do_not_advance(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(2))  # skipped day 1
        assert outcome.level_after == Level.WORDS_3

    def test_pass_then_fail_then_two_passes_advance(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, FAIL, _day(0))
        eng.record_attempt(USER, VERSE, PASS, _day(1))
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(2))
        assert outcome.level_after == Level.WORDS_5

    def test_advance_through_all_levels(self) -> None:
        eng = _engine()
        day = 0
        for expected_next in [
            Level.WORDS_5, Level.WORDS_8, Level.WORDS_12,
            Level.FULL_VERSE, Level.FULL_PASSAGE, Level.REFERENCE_ONLY,
        ]:
            eng.record_attempt(USER, VERSE, PASS, _day(day))
            outcome = eng.record_attempt(USER, VERSE, PASS, _day(day + 1))
            assert outcome.level_after == expected_next
            day += 2

    def test_cannot_advance_past_reference_only(self) -> None:
        eng = _engine()
        # Fast-forward to REFERENCE_ONLY
        day = 0
        for _ in range(6):
            eng.record_attempt(USER, VERSE, PASS, _day(day))
            eng.record_attempt(USER, VERSE, PASS, _day(day + 1))
            day += 2
        assert eng.get_level(USER, VERSE) == Level.REFERENCE_ONLY
        eng.record_attempt(USER, VERSE, PASS, _day(day))
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(day + 1))
        assert outcome.level_after == Level.REFERENCE_ONLY

    def test_pass_streak_resets_after_advance(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(1))
        assert outcome.pass_streak == 0  # reset after advancing


# ---------------------------------------------------------------------------
# Holding
# ---------------------------------------------------------------------------

class TestHolding:
    def test_single_fail_holds_level(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))  # get some history
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(1))
        assert outcome.level_after == Level.WORDS_3

    def test_fail_after_one_pass_resets_pass_streak(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(1))
        assert outcome.pass_streak == 0

    def test_two_non_consecutive_fails_hold(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, FAIL, _day(0))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(2))  # skipped a day
        assert outcome.level_after == Level.WORDS_3
        assert outcome.fail_streak == 1  # streak reset due to gap


# ---------------------------------------------------------------------------
# Dropping
# ---------------------------------------------------------------------------

class TestDropping:
    def _at_words_5(self) -> DefaultProgressionEngine:
        """Return an engine with USER/VERSE already at WORDS_5."""
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        eng.record_attempt(USER, VERSE, PASS, _day(1))
        assert eng.get_level(USER, VERSE) == Level.WORDS_5
        return eng

    def test_three_consecutive_fails_drop(self) -> None:
        eng = self._at_words_5()
        eng.record_attempt(USER, VERSE, FAIL, _day(2))
        eng.record_attempt(USER, VERSE, FAIL, _day(3))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(4))
        assert outcome.level_after == Level.WORDS_3

    def test_two_fails_do_not_drop(self) -> None:
        eng = self._at_words_5()
        eng.record_attempt(USER, VERSE, FAIL, _day(2))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(3))
        assert outcome.level_after == Level.WORDS_5

    def test_non_consecutive_fails_reset_streak(self) -> None:
        eng = self._at_words_5()
        eng.record_attempt(USER, VERSE, FAIL, _day(2))
        eng.record_attempt(USER, VERSE, FAIL, _day(3))
        # Skip a day — streak resets
        eng.record_attempt(USER, VERSE, FAIL, _day(5))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(6))
        assert outcome.level_after == Level.WORDS_5  # only 2 consecutive, not 3

    def test_cannot_drop_below_words_3(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, FAIL, _day(0))
        eng.record_attempt(USER, VERSE, FAIL, _day(1))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(2))
        assert outcome.level_after == Level.WORDS_3

    def test_fail_streak_resets_after_drop(self) -> None:
        eng = self._at_words_5()
        eng.record_attempt(USER, VERSE, FAIL, _day(2))
        eng.record_attempt(USER, VERSE, FAIL, _day(3))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(4))
        assert outcome.fail_streak == 0  # reset after dropping


# ---------------------------------------------------------------------------
# Pass threshold
# ---------------------------------------------------------------------------

class TestPassThreshold:
    def test_score_at_threshold_is_pass(self) -> None:
        eng = _engine(pass_threshold=0.9)
        outcome = eng.record_attempt(USER, VERSE, 0.9, _day(0))
        assert outcome.passed is True

    def test_score_below_threshold_is_fail(self) -> None:
        eng = _engine(pass_threshold=0.9)
        outcome = eng.record_attempt(USER, VERSE, 0.89, _day(0))
        assert outcome.passed is False

    def test_perfect_score_is_always_pass(self) -> None:
        eng = _engine()
        outcome = eng.record_attempt(USER, VERSE, 1.0, _day(0))
        assert outcome.passed is True

    def test_zero_score_is_always_fail(self) -> None:
        eng = _engine()
        outcome = eng.record_attempt(USER, VERSE, 0.0, _day(0))
        assert outcome.passed is False


# ---------------------------------------------------------------------------
# Same-day deduplication
# ---------------------------------------------------------------------------

class TestSameDayDeduplication:
    def test_second_pass_same_day_is_ignored(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(0))
        assert outcome.pass_streak == 1  # not incremented again

    def test_fail_does_not_override_same_day_pass(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        outcome = eng.record_attempt(USER, VERSE, FAIL, _day(0))
        assert outcome.passed is False        # this attempt is a fail
        assert outcome.pass_streak == 1       # but day-outcome stays as pass

    def test_pass_upgrades_same_day_fail(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, FAIL, _day(0))  # fail in the morning
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(0))  # pass in the evening
        assert outcome.passed is True
        assert outcome.fail_streak == 0

    def test_pass_after_same_day_fail_counts_toward_advance(self) -> None:
        eng = _engine()
        # Day 0: fail then pass (day 0 counts as pass)
        eng.record_attempt(USER, VERSE, FAIL, _day(0))
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        # Day 1: pass → two consecutive pass days → advance
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(1))
        assert outcome.level_after == Level.WORDS_5


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_to_words_3(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        eng.record_attempt(USER, VERSE, PASS, _day(1))
        assert eng.get_level(USER, VERSE) == Level.WORDS_5
        eng.reset(USER, VERSE)
        assert eng.get_level(USER, VERSE) == Level.WORDS_3

    def test_reset_clears_streaks(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        eng.reset(USER, VERSE)
        # After reset, a single pass on consecutive days should NOT advance
        eng.record_attempt(USER, VERSE, PASS, _day(1))
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(2))
        assert outcome.level_after == Level.WORDS_5  # two fresh consecutive passes

    def test_reset_does_not_affect_other_verses(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, "Psalm 23:1", PASS, _day(0))
        eng.record_attempt(USER, "Psalm 23:1", PASS, _day(1))
        eng.reset(USER, VERSE)
        assert eng.get_level(USER, "Psalm 23:1") == Level.WORDS_5


# ---------------------------------------------------------------------------
# AttemptOutcome structure
# ---------------------------------------------------------------------------

class TestAttemptOutcome:
    def test_level_before_and_after_populated(self) -> None:
        eng = _engine()
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(0))
        assert outcome.level_before == Level.WORDS_3
        assert outcome.level_after == Level.WORDS_3

    def test_level_changes_reflected_in_outcome(self) -> None:
        eng = _engine()
        eng.record_attempt(USER, VERSE, PASS, _day(0))
        outcome = eng.record_attempt(USER, VERSE, PASS, _day(1))
        assert outcome.level_before == Level.WORDS_3
        assert outcome.level_after == Level.WORDS_5
