"""Progression engine — tracks per-user, per-verse proficiency levels.

Rules
-----
Advancing   : pass the current level on two *consecutive calendar days*
              → move up one level, reset both streaks.
Holding     : a single failure (or a non-consecutive pass) does not change
              the level.
Dropping    : fail the same level on DROP_DAYS consecutive calendar days
              (configurable, default 3) → move down one level, reset streaks.

"Consecutive" means the calendar date of the last attempt is exactly one day
before today.  If the user skips a day, any in-progress streak is reset.

Same-day handling: if record_attempt() is called more than once on the same
calendar day, a pass can *upgrade* a fail recorded earlier that day, but a
fail cannot downgrade a pass.  This ensures the best outcome of the day
counts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta

import src.config as config
from src.progression.levels import Level, advance, drop


@dataclass
class _DayState:
    """Internal per-(user, verse) state used by DefaultProgressionEngine."""

    level: Level = Level.WORDS_3
    pass_streak: int = 0          # consecutive calendar days with a pass
    fail_streak: int = 0          # consecutive calendar days with a fail
    last_attempt_date: date | None = None
    last_day_passed: bool = False  # outcome of the most recent calendar day


@dataclass
class AttemptOutcome:
    """The result of recording a single practice attempt."""

    level_before: Level
    level_after: Level
    passed: bool
    pass_streak: int
    fail_streak: int


class ProgressionEngine(ABC):
    """Abstract base class for the progression engine."""

    @abstractmethod
    def record_attempt(
        self,
        user_id: int,
        verse_ref: str,
        score: float,
        attempt_date: date | None = None,
    ) -> AttemptOutcome:
        """Record a practice attempt and update the proficiency level.

        Args:
            user_id:      Identifier for the user.
            verse_ref:    Verse reference string (e.g. "John 3:16").
            score:        Score from the Scorer, in the range [0.0, 1.0].
            attempt_date: Calendar date of the attempt. Defaults to today.

        Returns:
            AttemptOutcome describing the level before/after and streak state.
        """

    @abstractmethod
    def get_level(self, user_id: int, verse_ref: str) -> Level:
        """Return the current proficiency level for a user-verse pair.

        New pairings start at Level.WORDS_3.
        """

    @abstractmethod
    def reset(self, user_id: int, verse_ref: str) -> None:
        """Reset a user-verse pair back to WORDS_3 with all streaks cleared."""


class DefaultProgressionEngine(ProgressionEngine):
    """In-memory progression engine.

    State is held in a plain dict keyed by (user_id, verse_ref).  This is
    intentionally storage-agnostic; a DB-backed subclass can override the
    _load / _save hooks without changing the progression logic.

    Configuration is read from src.config but can be overridden per-instance
    via the constructor, which is useful for tests.

    Args:
        pass_threshold: Minimum score to count as a pass (default: config.PASS_THRESHOLD).
        drop_days:      Consecutive fail days before dropping a level (default: config.DROP_DAYS).
    """

    def __init__(
        self,
        pass_threshold: float | None = None,
        drop_days: int | None = None,
    ) -> None:
        self._pass_threshold = pass_threshold if pass_threshold is not None else config.PASS_THRESHOLD
        self._drop_days = drop_days if drop_days is not None else config.DROP_DAYS
        self._states: dict[tuple[int, str], _DayState] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_attempt(
        self,
        user_id: int,
        verse_ref: str,
        score: float,
        attempt_date: date | None = None,
    ) -> AttemptOutcome:
        """Record an attempt and return the resulting AttemptOutcome."""
        today = attempt_date or date.today()
        passed = score >= self._pass_threshold
        state = self._load(user_id, verse_ref)
        level_before = state.level

        state = _apply_attempt(state, passed, today, self._drop_days)

        self._save(user_id, verse_ref, state)

        return AttemptOutcome(
            level_before=level_before,
            level_after=state.level,
            passed=passed,
            pass_streak=state.pass_streak,
            fail_streak=state.fail_streak,
        )

    def get_level(self, user_id: int, verse_ref: str) -> Level:
        """Return current level; new pairings start at WORDS_3."""
        return self._load(user_id, verse_ref).level

    def reset(self, user_id: int, verse_ref: str) -> None:
        """Reset state to initial WORDS_3 for this user-verse pair."""
        self._save(user_id, verse_ref, _DayState())

    # ------------------------------------------------------------------
    # Storage hooks (override in DB-backed subclass)
    # ------------------------------------------------------------------

    def _load(self, user_id: int, verse_ref: str) -> _DayState:
        """Return the current state, creating a default if none exists."""
        return self._states.get((user_id, verse_ref), _DayState())

    def _save(self, user_id: int, verse_ref: str, state: _DayState) -> None:
        """Persist the updated state."""
        self._states[(user_id, verse_ref)] = state


# ------------------------------------------------------------------
# Pure progression logic (independently testable)
# ------------------------------------------------------------------

def _apply_attempt(
    state: _DayState,
    passed: bool,
    today: date,
    drop_days: int,
) -> _DayState:
    """Return a new _DayState after applying the attempt.

    This function is pure — it does not mutate the input state.
    """
    from dataclasses import replace

    yesterday = today - timedelta(days=1)
    same_day = state.last_attempt_date == today
    consecutive = state.last_attempt_date == yesterday

    # ── Same-day deduplication ──────────────────────────────────────────
    # A pass can upgrade a same-day fail; a fail cannot downgrade a pass.
    if same_day:
        if state.last_day_passed:
            # Already passed today — ignore any further attempt (pass or fail).
            return state
        if not passed:
            # Still failing today — no change to streaks or level.
            return state
        # passed=True and today was previously a fail → upgrade to pass below.
        # Treat as if processing a fresh pass on a new day but keep streaks
        # consistent: un-count today's earlier fail from the fail streak.
        state = replace(
            state,
            fail_streak=max(0, state.fail_streak - 1),
        )
        # Fall through to normal pass handling.

    # ── Pass ────────────────────────────────────────────────────────────
    if passed:
        new_pass_streak = (state.pass_streak + 1) if consecutive else 1
        new_fail_streak = 0
        new_level = state.level

        if new_pass_streak >= 2:
            new_level = advance(state.level)
            new_pass_streak = 0  # reset after advancing

        return _DayState(
            level=new_level,
            pass_streak=new_pass_streak,
            fail_streak=new_fail_streak,
            last_attempt_date=today,
            last_day_passed=True,
        )

    # ── Fail ────────────────────────────────────────────────────────────
    new_fail_streak = (state.fail_streak + 1) if consecutive else 1
    new_pass_streak = 0
    new_level = state.level

    if new_fail_streak >= drop_days:
        new_level = drop(state.level)
        new_fail_streak = 0  # reset after dropping

    return _DayState(
        level=new_level,
        pass_streak=new_pass_streak,
        fail_streak=new_fail_streak,
        last_attempt_date=today,
        last_day_passed=False,
    )
