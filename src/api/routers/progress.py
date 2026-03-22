"""Progress router — query proficiency level and session history.

GET /progress/{passage_ref}    Current level, streaks, and full session history.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies import USER_ID, get_db
from src.data.models import ProficiencyRecord, SessionRecord
from src.progression.levels import Level

router = APIRouter(prefix="/progress", tags=["progress"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SessionHistoryItem(BaseModel):
    attempt_date: date
    score: float
    level_before: int
    level_after: int
    level_before_name: str
    level_after_name: str


class ProgressResponse(BaseModel):
    passage_ref: str
    level: int
    level_name: str
    pass_streak: int
    fail_streak: int
    last_attempt_date: Optional[date]
    history: list[SessionHistoryItem]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/{passage_ref}", response_model=ProgressResponse)
def get_progress(
    passage_ref: str,
    db: Session = Depends(get_db),
) -> ProgressResponse:
    """Return the proficiency level and full session history for a passage.

    If the user has never attempted this passage, default values are returned
    (level 0 / WORDS_3, empty history).  A 404 is only raised when the
    passage_ref contains characters that make it unroutable — ordinary
    unfamiliar references return defaults.
    """
    pr = (
        db.query(ProficiencyRecord)
        .filter_by(user_id=USER_ID, verse_ref=passage_ref)
        .first()
    )

    if pr is not None:
        level = pr.level
        pass_streak = pr.pass_streak
        fail_streak = pr.fail_streak
        last_attempt_date = pr.last_attempt_date
    else:
        level = 0
        pass_streak = 0
        fail_streak = 0
        last_attempt_date = None

    records = (
        db.query(SessionRecord)
        .filter_by(user_id=USER_ID, verse_ref=passage_ref)
        .order_by(SessionRecord.attempt_date.desc())
        .all()
    )

    history = [
        SessionHistoryItem(
            attempt_date=r.attempt_date,
            score=r.score,
            level_before=r.level_before,
            level_after=r.level_after,
            level_before_name=Level(r.level_before).name,
            level_after_name=Level(r.level_after).name,
        )
        for r in records
    ]

    return ProgressResponse(
        passage_ref=passage_ref,
        level=level,
        level_name=Level(level).name,
        pass_streak=pass_streak,
        fail_streak=fail_streak,
        last_attempt_date=last_attempt_date,
        history=history,
    )
