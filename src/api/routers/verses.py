"""Verses router — manage the user's verse list.

GET    /verses                  List all verses in the user's list.
POST   /verses                  Add a verse (triggers TTS + alignment preparation).
DELETE /verses/{passage_ref}    Remove a verse from the list.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies import DEFAULT_LIST_NAME, USER_ID, get_db, get_session_manager
from src.data.models import Verse, VerseList
from src.session.session_manager import SessionManager

router = APIRouter(prefix="/verses", tags=["verses"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AddVerseRequest(BaseModel):
    """Request body for adding a verse to the user's list."""

    passage_ref: str


class VerseResponse(BaseModel):
    """A single verse entry in the user's list."""

    passage_ref: str
    added_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_verse_list(db: Session) -> VerseList:
    """Return the user's default verse list, raising 500 if missing."""
    vl = db.query(VerseList).filter_by(user_id=USER_ID).first()
    if vl is None:
        raise HTTPException(status_code=500, detail="Default verse list not found. Is the app initialised?")
    return vl


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[VerseResponse])
def list_verses(db: Session = Depends(get_db)) -> list[VerseResponse]:
    """Return all verse references in the user's list, ordered by when they were added."""
    vl = _get_verse_list(db)
    return [
        VerseResponse(passage_ref=v.verse_ref, added_at=v.added_at)
        for v in sorted(vl.verses, key=lambda v: v.added_at)
    ]


@router.post("", status_code=201, response_model=VerseResponse)
def add_verse(
    body: AddVerseRequest,
    db: Session = Depends(get_db),
    sm: SessionManager = Depends(get_session_manager),
) -> VerseResponse:
    """Add a verse to the user's list and prepare its audio + alignment.

    Returns 409 if the verse is already in the list.
    The preparation step (TTS synthesis and forced alignment) is performed
    synchronously before the response is returned.
    """
    vl = _get_verse_list(db)

    existing = (
        db.query(Verse)
        .filter_by(verse_list_id=vl.id, verse_ref=body.passage_ref)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"{body.passage_ref!r} is already in your list",
        )

    verse = Verse(verse_list_id=vl.id, verse_ref=body.passage_ref)
    db.add(verse)
    db.commit()          # commit before preparing so the session is free for prepare_verse
    db.refresh(verse)   # pick up db-generated added_at

    sm.prepare_verse(body.passage_ref)

    return VerseResponse(passage_ref=verse.verse_ref, added_at=verse.added_at)


@router.delete("/{passage_ref}", status_code=204)
def remove_verse(
    passage_ref: str,
    db: Session = Depends(get_db),
) -> None:
    """Remove a verse from the user's list.

    Returns 404 if the verse is not in the list.
    """
    vl = _get_verse_list(db)
    verse = (
        db.query(Verse)
        .filter_by(verse_list_id=vl.id, verse_ref=passage_ref)
        .first()
    )
    if verse is None:
        raise HTTPException(
            status_code=404,
            detail=f"{passage_ref!r} is not in your list",
        )
    db.delete(verse)
