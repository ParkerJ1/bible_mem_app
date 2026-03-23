"""Sessions router — drive a practice session.

POST   /sessions/prepare                               Pre-compute audio and alignment.
GET    /sessions/{passage_ref}/segments                Get text segments for current level.
GET    /sessions/{passage_ref}/segments/{idx}/audio    Stream audio for one segment.
POST   /sessions/score                                 Save uploaded segment audio to temp folder.
POST   /sessions/finish                                Stitch audio, transcribe, score, update progression.
"""

import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.dependencies import USER_ID, get_session_manager
from src.progression.levels import Level
from src.scorer.base import DiffToken
from src.session.session_manager import SessionManager

router = APIRouter(prefix="/sessions", tags=["sessions"])

# Temporary directory for mid-session audio files.
# Created by POST /sessions/score, consumed and deleted by POST /sessions/finish.
_TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "bible_mem_sessions"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PrepareRequest(BaseModel):
    """Request body for the prepare endpoint."""

    passage_ref: str


class PrepareResponse(BaseModel):
    passage_ref: str
    prepared: bool = True


class SegmentsResponse(BaseModel):
    passage_ref: str
    level: int
    level_name: str
    segments: list[str]
    count: int


class DiffTokenResponse(BaseModel):
    word: str
    status: str  # "correct" | "missing" | "inserted"


class SaveAudioResponse(BaseModel):
    """Returned by POST /sessions/score after saving a segment audio file."""

    saved: bool = True
    segment_idx: int


class FinishRequest(BaseModel):
    """Request body for the finish endpoint."""

    passage_ref: str
    segment_count: int  # total segments in the session, including any that were skipped


class SessionResultResponse(BaseModel):
    passage_ref: str
    overall_score: float
    transcript: str
    diff: list[DiffTokenResponse]
    level_before: int
    level_after: int
    level_before_name: str
    level_after_name: str
    passed: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_slug(passage_ref: str) -> str:
    """Convert a passage reference to a safe filename component, e.g. 'john_3_16'."""
    return re.sub(r"[^a-z0-9]+", "_", passage_ref.lower()).strip("_")


def _diff_response(diff: list[DiffToken]) -> list[DiffTokenResponse]:
    return [DiffTokenResponse(word=t.word, status=t.status) for t in diff]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/prepare", response_model=PrepareResponse)
def prepare_passage(
    body: PrepareRequest,
    sm: SessionManager = Depends(get_session_manager),
) -> PrepareResponse:
    """Pre-compute TTS audio and forced alignment for a passage.

    Idempotent — safe to call more than once for the same passage.
    Called automatically by POST /verses; this endpoint allows manual
    re-preparation if needed.
    """
    sm.prepare_verse(body.passage_ref)
    return PrepareResponse(passage_ref=body.passage_ref)


@router.get("/{passage_ref}/segments", response_model=SegmentsResponse)
def get_segments(
    passage_ref: str,
    sm: SessionManager = Depends(get_session_manager),
) -> SegmentsResponse:
    """Return the text segments for the user's current proficiency level.

    The passage must have been prepared first (POST /sessions/prepare or
    POST /verses).
    """
    try:
        segments = sm.get_segments(USER_ID, passage_ref)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    level: Level = sm.get_level(USER_ID, passage_ref)
    return SegmentsResponse(
        passage_ref=passage_ref,
        level=level.value,
        level_name=level.name,
        segments=segments,
        count=len(segments),
    )


@router.get("/{passage_ref}/segments/{segment_idx}/audio")
def get_segment_audio(
    passage_ref: str,
    segment_idx: int,
    sm: SessionManager = Depends(get_session_manager),
) -> FileResponse:
    """Stream the pre-computed audio clip for a single segment.

    Returns 404 if the segment index is out of range, if no audio has been
    prepared for the passage, or if the audio file is not present on disk.
    """
    try:
        path = sm.get_segment_audio(USER_ID, passage_ref, segment_idx)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Audio not available for this segment")

    return FileResponse(path, media_type="audio/mpeg")


@router.post("/score", response_model=SaveAudioResponse)
async def save_segment_audio(
    passage_ref: str = Form(...),
    segment_idx: int = Form(...),
    audio: UploadFile = File(...),
) -> SaveAudioResponse:
    """Save uploaded segment audio to a temporary folder.

    No transcription or scoring is performed here.  All processing happens
    in POST /sessions/finish once all segments have been recorded.

    Files are named ``{user_id}_{passage_slug}_{segment_idx:03d}.webm`` and
    stored in a shared temp directory.  Re-recording a segment (e.g. after a
    restart) overwrites the previous file for that index.
    """
    _TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{USER_ID}_{_ref_slug(passage_ref)}_{segment_idx:03d}.webm"
    dest = _TEMP_AUDIO_DIR / filename
    dest.write_bytes(await audio.read())
    return SaveAudioResponse(segment_idx=segment_idx)


@router.post("/finish", response_model=SessionResultResponse)
def finish_session(
    body: FinishRequest,
    sm: SessionManager = Depends(get_session_manager),
) -> SessionResultResponse:
    """Stitch saved segment audio, transcribe once, score, and update progression.

    Collects the audio files saved by POST /sessions/score for this user and
    passage (in segment order, skipping any that were not recorded), concatenates
    them with ffmpeg, transcribes the result with Whisper, scores the transcript
    against the full passage text, updates the user's proficiency level, then
    cleans up all temp files for this passage.

    Returns 422 if no audio files are found for the passage.
    """
    slug = _ref_slug(body.passage_ref)

    # Collect only segments that were actually recorded; skipped segments
    # (no file on disk) are silently excluded from the stitched audio.
    audio_files = [
        p
        for i in range(body.segment_count)
        if (p := _TEMP_AUDIO_DIR / f"{USER_ID}_{slug}_{i:03d}.webm").exists()
    ]

    if not audio_files:
        raise HTTPException(status_code=422, detail="No audio found for this session")

    try:
        result = sm.finish_session_from_audio(USER_ID, body.passage_ref, audio_files)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        # Remove all temp files for this user+passage, including any stale ones
        # left over from a restarted verse attempt.
        if _TEMP_AUDIO_DIR.exists():
            for stale in _TEMP_AUDIO_DIR.glob(f"{USER_ID}_{slug}_*.webm"):
                stale.unlink(missing_ok=True)

    full_result = result.segment_results[0]
    return SessionResultResponse(
        passage_ref=result.passage_ref,
        overall_score=result.overall_score,
        transcript=full_result.got,
        diff=_diff_response(full_result.diff),
        level_before=result.level_before.value,
        level_after=result.level_after.value,
        level_before_name=result.level_before.name,
        level_after_name=result.level_after.name,
        passed=result.passed,
    )
