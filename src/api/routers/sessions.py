"""Sessions router — drive a practice session.

POST   /sessions/prepare                    Pre-compute audio and alignment for a passage.
GET    /sessions/{passage_ref}/segments     Get text segments for the user's current level.
POST   /sessions/score                      Submit user audio for one segment; receive score.
POST   /sessions/finish                     Submit all segment scores; update progression.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.dependencies import USER_ID, get_session_manager
from src.progression.levels import Level
from src.scorer.base import DiffToken
from src.session.session_manager import SegmentResult, SessionManager

router = APIRouter(prefix="/sessions", tags=["sessions"])


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


class SegmentResultResponse(BaseModel):
    segment_idx: int
    expected: str
    score: float
    diff: list[DiffTokenResponse]


class SegmentScoreSubmission(BaseModel):
    """A single segment result sent by the client in the finish request.

    The client accumulates these from successive calls to POST /sessions/score.
    """

    segment_idx: int
    expected: str
    score: float


class FinishRequest(BaseModel):
    passage_ref: str
    segment_results: list[SegmentScoreSubmission]


class SessionResultResponse(BaseModel):
    passage_ref: str
    overall_score: float
    segment_results: list[SegmentResultResponse]
    level_before: int
    level_after: int
    level_before_name: str
    level_after_name: str
    passed: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diff_response(diff: list[DiffToken]) -> list[DiffTokenResponse]:
    return [DiffTokenResponse(word=t.word, status=t.status) for t in diff]


def _segment_result_response(result: SegmentResult) -> SegmentResultResponse:
    return SegmentResultResponse(
        segment_idx=result.segment_idx,
        expected=result.expected,
        score=result.score,
        diff=_diff_response(result.diff),
    )


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


@router.post("/score", response_model=SegmentResultResponse)
async def score_segment(
    passage_ref: str = Form(...),
    segment_idx: int = Form(...),
    audio: UploadFile = File(...),
    sm: SessionManager = Depends(get_session_manager),
) -> SegmentResultResponse:
    """Transcribe the user's recorded audio and score it against the expected segment.

    Accepts multipart/form-data with:
    - ``passage_ref``: the passage reference string
    - ``segment_idx``: zero-based index of the segment being scored
    - ``audio``: the user's recorded audio file (WAV, MP3, M4A, etc.)
    """
    content = await audio.read()
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = sm.score_attempt(USER_ID, passage_ref, segment_idx, tmp_path)
    except IndexError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return _segment_result_response(result)


@router.post("/finish", response_model=SessionResultResponse)
def finish_session(
    body: FinishRequest,
    sm: SessionManager = Depends(get_session_manager),
) -> SessionResultResponse:
    """Record the completed session and update the user's proficiency level.

    The client passes back the segment results it received from successive
    POST /sessions/score calls.  The overall score is computed as the mean
    of all segment scores.
    """
    if not body.segment_results:
        raise HTTPException(status_code=422, detail="segment_results must not be empty")

    sm_results = [
        SegmentResult(segment_idx=r.segment_idx, expected=r.expected, score=r.score)
        for r in body.segment_results
    ]

    try:
        result = sm.finish_session(USER_ID, body.passage_ref, sm_results)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SessionResultResponse(
        passage_ref=result.passage_ref,
        overall_score=result.overall_score,
        segment_results=[_segment_result_response(r) for r in result.segment_results],
        level_before=result.level_before.value,
        level_after=result.level_after.value,
        level_before_name=result.level_before.name,
        level_after_name=result.level_after.name,
        passed=result.passed,
    )
