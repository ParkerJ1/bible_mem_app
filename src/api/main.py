"""FastAPI application entry point.

Start the development server with:
    uv run uvicorn src.api.main:app --reload

On startup the app:
1. Creates all database tables (idempotent).
2. Ensures the default user (id=1) and their verse list exist.
3. Builds the SessionManager singleton and stores it on app.state so it
   can be injected into route handlers via ``get_session_manager``.

All heavy model loading (Whisper, WhisperX) is deferred to
``_build_session_manager()``, which is called inside the lifespan so that
importing this module does not trigger model loading.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import DEFAULT_LIST_NAME, USER_ID
from src.api.routers import progress, sessions, verses
from src.data.database import get_session, init_db
from src.data.models import User, VerseList


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise DB and build the SessionManager."""
    init_db()
    _ensure_default_user()
    app.state.session_manager = _build_session_manager()
    yield


app = FastAPI(
    title="Bible Memorisation App",
    description="Practice Bible verse memorisation with speech recognition and spaced repetition.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(verses.router)
app.include_router(sessions.router)
app.include_router(progress.router)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse("frontend/index.html")


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def _ensure_default_user() -> None:
    """Create user id=1 and their default verse list if they do not exist."""
    with get_session() as session:
        user = session.query(User).filter_by(id=USER_ID).first()
        if user is None:
            user = User(username="default")
            session.add(user)
            session.flush()  # assign the id before using it

        vl = session.query(VerseList).filter_by(user_id=user.id).first()
        if vl is None:
            session.add(VerseList(user_id=user.id, name=DEFAULT_LIST_NAME))


def _build_session_manager():
    """Instantiate and return the production SessionManager.

    All imports are local so that importing ``main`` (e.g. in tests) does not
    trigger ML model loading.
    """
    from src.aligner.whisperx_aligner import WhisperXAligner
    from src.progression.engine import DBProgressionEngine
    from src.scorer.sequence_aligner import SequenceAligner
    from src.session.session_manager import SessionManager
    from src.sr.whisper_sr import WhisperSpeechRecogniser
    from src.text.web_provider import WEBProvider
    from src.tts.google_tts_provider import GoogleTTSProvider

    return SessionManager(
        text_provider=WEBProvider(),
        tts_provider=GoogleTTSProvider(),
        aligner=WhisperXAligner(),
        recogniser=WhisperSpeechRecogniser(),
        scorer=SequenceAligner(),
        progression=DBProgressionEngine(db_factory=get_session),
        db_factory=get_session,
    )
