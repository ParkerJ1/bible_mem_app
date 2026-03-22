"""Shared FastAPI dependencies for the Bible memorisation API.

``get_db`` yields a SQLAlchemy session for the lifetime of a single request.
``get_session_manager`` reads the :class:`~src.session.session_manager.SessionManager`
singleton from ``app.state``, which is set during the application lifespan.

Both dependencies can be overridden in tests via ``app.dependency_overrides``.
"""

from typing import Generator

from fastapi import Request
from sqlalchemy.orm import Session

from src.data.database import SessionLocal
from src.session.session_manager import SessionManager

#: The single hard-coded user id for this personal app.
USER_ID: int = 1

#: Name of the default verse list created on startup.
DEFAULT_LIST_NAME: str = "My Verses"


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for a single request, committing on success."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_session_manager(request: Request) -> SessionManager:
    """Return the SessionManager singleton stored on app.state."""
    return request.app.state.session_manager
