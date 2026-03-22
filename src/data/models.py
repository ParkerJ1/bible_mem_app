from datetime import UTC, date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class CachedVerse(Base):
    """A single cached Bible verse."""

    __tablename__ = "cached_verses"
    __table_args__ = (
        UniqueConstraint("book", "chapter", "verse", "version", name="uq_verse_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book: Mapped[str] = mapped_column(String, nullable=False)
    chapter: Mapped[int] = mapped_column(Integer, nullable=False)
    verse: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)


class User(Base):
    """An app user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    verse_lists: Mapped[list["VerseList"]] = relationship("VerseList", back_populates="user", cascade="all, delete-orphan")
    proficiency_records: Mapped[list["ProficiencyRecord"]] = relationship("ProficiencyRecord", back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["SessionRecord"]] = relationship("SessionRecord", back_populates="user", cascade="all, delete-orphan")


class VerseList(Base):
    """A named collection of verse references belonging to a user."""

    __tablename__ = "verse_lists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    user: Mapped["User"] = relationship("User", back_populates="verse_lists")
    verses: Mapped[list["Verse"]] = relationship("Verse", back_populates="verse_list", cascade="all, delete-orphan")


class Verse(Base):
    """A verse reference entry within a VerseList."""

    __tablename__ = "verses"
    __table_args__ = (
        UniqueConstraint("verse_list_id", "verse_ref", name="uq_list_verse"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    verse_list_id: Mapped[int] = mapped_column(Integer, ForeignKey("verse_lists.id"), nullable=False)
    verse_ref: Mapped[str] = mapped_column(String, nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    verse_list: Mapped["VerseList"] = relationship("VerseList", back_populates="verses")


class ProficiencyRecord(Base):
    """Per-user, per-verse proficiency state.

    Mirrors the fields of _DayState from the progression engine so a
    DB-backed engine subclass can load and save state via this table.
    """

    __tablename__ = "proficiency_records"
    __table_args__ = (
        UniqueConstraint("user_id", "verse_ref", name="uq_user_verse"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    verse_ref: Mapped[str] = mapped_column(String, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pass_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fail_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_attempt_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_day_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    user: Mapped["User"] = relationship("User", back_populates="proficiency_records")


class SessionRecord(Base):
    """A record of a single practice session attempt."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    verse_ref: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    level_before: Mapped[int] = mapped_column(Integer, nullable=False)
    level_after: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    user: Mapped["User"] = relationship("User", back_populates="sessions")
