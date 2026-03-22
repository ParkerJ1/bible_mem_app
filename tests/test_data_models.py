"""Tests for the data layer ORM models.

Uses an in-memory SQLite database so no files are created on disk.
"""

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.data.models import Base, ProficiencyRecord, SessionRecord, User, Verse, VerseList


@pytest.fixture()
def db_session():
    """Yield a fresh in-memory SQLite session for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------


class TestUser:
    def test_create_user(self, db_session):
        user = User(username="alice")
        db_session.add(user)
        db_session.commit()

        fetched = db_session.query(User).filter_by(username="alice").one()
        assert fetched.id is not None
        assert fetched.username == "alice"
        assert isinstance(fetched.created_at, datetime)

    def test_username_must_be_unique(self, db_session):
        db_session.add(User(username="alice"))
        db_session.commit()
        db_session.add(User(username="alice"))
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_delete_user_cascades(self, db_session):
        user = User(username="bob")
        db_session.add(user)
        db_session.commit()

        vl = VerseList(user_id=user.id, name="My List")
        db_session.add(vl)
        db_session.commit()

        verse = Verse(verse_list_id=vl.id, verse_ref="John 3:16")
        pr = ProficiencyRecord(user_id=user.id, verse_ref="John 3:16")
        sr = SessionRecord(
            user_id=user.id,
            verse_ref="John 3:16",
            score=0.9,
            level_before=0,
            level_after=0,
            attempt_date=date.today(),
        )
        db_session.add_all([verse, pr, sr])
        db_session.commit()

        db_session.delete(user)
        db_session.commit()

        assert db_session.query(VerseList).count() == 0
        assert db_session.query(Verse).count() == 0
        assert db_session.query(ProficiencyRecord).count() == 0
        assert db_session.query(SessionRecord).count() == 0


# ---------------------------------------------------------------------------
# VerseList
# ---------------------------------------------------------------------------


class TestVerseList:
    def test_create_verse_list(self, db_session):
        user = User(username="carol")
        db_session.add(user)
        db_session.commit()

        vl = VerseList(user_id=user.id, name="Romans Road")
        db_session.add(vl)
        db_session.commit()

        fetched = db_session.query(VerseList).filter_by(name="Romans Road").one()
        assert fetched.user_id == user.id
        assert isinstance(fetched.created_at, datetime)

    def test_relationship_to_user(self, db_session):
        user = User(username="dave")
        db_session.add(user)
        db_session.commit()

        vl = VerseList(user_id=user.id, name="Psalms")
        db_session.add(vl)
        db_session.commit()

        db_session.refresh(vl)
        assert vl.user.username == "dave"

    def test_user_can_have_multiple_lists(self, db_session):
        user = User(username="eve")
        db_session.add(user)
        db_session.commit()

        db_session.add_all([
            VerseList(user_id=user.id, name="List A"),
            VerseList(user_id=user.id, name="List B"),
        ])
        db_session.commit()

        db_session.refresh(user)
        assert len(user.verse_lists) == 2


# ---------------------------------------------------------------------------
# Verse
# ---------------------------------------------------------------------------


class TestVerse:
    def _make_list(self, db_session, username="frank"):
        user = User(username=username)
        db_session.add(user)
        db_session.commit()
        vl = VerseList(user_id=user.id, name="Test List")
        db_session.add(vl)
        db_session.commit()
        return vl

    def test_add_verse_to_list(self, db_session):
        vl = self._make_list(db_session)
        verse = Verse(verse_list_id=vl.id, verse_ref="John 3:16")
        db_session.add(verse)
        db_session.commit()

        fetched = db_session.query(Verse).filter_by(verse_ref="John 3:16").one()
        assert fetched.verse_list_id == vl.id
        assert isinstance(fetched.added_at, datetime)

    def test_duplicate_verse_in_same_list_rejected(self, db_session):
        vl = self._make_list(db_session)
        db_session.add(Verse(verse_list_id=vl.id, verse_ref="Ps 23:1"))
        db_session.commit()
        db_session.add(Verse(verse_list_id=vl.id, verse_ref="Ps 23:1"))
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_same_verse_in_different_lists_allowed(self, db_session):
        vl1 = self._make_list(db_session, username="grace")
        user2 = User(username="heidi")
        db_session.add(user2)
        db_session.commit()
        vl2 = VerseList(user_id=user2.id, name="List 2")
        db_session.add(vl2)
        db_session.commit()

        db_session.add_all([
            Verse(verse_list_id=vl1.id, verse_ref="Gen 1:1"),
            Verse(verse_list_id=vl2.id, verse_ref="Gen 1:1"),
        ])
        db_session.commit()

        assert db_session.query(Verse).filter_by(verse_ref="Gen 1:1").count() == 2

    def test_delete_list_cascades_to_verses(self, db_session):
        vl = self._make_list(db_session, username="ivan")
        db_session.add(Verse(verse_list_id=vl.id, verse_ref="Rev 1:1"))
        db_session.commit()

        db_session.delete(vl)
        db_session.commit()

        assert db_session.query(Verse).count() == 0


# ---------------------------------------------------------------------------
# ProficiencyRecord
# ---------------------------------------------------------------------------


class TestProficiencyRecord:
    def _make_user(self, db_session, username):
        user = User(username=username)
        db_session.add(user)
        db_session.commit()
        return user

    def test_create_proficiency_record_defaults(self, db_session):
        user = self._make_user(db_session, "judy")
        pr = ProficiencyRecord(user_id=user.id, verse_ref="John 3:16")
        db_session.add(pr)
        db_session.commit()

        fetched = db_session.query(ProficiencyRecord).filter_by(verse_ref="John 3:16").one()
        assert fetched.level == 0
        assert fetched.pass_streak == 0
        assert fetched.fail_streak == 0
        assert fetched.last_attempt_date is None
        assert fetched.last_day_passed is False

    def test_update_proficiency_record(self, db_session):
        user = self._make_user(db_session, "karl")
        pr = ProficiencyRecord(user_id=user.id, verse_ref="Rom 3:23")
        db_session.add(pr)
        db_session.commit()

        pr.level = 2
        pr.pass_streak = 1
        pr.last_attempt_date = date(2026, 3, 21)
        pr.last_day_passed = True
        db_session.commit()

        fetched = db_session.query(ProficiencyRecord).filter_by(verse_ref="Rom 3:23").one()
        assert fetched.level == 2
        assert fetched.pass_streak == 1
        assert fetched.last_attempt_date == date(2026, 3, 21)
        assert fetched.last_day_passed is True

    def test_unique_per_user_verse(self, db_session):
        user = self._make_user(db_session, "lena")
        db_session.add(ProficiencyRecord(user_id=user.id, verse_ref="Jn 1:1"))
        db_session.commit()
        db_session.add(ProficiencyRecord(user_id=user.id, verse_ref="Jn 1:1"))
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_different_users_same_verse_allowed(self, db_session):
        u1 = self._make_user(db_session, "mike")
        u2 = self._make_user(db_session, "nina")

        db_session.add_all([
            ProficiencyRecord(user_id=u1.id, verse_ref="Ps 119:11"),
            ProficiencyRecord(user_id=u2.id, verse_ref="Ps 119:11"),
        ])
        db_session.commit()

        assert db_session.query(ProficiencyRecord).filter_by(verse_ref="Ps 119:11").count() == 2


# ---------------------------------------------------------------------------
# SessionRecord
# ---------------------------------------------------------------------------


class TestSessionRecord:
    def _make_user(self, db_session, username):
        user = User(username=username)
        db_session.add(user)
        db_session.commit()
        return user

    def test_create_session_record(self, db_session):
        user = self._make_user(db_session, "omar")
        sr = SessionRecord(
            user_id=user.id,
            verse_ref="Jn 3:16",
            score=0.95,
            level_before=1,
            level_after=1,
            attempt_date=date(2026, 3, 22),
        )
        db_session.add(sr)
        db_session.commit()

        fetched = db_session.query(SessionRecord).filter_by(verse_ref="Jn 3:16").one()
        assert fetched.score == pytest.approx(0.95)
        assert fetched.level_before == 1
        assert fetched.level_after == 1
        assert fetched.attempt_date == date(2026, 3, 22)
        assert isinstance(fetched.created_at, datetime)

    def test_level_advance_recorded(self, db_session):
        user = self._make_user(db_session, "pam")
        sr = SessionRecord(
            user_id=user.id,
            verse_ref="Rom 6:23",
            score=1.0,
            level_before=2,
            level_after=3,
            attempt_date=date(2026, 3, 22),
        )
        db_session.add(sr)
        db_session.commit()

        fetched = db_session.query(SessionRecord).filter_by(verse_ref="Rom 6:23").one()
        assert fetched.level_before == 2
        assert fetched.level_after == 3

    def test_multiple_sessions_for_same_verse(self, db_session):
        user = self._make_user(db_session, "quinn")
        for i, score in enumerate([0.6, 0.8, 1.0]):
            db_session.add(SessionRecord(
                user_id=user.id,
                verse_ref="Ps 23:1",
                score=score,
                level_before=0,
                level_after=0,
                attempt_date=date(2026, 3, 20 + i),
            ))
        db_session.commit()

        records = (
            db_session.query(SessionRecord)
            .filter_by(user_id=user.id, verse_ref="Ps 23:1")
            .order_by(SessionRecord.attempt_date)
            .all()
        )
        assert len(records) == 3
        assert [r.score for r in records] == pytest.approx([0.6, 0.8, 1.0])

    def test_relationship_to_user(self, db_session):
        user = self._make_user(db_session, "rosa")
        sr = SessionRecord(
            user_id=user.id,
            verse_ref="Gen 1:1",
            score=0.75,
            level_before=0,
            level_after=0,
            attempt_date=date(2026, 3, 22),
        )
        db_session.add(sr)
        db_session.commit()

        db_session.refresh(sr)
        assert sr.user.username == "rosa"
