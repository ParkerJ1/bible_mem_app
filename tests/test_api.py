"""API layer tests.

All heavy dependencies (TTS, SR, Aligner, TextProvider) are replaced by a
MagicMock'd SessionManager.  The database uses a fresh in-memory SQLite
instance for every test so there is no cross-test state leakage.

The FastAPI TestClient is used without its context manager so the application
lifespan (which loads ML models) does not run.  Database initialisation and
the default user / verse-list seed are performed directly in fixtures.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dependencies import USER_ID, get_db, get_session_manager
from src.api.main import app
from src.data.models import (
    Base,
    ProficiencyRecord,
    SessionRecord,
    User,
    Verse,
    VerseList,
)
from src.progression.levels import Level
from src.session.session_manager import SegmentResult, SessionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """Fresh in-memory SQLite engine per test.

    StaticPool is required so that all sessions share the same underlying
    DBAPI connection.  Without it, each new connection gets an empty
    in-memory database, making seed data written by one session invisible
    to others.
    """
    e = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(e)
    yield e
    e.dispose()


@pytest.fixture()
def TestSession(engine):
    """sessionmaker bound to the test engine."""
    return sessionmaker(bind=engine)


@pytest.fixture()
def seed_db(TestSession):
    """Seed user id=1 and a default verse list."""
    s = TestSession()
    user = User(username="default")
    s.add(user)
    s.flush()
    s.add(VerseList(user_id=user.id, name="My Verses"))
    s.commit()
    s.close()


@pytest.fixture()
def mock_sm():
    """A MagicMock configured with the SessionManager spec."""
    return MagicMock(spec_set=["prepare_verse", "get_segments", "get_level",
                                "get_segment_audio", "finish_session_from_audio"])


@pytest.fixture()
def client(TestSession, seed_db, mock_sm):
    """TestClient with overridden DB and SessionManager dependencies."""

    def override_get_db():
        s = TestSession()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_session_manager] = lambda: mock_sm

    yield TestClient(app)

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_verse_to_db(TestSession, verse_ref: str) -> None:
    """Directly insert a verse into the test DB (bypasses the API)."""
    with TestSession() as s:
        vl = s.query(VerseList).filter_by(user_id=USER_ID).first()
        s.add(Verse(verse_list_id=vl.id, verse_ref=verse_ref))
        s.commit()


def _make_session_result(
    passage_ref: str = "John 3:16",
    overall_score: float = 1.0,
    level_before: Level = Level.WORDS_3,
    level_after: Level = Level.WORDS_3,
    passed: bool = True,
) -> SessionResult:
    seg = SegmentResult(
        segment_idx=0,
        expected="For God so loved the world",
        score=overall_score,
        got="for god so loved the world",
    )
    return SessionResult(
        passage_ref=passage_ref,
        overall_score=overall_score,
        segment_results=[seg],
        level_before=level_before,
        level_after=level_after,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# GET /verses
# ---------------------------------------------------------------------------


class TestListVerses:
    def test_empty_list(self, client):
        response = client.get("/verses")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_verses_in_db(self, client, TestSession):
        _add_verse_to_db(TestSession, "John 3:16")
        _add_verse_to_db(TestSession, "Romans 8:28")

        response = client.get("/verses")
        assert response.status_code == 200
        refs = [v["passage_ref"] for v in response.json()]
        assert "John 3:16" in refs
        assert "Romans 8:28" in refs

    def test_response_includes_added_at(self, client, TestSession):
        _add_verse_to_db(TestSession, "Psalm 23:1")
        items = client.get("/verses").json()
        assert "added_at" in items[0]


# ---------------------------------------------------------------------------
# POST /verses
# ---------------------------------------------------------------------------


class TestAddVerse:
    def test_returns_201(self, client, mock_sm):
        response = client.post("/verses", json={"passage_ref": "John 3:16"})
        assert response.status_code == 201

    def test_response_body(self, client, mock_sm):
        response = client.post("/verses", json={"passage_ref": "John 3:16"})
        body = response.json()
        assert body["passage_ref"] == "John 3:16"
        assert "added_at" in body

    def test_calls_prepare_verse(self, client, mock_sm):
        client.post("/verses", json={"passage_ref": "John 3:16"})
        mock_sm.prepare_verse.assert_called_once_with("John 3:16")

    def test_verse_persisted_to_db(self, client, mock_sm, TestSession):
        client.post("/verses", json={"passage_ref": "Romans 6:23"})
        with TestSession() as s:
            vl = s.query(VerseList).filter_by(user_id=USER_ID).first()
            verse = s.query(Verse).filter_by(verse_list_id=vl.id, verse_ref="Romans 6:23").first()
            assert verse is not None

    def test_duplicate_returns_409(self, client, mock_sm, TestSession):
        _add_verse_to_db(TestSession, "John 3:16")
        response = client.post("/verses", json={"passage_ref": "John 3:16"})
        assert response.status_code == 409

    def test_duplicate_does_not_call_prepare(self, client, mock_sm, TestSession):
        _add_verse_to_db(TestSession, "John 3:16")
        client.post("/verses", json={"passage_ref": "John 3:16"})
        mock_sm.prepare_verse.assert_not_called()


# ---------------------------------------------------------------------------
# DELETE /verses/{passage_ref}
# ---------------------------------------------------------------------------


class TestRemoveVerse:
    def test_returns_204(self, client, TestSession):
        _add_verse_to_db(TestSession, "John 3:16")
        response = client.delete("/verses/John 3:16")
        assert response.status_code == 204

    def test_verse_removed_from_db(self, client, TestSession):
        _add_verse_to_db(TestSession, "John 3:16")
        client.delete("/verses/John 3:16")
        with TestSession() as s:
            vl = s.query(VerseList).filter_by(user_id=USER_ID).first()
            verse = s.query(Verse).filter_by(verse_list_id=vl.id, verse_ref="John 3:16").first()
            assert verse is None

    def test_not_found_returns_404(self, client):
        response = client.delete("/verses/John 3:16")
        assert response.status_code == 404

    def test_only_removes_named_verse(self, client, TestSession):
        _add_verse_to_db(TestSession, "John 3:16")
        _add_verse_to_db(TestSession, "Romans 8:28")
        client.delete("/verses/John 3:16")
        with TestSession() as s:
            vl = s.query(VerseList).filter_by(user_id=USER_ID).first()
            remaining = s.query(Verse).filter_by(verse_list_id=vl.id).all()
            assert len(remaining) == 1
            assert remaining[0].verse_ref == "Romans 8:28"


# ---------------------------------------------------------------------------
# POST /sessions/prepare
# ---------------------------------------------------------------------------


class TestPreparePassage:
    def test_returns_200(self, client, mock_sm):
        response = client.post("/sessions/prepare", json={"passage_ref": "John 3:16"})
        assert response.status_code == 200

    def test_response_body(self, client, mock_sm):
        response = client.post("/sessions/prepare", json={"passage_ref": "John 3:16"})
        body = response.json()
        assert body["passage_ref"] == "John 3:16"
        assert body["prepared"] is True

    def test_calls_prepare_verse(self, client, mock_sm):
        client.post("/sessions/prepare", json={"passage_ref": "Romans 8:28"})
        mock_sm.prepare_verse.assert_called_once_with("Romans 8:28")


# ---------------------------------------------------------------------------
# GET /sessions/{passage_ref}/segments
# ---------------------------------------------------------------------------


class TestGetSegments:
    def test_returns_200(self, client, mock_sm):
        mock_sm.get_segments.return_value = ["For God so", "loved the world"]
        mock_sm.get_level.return_value = Level.WORDS_3

        response = client.get("/sessions/John 3:16/segments")
        assert response.status_code == 200

    def test_response_shape(self, client, mock_sm):
        mock_sm.get_segments.return_value = ["For God so", "loved the world"]
        mock_sm.get_level.return_value = Level.WORDS_3

        body = client.get("/sessions/John 3:16/segments").json()
        assert body["passage_ref"] == "John 3:16"
        assert body["level"] == Level.WORDS_3.value
        assert body["level_name"] == "WORDS_3"
        assert body["segments"] == ["For God so", "loved the world"]
        assert body["count"] == 2

    def test_passes_user_id_to_get_segments(self, client, mock_sm):
        mock_sm.get_segments.return_value = ["For God so loved the world"]
        mock_sm.get_level.return_value = Level.FULL_VERSE

        client.get("/sessions/John 3:16/segments")
        mock_sm.get_segments.assert_called_once_with(USER_ID, "John 3:16")

    def test_passage_ref_with_range(self, client, mock_sm):
        mock_sm.get_segments.return_value = ["For God so", "loved the world"]
        mock_sm.get_level.return_value = Level.WORDS_3

        response = client.get("/sessions/John 3:16-17/segments")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /sessions/{passage_ref}/segments/{segment_idx}/audio
# ---------------------------------------------------------------------------


class TestGetSegmentAudio:
    def test_returns_200_with_valid_audio(self, client, mock_sm, tmp_path):
        audio_file = tmp_path / "seg_0.mp3"
        audio_file.write_bytes(b"FAKE_MP3_DATA")
        mock_sm.get_segment_audio.return_value = audio_file

        response = client.get("/sessions/John 3:16/segments/0/audio")
        assert response.status_code == 200

    def test_response_content_type_is_audio(self, client, mock_sm, tmp_path):
        audio_file = tmp_path / "seg_0.mp3"
        audio_file.write_bytes(b"FAKE_MP3_DATA")
        mock_sm.get_segment_audio.return_value = audio_file

        response = client.get("/sessions/John 3:16/segments/0/audio")
        assert "audio" in response.headers["content-type"]

    def test_response_body_matches_file(self, client, mock_sm, tmp_path):
        audio_file = tmp_path / "seg_0.mp3"
        audio_file.write_bytes(b"FAKE_MP3_DATA")
        mock_sm.get_segment_audio.return_value = audio_file

        response = client.get("/sessions/John 3:16/segments/0/audio")
        assert response.content == b"FAKE_MP3_DATA"

    def test_calls_get_segment_audio_with_correct_args(self, client, mock_sm, tmp_path):
        audio_file = tmp_path / "seg_2.mp3"
        audio_file.write_bytes(b"DATA")
        mock_sm.get_segment_audio.return_value = audio_file

        client.get("/sessions/Romans 8:28/segments/2/audio")
        mock_sm.get_segment_audio.assert_called_once_with(USER_ID, "Romans 8:28", 2)

    def test_returns_404_when_none(self, client, mock_sm):
        mock_sm.get_segment_audio.return_value = None

        response = client.get("/sessions/John 3:16/segments/0/audio")
        assert response.status_code == 404

    def test_returns_404_when_file_missing(self, client, mock_sm, tmp_path):
        missing = tmp_path / "does_not_exist.mp3"
        mock_sm.get_segment_audio.return_value = missing

        response = client.get("/sessions/John 3:16/segments/0/audio")
        assert response.status_code == 404

    def test_returns_404_on_index_error(self, client, mock_sm):
        mock_sm.get_segment_audio.side_effect = IndexError("segment_idx 99 out of range")

        response = client.get("/sessions/John 3:16/segments/99/audio")
        assert response.status_code == 404

    def test_segment_idx_in_url_is_parsed_as_int(self, client, mock_sm, tmp_path):
        audio_file = tmp_path / "seg_3.mp3"
        audio_file.write_bytes(b"DATA")
        mock_sm.get_segment_audio.return_value = audio_file

        client.get("/sessions/John 3:16/segments/3/audio")
        call_args = mock_sm.get_segment_audio.call_args[0]
        assert call_args[2] == 3
        assert isinstance(call_args[2], int)


# ---------------------------------------------------------------------------
# POST /sessions/score
# ---------------------------------------------------------------------------


class TestScoreSegment:
    def _post_score(self, client, tmp_path, passage_ref="John 3:16", segment_idx=0,
                    audio_bytes=b"FAKE_AUDIO"):
        with patch("src.api.routers.sessions._TEMP_AUDIO_DIR", tmp_path):
            return client.post(
                "/sessions/score",
                data={"passage_ref": passage_ref, "segment_idx": str(segment_idx)},
                files={"audio": ("recording.webm", audio_bytes, "audio/webm")},
            )

    def test_returns_200(self, client, tmp_path):
        response = self._post_score(client, tmp_path)
        assert response.status_code == 200

    def test_response_shape(self, client, tmp_path):
        body = self._post_score(client, tmp_path).json()
        assert body["saved"] is True
        assert body["segment_idx"] == 0

    def test_saves_audio_file_with_correct_name(self, client, tmp_path):
        self._post_score(client, tmp_path, segment_idx=2)
        expected = tmp_path / f"{USER_ID}_john_3_16_002.webm"
        assert expected.exists()
        assert expected.read_bytes() == b"FAKE_AUDIO"

    def test_overwrite_same_segment_on_restart(self, client, tmp_path):
        """Re-recording segment 0 overwrites the previous file."""
        self._post_score(client, tmp_path, segment_idx=0, audio_bytes=b"FIRST")
        self._post_score(client, tmp_path, segment_idx=0, audio_bytes=b"SECOND")
        assert (tmp_path / f"{USER_ID}_john_3_16_000.webm").read_bytes() == b"SECOND"

    def test_does_not_use_session_manager(self, client, mock_sm, tmp_path):
        self._post_score(client, tmp_path)
        assert not mock_sm.finish_session_from_audio.called


# ---------------------------------------------------------------------------
# POST /sessions/finish
# ---------------------------------------------------------------------------


class TestFinishSession:
    def _post_finish(self, client, mock_sm, tmp_path,
                     passage_ref="John 3:16", segment_count=1,
                     audio_content=b"AUDIO"):
        slug = f"{USER_ID}_john_3_16"
        for i in range(segment_count):
            (tmp_path / f"{slug}_{i:03d}.webm").write_bytes(audio_content)
        mock_sm.finish_session_from_audio.return_value = _make_session_result(
            passage_ref=passage_ref,
        )
        with patch("src.api.routers.sessions._TEMP_AUDIO_DIR", tmp_path):
            return client.post("/sessions/finish", json={
                "passage_ref": passage_ref,
                "segment_count": segment_count,
            })

    def test_returns_200(self, client, mock_sm, tmp_path):
        response = self._post_finish(client, mock_sm, tmp_path)
        assert response.status_code == 200

    def test_response_shape(self, client, mock_sm, tmp_path):
        mock_sm.finish_session_from_audio.return_value = _make_session_result(overall_score=0.95)
        (tmp_path / f"{USER_ID}_john_3_16_000.webm").write_bytes(b"A")
        with patch("src.api.routers.sessions._TEMP_AUDIO_DIR", tmp_path):
            body = client.post("/sessions/finish", json={
                "passage_ref": "John 3:16", "segment_count": 1,
            }).json()
        assert body["passage_ref"] == "John 3:16"
        assert body["overall_score"] == pytest.approx(0.95)
        assert "transcript" in body
        assert "diff" in body
        assert isinstance(body["diff"], list)
        assert "level_before" in body
        assert "level_after" in body
        assert "level_before_name" in body
        assert "level_after_name" in body
        assert "passed" in body

    def test_calls_finish_session_from_audio_with_correct_args(self, client, mock_sm, tmp_path):
        mock_sm.finish_session_from_audio.return_value = _make_session_result()
        (tmp_path / f"{USER_ID}_john_3_16_000.webm").write_bytes(b"A")
        (tmp_path / f"{USER_ID}_john_3_16_001.webm").write_bytes(b"B")
        with patch("src.api.routers.sessions._TEMP_AUDIO_DIR", tmp_path):
            client.post("/sessions/finish", json={
                "passage_ref": "John 3:16", "segment_count": 2,
            })
        call_args = mock_sm.finish_session_from_audio.call_args[0]
        assert call_args[0] == USER_ID
        assert call_args[1] == "John 3:16"
        assert len(call_args[2]) == 2
        assert all(isinstance(p, Path) for p in call_args[2])

    def test_skipped_segments_excluded_from_audio_list(self, client, mock_sm, tmp_path):
        """Segment 1 was skipped; only segments 0 and 2 should be stitched."""
        mock_sm.finish_session_from_audio.return_value = _make_session_result()
        (tmp_path / f"{USER_ID}_john_3_16_000.webm").write_bytes(b"A")
        (tmp_path / f"{USER_ID}_john_3_16_002.webm").write_bytes(b"C")
        with patch("src.api.routers.sessions._TEMP_AUDIO_DIR", tmp_path):
            client.post("/sessions/finish", json={
                "passage_ref": "John 3:16", "segment_count": 3,
            })
        audio_files = mock_sm.finish_session_from_audio.call_args[0][2]
        assert len(audio_files) == 2

    def test_level_advance_reflected_in_response(self, client, mock_sm, tmp_path):
        mock_sm.finish_session_from_audio.return_value = _make_session_result(
            level_before=Level.WORDS_3,
            level_after=Level.WORDS_5,
            passed=True,
        )
        (tmp_path / f"{USER_ID}_john_3_16_000.webm").write_bytes(b"A")
        with patch("src.api.routers.sessions._TEMP_AUDIO_DIR", tmp_path):
            body = client.post("/sessions/finish", json={
                "passage_ref": "John 3:16", "segment_count": 1,
            }).json()
        assert body["level_before"] == Level.WORDS_3.value
        assert body["level_after"] == Level.WORDS_5.value
        assert body["passed"] is True

    def test_no_audio_files_returns_422(self, client, mock_sm, tmp_path):
        with patch("src.api.routers.sessions._TEMP_AUDIO_DIR", tmp_path):
            response = client.post("/sessions/finish", json={
                "passage_ref": "John 3:16", "segment_count": 2,
            })
        assert response.status_code == 422

    def test_cleans_up_temp_files_after_finish(self, client, mock_sm, tmp_path):
        mock_sm.finish_session_from_audio.return_value = _make_session_result()
        audio_file = tmp_path / f"{USER_ID}_john_3_16_000.webm"
        stale_file = tmp_path / f"{USER_ID}_john_3_16_001.webm"
        audio_file.write_bytes(b"A")
        stale_file.write_bytes(b"STALE")
        with patch("src.api.routers.sessions._TEMP_AUDIO_DIR", tmp_path):
            client.post("/sessions/finish", json={
                "passage_ref": "John 3:16", "segment_count": 1,
            })
        assert not audio_file.exists()
        assert not stale_file.exists()


# ---------------------------------------------------------------------------
# GET /progress/{passage_ref}
# ---------------------------------------------------------------------------


class TestGetProgress:
    def test_returns_defaults_when_no_history(self, client):
        response = client.get("/progress/John 3:16")
        assert response.status_code == 200
        body = response.json()
        assert body["passage_ref"] == "John 3:16"
        assert body["level"] == 0
        assert body["level_name"] == "WORDS_3"
        assert body["pass_streak"] == 0
        assert body["fail_streak"] == 0
        assert body["last_attempt_date"] is None
        assert body["history"] == []

    def test_returns_proficiency_record_data(self, client, TestSession):
        with TestSession() as s:
            s.add(ProficiencyRecord(
                user_id=USER_ID,
                verse_ref="John 3:16",
                level=2,
                pass_streak=1,
                fail_streak=0,
                last_attempt_date=date(2026, 3, 22),
                last_day_passed=True,
            ))
            s.commit()

        body = client.get("/progress/John 3:16").json()
        assert body["level"] == 2
        assert body["level_name"] == "WORDS_8"
        assert body["pass_streak"] == 1
        assert body["last_attempt_date"] == "2026-03-22"

    def test_returns_session_history(self, client, TestSession):
        with TestSession() as s:
            s.add(SessionRecord(
                user_id=USER_ID,
                verse_ref="John 3:16",
                score=0.85,
                level_before=0,
                level_after=0,
                attempt_date=date(2026, 3, 20),
            ))
            s.add(SessionRecord(
                user_id=USER_ID,
                verse_ref="John 3:16",
                score=1.0,
                level_before=0,
                level_after=1,
                attempt_date=date(2026, 3, 21),
            ))
            s.commit()

        body = client.get("/progress/John 3:16").json()
        assert len(body["history"]) == 2
        # Most recent first
        assert body["history"][0]["attempt_date"] == "2026-03-21"
        assert body["history"][0]["score"] == pytest.approx(1.0)
        assert body["history"][0]["level_after"] == 1
        assert body["history"][0]["level_after_name"] == "WORDS_5"

    def test_history_ordered_most_recent_first(self, client, TestSession):
        with TestSession() as s:
            for day, score in [(20, 0.5), (21, 0.7), (22, 0.9)]:
                s.add(SessionRecord(
                    user_id=USER_ID,
                    verse_ref="Psalm 23:1",
                    score=score,
                    level_before=0,
                    level_after=0,
                    attempt_date=date(2026, 3, day),
                ))
            s.commit()

        body = client.get("/progress/Psalm 23:1").json()
        scores = [h["score"] for h in body["history"]]
        assert scores == pytest.approx([0.9, 0.7, 0.5])

    def test_history_does_not_mix_verses(self, client, TestSession):
        with TestSession() as s:
            s.add(SessionRecord(
                user_id=USER_ID, verse_ref="John 3:16", score=1.0,
                level_before=0, level_after=0, attempt_date=date(2026, 3, 22),
            ))
            s.add(SessionRecord(
                user_id=USER_ID, verse_ref="Romans 8:28", score=0.5,
                level_before=0, level_after=0, attempt_date=date(2026, 3, 22),
            ))
            s.commit()

        body = client.get("/progress/John 3:16").json()
        assert len(body["history"]) == 1
        assert body["history"][0]["score"] == pytest.approx(1.0)
