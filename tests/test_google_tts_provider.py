"""Tests for GoogleTTSProvider.

All tests are fully mocked — no real API calls or filesystem side-effects
outside of a temporary directory provided by pytest's tmp_path fixture.
"""

import base64
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import httpx
import pytest

from src.tts.google_tts_provider import GoogleTTSProvider, _hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_API_KEY = "test-api-key-123"
SAMPLE_TEXT = "For God so loved the world."
FAKE_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 64  # minimal fake MP3 bytes


def _b64(data: bytes) -> str:
    """Return the base64 string Google TTS would put in audioContent."""
    return base64.b64encode(data).decode()


def _mock_response(audio_bytes: bytes, status: int = 200) -> MagicMock:
    """Build a mock httpx.Response returning a TTS-style JSON payload."""
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"audioContent": _b64(audio_bytes)}
    return mock


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GoogleTTSProvider:
    """Return a GoogleTTSProvider whose audio_cache is redirected to tmp_path."""
    monkeypatch.setenv("GOOGLE_TTS_KEY", FAKE_API_KEY)
    monkeypatch.setattr("src.tts.google_tts_provider._AUDIO_CACHE", tmp_path)
    return GoogleTTSProvider()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_raises_if_key_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing GOOGLE_TTS_KEY must raise EnvironmentError with a clear message."""
        monkeypatch.delenv("GOOGLE_TTS_KEY", raising=False)
        monkeypatch.setattr("src.tts.google_tts_provider._AUDIO_CACHE", tmp_path)
        with pytest.raises(EnvironmentError, match="GOOGLE_TTS_KEY"):
            GoogleTTSProvider()

    def test_creates_cache_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """__init__ must create the audio_cache directory if it does not exist."""
        cache = tmp_path / "audio_cache"
        monkeypatch.setenv("GOOGLE_TTS_KEY", FAKE_API_KEY)
        monkeypatch.setattr("src.tts.google_tts_provider._AUDIO_CACHE", cache)
        assert not cache.exists()
        GoogleTTSProvider()
        assert cache.is_dir()


# ---------------------------------------------------------------------------
# API request payload
# ---------------------------------------------------------------------------

class TestAPIPayload:
    def test_correct_url_and_api_key(self, provider: GoogleTTSProvider) -> None:
        """The request must go to the Google TTS endpoint with the API key as a query param."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            provider.synthesise(SAMPLE_TEXT)

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["params"] == {"key": FAKE_API_KEY}
        assert "texttospeech.googleapis.com" in mock_post.call_args[0][0]

    def test_payload_contains_text(self, provider: GoogleTTSProvider) -> None:
        """The request body must include the exact input text."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            provider.synthesise(SAMPLE_TEXT)

        payload = mock_post.call_args[1]["json"]
        assert payload["input"]["text"] == SAMPLE_TEXT

    def test_payload_audio_encoding_is_mp3(self, provider: GoogleTTSProvider) -> None:
        """audioEncoding must be MP3."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            provider.synthesise(SAMPLE_TEXT)

        payload = mock_post.call_args[1]["json"]
        assert payload["audioConfig"]["audioEncoding"] == "MP3"

    def test_payload_voice_language(self, provider: GoogleTTSProvider) -> None:
        """Voice must be an English voice."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            provider.synthesise(SAMPLE_TEXT)

        payload = mock_post.call_args[1]["json"]
        assert payload["voice"]["languageCode"].startswith("en")


# ---------------------------------------------------------------------------
# Response decoding
# ---------------------------------------------------------------------------

class TestResponseDecoding:
    def test_returned_path_contains_correct_bytes(self, provider: GoogleTTSProvider) -> None:
        """The saved file must contain exactly the decoded audioContent bytes."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            path = provider.synthesise(SAMPLE_TEXT)

        assert path.read_bytes() == FAKE_MP3

    def test_returned_path_is_mp3(self, provider: GoogleTTSProvider) -> None:
        """The saved file must have a .mp3 extension."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            path = provider.synthesise(SAMPLE_TEXT)

        assert path.suffix == ".mp3"

    def test_filename_is_hash_of_text(self, provider: GoogleTTSProvider) -> None:
        """The filename (without extension) must be the hash of the input text."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            path = provider.synthesise(SAMPLE_TEXT)

        assert path.stem == _hash(SAMPLE_TEXT)

    def test_different_texts_produce_different_files(self, provider: GoogleTTSProvider) -> None:
        """Two distinct input strings must produce two distinct output files."""
        text_a = "In the beginning God created the heavens."
        text_b = "The Lord is my shepherd."
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(b"audio_a")
            path_a = provider.synthesise(text_a)
            mock_post.return_value = _mock_response(b"audio_b")
            path_b = provider.synthesise(text_b)

        assert path_a != path_b
        assert path_a.read_bytes() == b"audio_a"
        assert path_b.read_bytes() == b"audio_b"


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

class TestCaching:
    def test_second_call_does_not_hit_api(self, provider: GoogleTTSProvider) -> None:
        """A second synthesise call for the same text must not make an API request."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            provider.synthesise(SAMPLE_TEXT)
            provider.synthesise(SAMPLE_TEXT)

        assert mock_post.call_count == 1

    def test_cached_path_has_same_bytes(self, provider: GoogleTTSProvider) -> None:
        """The cached file must still contain the original bytes on the second call."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock_post.return_value = _mock_response(FAKE_MP3)
            path1 = provider.synthesise(SAMPLE_TEXT)
            path2 = provider.synthesise(SAMPLE_TEXT)

        assert path1 == path2
        assert path2.read_bytes() == FAKE_MP3

    def test_pre_existing_file_skips_api(self, provider: GoogleTTSProvider, tmp_path: Path) -> None:
        """If the audio file already exists on disk, the API must not be called at all."""
        dest = tmp_path / f"{_hash(SAMPLE_TEXT)}.mp3"
        dest.write_bytes(FAKE_MP3)

        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            path = provider.synthesise(SAMPLE_TEXT)

        mock_post.assert_not_called()
        assert path == dest


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_empty_text_raises_value_error(self, provider: GoogleTTSProvider) -> None:
        """Empty string input must raise ValueError before hitting the API."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            with pytest.raises(ValueError, match="empty"):
                provider.synthesise("")
        mock_post.assert_not_called()

    def test_http_error_propagates(self, provider: GoogleTTSProvider) -> None:
        """An HTTP error from the API must propagate as httpx.HTTPStatusError."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock = MagicMock(spec=httpx.Response)
            mock.raise_for_status.side_effect = httpx.HTTPStatusError(
                "403", request=MagicMock(), response=MagicMock()
            )
            mock_post.return_value = mock
            with pytest.raises(httpx.HTTPStatusError):
                provider.synthesise(SAMPLE_TEXT)

    def test_failed_request_does_not_create_file(self, provider: GoogleTTSProvider, tmp_path: Path) -> None:
        """If the API call fails, no partial file must be left in the cache."""
        with patch("src.tts.google_tts_provider.httpx.post") as mock_post:
            mock = MagicMock(spec=httpx.Response)
            mock.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
            mock_post.return_value = mock
            with pytest.raises(httpx.HTTPStatusError):
                provider.synthesise(SAMPLE_TEXT)

        dest = tmp_path / f"{_hash(SAMPLE_TEXT)}.mp3"
        assert not dest.exists()


# ---------------------------------------------------------------------------
# _hash helper
# ---------------------------------------------------------------------------

class TestHash:
    def test_same_input_same_output(self) -> None:
        assert _hash("hello") == _hash("hello")

    def test_different_inputs_different_output(self) -> None:
        assert _hash("hello") != _hash("world")

    def test_length_is_16_chars(self) -> None:
        assert len(_hash("any text")) == 16
