"""Tests for WEBProvider.

Legitimate calls use real HTTP requests to bible-api.com.
Error cases use httpx mocking to avoid network dependency.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.shared import Verse
from src.text.web_provider import WEBProvider


@pytest.fixture
def provider() -> WEBProvider:
    """Return a WEBProvider instance."""
    return WEBProvider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(data: dict) -> MagicMock:
    """Build a mock httpx response that returns *data* as JSON."""
    mock = MagicMock(spec=httpx.Response)
    mock.json.return_value = data
    mock.raise_for_status.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Live API tests (require network)
# ---------------------------------------------------------------------------

class TestWEBProviderLive:
    """Integration tests that hit the real bible-api.com."""

    @pytest.mark.integration
    def test_single_verse_john_3_16(self, provider: WEBProvider) -> None:
        """John 3:16 should return one Verse with the correct text."""
        verses = provider.get_verses("John", 3, 16)

        assert len(verses) == 1
        v = verses[0]
        assert isinstance(v, Verse)
        assert v.book == "John"
        assert v.chapter == 3
        assert v.verse == 16
        assert "God so loved the world" in v.text

    @pytest.mark.integration
    def test_single_verse_has_no_newlines(self, provider: WEBProvider) -> None:
        """Returned verse text should contain no newline or tab characters."""
        verses = provider.get_verses("John", 3, 16)
        assert "\n" not in verses[0].text
        assert "\r" not in verses[0].text
        assert "\t" not in verses[0].text

    @pytest.mark.integration
    def test_verse_range_returns_multiple_verses(self, provider: WEBProvider) -> None:
        """Requesting a range should return one Verse per verse number."""
        verses = provider.get_verses("Psalms", 23, 1, 3)

        assert len(verses) == 3
        assert verses[0].verse == 1
        assert verses[1].verse == 2
        assert verses[2].verse == 3
        for v in verses:
            assert v.chapter == 23
            assert v.book == "Psalms"

    @pytest.mark.integration
    def test_verse_range_text_is_clean(self, provider: WEBProvider) -> None:
        """Every verse in a multi-verse fetch should be free of raw whitespace."""
        verses = provider.get_verses("Genesis", 1, 1, 3)
        for v in verses:
            assert "\n" not in v.text
            assert "  " not in v.text  # no double spaces


# ---------------------------------------------------------------------------
# Error-handling tests (mocked)
# ---------------------------------------------------------------------------

class TestWEBProviderErrors:
    """Unit tests for error handling, using mocked HTTP responses."""

    def test_not_found_error_raises_value_error(self, provider: WEBProvider) -> None:
        """An API response with 'error' key should raise ValueError."""
        with patch("src.text.web_provider.httpx.get") as mock_get:
            mock_get.return_value = _mock_response({"error": "not found"})

            with pytest.raises(ValueError, match="not found"):
                provider.get_verses("FakeBook", 1, 1)

    def test_error_message_includes_reference(self, provider: WEBProvider) -> None:
        """The ValueError message should include the requested reference."""
        with patch("src.text.web_provider.httpx.get") as mock_get:
            mock_get.return_value = _mock_response({"error": "not found"})

            with pytest.raises(ValueError, match="FakeBook"):
                provider.get_verses("FakeBook", 99, 1)

    def test_http_error_propagates(self, provider: WEBProvider) -> None:
        """An HTTP-level error (e.g. 500) should propagate via raise_for_status."""
        with patch("src.text.web_provider._load_from_cache", return_value={}), \
             patch("src.text.web_provider._store_in_cache"), \
             patch("src.text.web_provider.httpx.get") as mock_get:
            mock = MagicMock(spec=httpx.Response)
            mock.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock()
            )
            mock_get.return_value = mock

            with pytest.raises(httpx.HTTPStatusError):
                provider.get_verses("John", 3, 16)
