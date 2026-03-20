import re

import httpx

from src.data.database import get_session, init_db
from src.data.models import CachedVerse
from src.shared import Verse
from src.text.base import TextProvider

_BASE_URL = "https://bible-api.com"
_VERSION = "web"


class WEBProvider(TextProvider):
    """TextProvider implementation using the World English Bible via bible-api.com.

    Checks a local SQLite cache before making network requests. Any verse
    fetched from the API is stored in the cache for future lookups.
    """

    def __init__(self) -> None:
        """Initialise the provider and ensure the cache table exists."""
        init_db()

    def get_verses(
        self,
        book: str,
        chapter: int,
        verse_start: int,
        verse_end: int = None,
    ) -> list[Verse]:
        """Return verses for the given reference, using the cache where possible.

        For each verse number in the requested range, checks the local cache
        first. Any verse not found in the cache is fetched from bible-api.com
        in a single API call and then stored for future use.
        """
        verse_numbers = (
            list(range(verse_start, verse_end + 1))
            if verse_end is not None
            else [verse_start]
        )

        with get_session() as session:
            cached = _load_from_cache(session, book, chapter, verse_numbers)
            missing = [v for v in verse_numbers if v not in cached]

            if missing:
                fetched = _fetch_from_api(book, chapter, min(missing), max(missing))
                for verse in fetched:
                    _store_in_cache(session, verse)
                    cached[verse.verse] = verse

        return [cached[v] for v in verse_numbers]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_from_cache(
    session, book: str, chapter: int, verse_numbers: list[int]
) -> dict[int, Verse]:
    """Query the cache for the given verse numbers; return a dict keyed by verse number."""
    rows = (
        session.query(CachedVerse)
        .filter(
            CachedVerse.book == book,
            CachedVerse.chapter == chapter,
            CachedVerse.version == _VERSION,
            CachedVerse.verse.in_(verse_numbers),
        )
        .all()
    )
    return {
        row.verse: Verse(book=row.book, chapter=row.chapter, verse=row.verse, text=row.text)
        for row in rows
    }


def _fetch_from_api(book: str, chapter: int, verse_start: int, verse_end: int) -> list[Verse]:
    """Fetch a verse range from bible-api.com and return a list of Verse objects."""
    verse_ref = f"{verse_start}-{verse_end}" if verse_end != verse_start else str(verse_start)
    reference = f"{book}+{chapter}:{verse_ref}"
    url = f"{_BASE_URL}/{reference}?translation=web"

    response = httpx.get(url)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise ValueError(f"bible-api.com: {data['error']} ({book} {chapter}:{verse_ref})")

    return [
        Verse(
            book=v["book_name"],
            chapter=v["chapter"],
            verse=v["verse"],
            text=_clean(v["text"]),
        )
        for v in data["verses"]
    ]


def _store_in_cache(session, verse: Verse) -> None:
    """Insert a verse into the cache, ignoring duplicates."""
    exists = (
        session.query(CachedVerse)
        .filter(
            CachedVerse.book == verse.book,
            CachedVerse.chapter == verse.chapter,
            CachedVerse.verse == verse.verse,
            CachedVerse.version == _VERSION,
        )
        .first()
    )
    if not exists:
        session.add(
            CachedVerse(
                book=verse.book,
                chapter=verse.chapter,
                verse=verse.verse,
                version=_VERSION,
                text=verse.text,
            )
        )


def _clean(text: str) -> str:
    """Strip newlines, tabs, and extra whitespace from verse text."""
    text = re.sub(r"[\n\r\t]", " ", text)
    return re.sub(r" {2,}", " ", text).strip()
