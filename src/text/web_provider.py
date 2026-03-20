import re

import httpx

from src.shared import Verse
from src.text.base import TextProvider

_BASE_URL = "https://bible-api.com"


class WEBProvider(TextProvider):
    """TextProvider implementation using the World English Bible via bible-api.com."""

    def get_verses(
        self,
        book: str,
        chapter: int,
        verse_start: int,
        verse_end: int = None,
    ) -> list[Verse]:
        """Fetch one or more WEB verses from bible-api.com.

        Constructs a range query when verse_end is provided, otherwise fetches
        a single verse. Returns an ordered list of Verse objects with
        whitespace and formatting characters stripped from the text.
        """
        verse_ref = (
            f"{verse_start}-{verse_end}" if verse_end is not None else str(verse_start)
        )
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


def _clean(text: str) -> str:
    """Strip newlines, tabs, and extra whitespace from verse text."""
    text = re.sub(r"[\n\r\t]", " ", text)
    return re.sub(r" {2,}", " ", text).strip()
