import base64
import hashlib
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from src.tts.base import TTSProvider

load_dotenv()

_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
_AUDIO_CACHE = Path("audio_cache")


class GoogleTTSProvider(TTSProvider):
    """TTS provider backed by the Google Cloud Text-to-Speech REST API.

    Requires GOOGLE_TTS_KEY to be set in the environment or .env file.
    Audio files are cached in audio_cache/ using a hash of the input text,
    so repeated requests for the same text never make a second API call.
    """

    def __init__(self) -> None:
        """Load the API key and ensure the audio cache directory exists."""
        api_key = os.getenv("GOOGLE_TTS_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_TTS_KEY is not set. Add it to your .env file."
            )
        self._api_key = api_key
        _AUDIO_CACHE.mkdir(exist_ok=True)

    def synthesise(self, text: str) -> Path:
        """Synthesise text via Google Cloud TTS and return the path to the MP3 file.

        Checks the local cache first; only calls the API when the audio for
        this exact text has not been generated before.
        """
        if not text:
            raise ValueError("text must not be empty")

        dest = _AUDIO_CACHE / f"{_hash(text)}.mp3"
        if dest.exists():
            return dest

        audio_bytes = self._call_api(text)
        dest.write_bytes(audio_bytes)
        return dest

    def _call_api(self, text: str) -> bytes:
        """Make a single synthesis request and return raw MP3 bytes."""
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": "en-US-Wavenet-D",
            },
            "audioConfig": {
                "audioEncoding": "MP3",
            },
        }
        response = httpx.post(
            _API_URL,
            params={"key": self._api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return base64.b64decode(data["audioContent"])


def _hash(text: str) -> str:
    """Return a short, stable hex digest of the input text for use as a filename."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
