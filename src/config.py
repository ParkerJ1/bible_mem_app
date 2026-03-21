"""Central configuration module.

All environment variables are read here. Other modules import from this
module rather than calling os.getenv() directly.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Text / audio providers
TEXT_PROVIDER: str = os.getenv("TEXT_PROVIDER", "web")
TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "google")

# Speech recognition
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# Scoring
SCORING_MODE: str = os.getenv("SCORING_MODE", "strict")  # "strict" or "lenient"

# Database
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./bible_mem.db")
