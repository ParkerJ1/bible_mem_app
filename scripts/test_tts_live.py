"""One-off script to verify GoogleTTSProvider against the live API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tts.google_tts_provider import GoogleTTSProvider

provider = GoogleTTSProvider()
path = provider.synthesise("For God so loved the world.")
print(f"Audio saved to: {path}")
