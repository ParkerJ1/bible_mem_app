"""One-off script to measure WhisperXAligner alignment speed.

Creates a single aligner instance (model loaded once) and runs align()
twice on the same audio file, timing each call separately.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aligner.whisperx_aligner import WhisperXAligner

AUDIO = Path("audio_cache/cf006606aa876074.mp3")
TRANSCRIPT = "For God so loved the world."

print("Loading WhisperXAligner (model download may occur on first run)...")
t0 = time.perf_counter()
aligner = WhisperXAligner()
print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n")

for i in range(1, 3):
    t_start = time.perf_counter()
    result = aligner.align(AUDIO, TRANSCRIPT)
    elapsed = time.perf_counter() - t_start
    print(f"Call {i}: {elapsed:.3f}s  ->  {len(result)} words aligned")
    for wt in result:
        print(f"  {wt.word:<12} {wt.start:.3f}s – {wt.end:.3f}s")
    print()
