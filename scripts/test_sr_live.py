"""Record 5 seconds of microphone audio, then transcribe it with WhisperSpeechRecogniser.

Usage:
    uv run python scripts/test_sr_live.py           # use default input device
    uv run python scripts/test_sr_live.py --device 4  # use device index 4
"""

import argparse
import sys
import tempfile
import wave
from pathlib import Path

import sounddevice as sd
import numpy as np

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sr.whisper_sr import WhisperSpeechRecogniser

DURATION    = 5       # seconds
SAMPLE_RATE = 16_000  # Hz — Whisper's native sample rate


def print_input_devices() -> None:
    print("Available input devices:")
    devices = sd.query_devices()
    default_idx = sd.default.device[0]  # (input, output) tuple
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " <-- default" if i == default_idx else ""
            print(f"  [{i}] {dev['name']}{marker}")
    print()


def record(duration: int, sample_rate: int, device: int | None) -> np.ndarray:
    if device is None:
        default_idx = sd.default.device[0]
        default_name = sd.query_devices(default_idx)["name"]
        print(f"Using default input device: [{default_idx}] {default_name}")
    else:
        print(f"Using input device: [{device}] {sd.query_devices(device)['name']}")

    print(f"Recording for {duration}s... speak now.")
    audio = sd.rec(
        frames=duration * sample_rate,
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()
    print("Recording complete.")
    return audio


def save_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=None, help="Input device index (see list above)")
    args = parser.parse_args()

    print_input_devices()
    audio = record(DURATION, SAMPLE_RATE, device=args.device)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)

    save_wav(audio, SAMPLE_RATE, out_path)
    print(f"Saved to: {out_path}")

    print("Loading Whisper model and transcribing...")
    sr = WhisperSpeechRecogniser()
    result = sr.recognise(out_path)

    print()
    print(f"Transcript:  {result.text!r}")
    print(f"Confidence:  {result.confidence:.3f}")


if __name__ == "__main__":
    main()
