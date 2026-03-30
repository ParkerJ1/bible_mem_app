"""Print word timestamps for a prepared passage stored in the database.

Usage:
    uv run python scripts/inspect_timestamps.py
    uv run python scripts/inspect_timestamps.py "Romans 8:28"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.database import get_session
from src.data.models import PreparedPassage
from src.session.session_manager import _deserialise_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("passage_ref", nargs="?", default="John 3:16",
                        help="Passage reference to inspect (default: John 3:16)")
    args = parser.parse_args()

    with get_session() as session:
        pp = session.query(PreparedPassage).filter_by(passage_ref=args.passage_ref).first()
        if pp is None:
            print(f"No prepared passage found for {args.passage_ref!r}.")
            print("Add the verse via the app or POST /verses first.")
            sys.exit(1)
        # Read all attributes inside the session to avoid DetachedInstanceError
        passage_ref = pp.passage_ref
        full_text = pp.full_text
        audio_path = pp.audio_path
        timestamps = _deserialise_timestamps(pp.timestamps_json)

    print(f"Passage : {passage_ref}")
    print(f"Text    : {full_text}")
    print(f"Audio   : {audio_path}")
    print(f"Words   : {len(timestamps)}")
    print()

    col_w = max((len(t.word) for t in timestamps), default=4)
    header = f"{'#':>4}  {'Word':<{col_w}}  {'Start':>7}  {'End':>7}  {'Dur':>6}"
    print(header)
    print("-" * len(header))

    for i, t in enumerate(timestamps):
        dur = t.end - t.start
        print(f"{i:>4}  {t.word:<{col_w}}  {t.start:>7.3f}  {t.end:>7.3f}  {dur:>6.3f}s")


if __name__ == "__main__":
    main()
