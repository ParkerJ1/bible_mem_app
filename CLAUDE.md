# bible-mem-app

This is a Bible memorisation app that works by allowing users to select verses, reads those verses to users and has the user say the verse back. This start with small segments and, as the user's accuracy increases, increases segment length, until the user can recite full passages. 

This app is written in Python and served through a web interface. 

---

## Goals and Constraints

- **Primary goal**: A working personal memorisation app
- **Secondary goal**: Potential commercial product (requires ESV licensing - see below)

### ESV Copyright Constraint

The ESV Bible is not freely licensed. The ESV API (api.esv.org) permits non-commercial use with local caching of up to 500 verses. It does not clearly permit storing the ESV audio files locally.

The app is built with a modular text and audio provider architecture so that:

- Phase 1 uses the World English Bible (WEB), which is public domain and has no storage restrictions
- Phase 2 adds an ESV provider, activated when a valid ESV API key is present
- Phase 3 (commercial) would require a direct license from Crossway

Never hardcode ESV text or store ESV audio files in the repository.

---

## Architecture Overview

The application is structured as a set of independent, swappable modules behind abstract interfaces, connected by a FastAPI backend, served to a browser-based frontend.

### Modules

**Text Provider** (`src/text/`)
Abstract interface: given a reference (e.g. "John 3:16"), return the verse text. Note that this can be multiple verses.

- `web_provider.py` - World English Bible, fetched from a public API and cached locally in SQLite
- `esv_provider.py` - ESV API, requires `ESV_API_KEY` in environment, respects the 500-verse cache limit

**Segmenter** (`src/segmenter/`)
Takes a verse string and a word count length target and  returns an ordered list of text segments of that word count length. Note that the word count length is not only numerical but can also be "whole verse" or "whole passage"
Segmentation is punctuation-guided and word-count-constrained. This module is self-contained and independently testable.

- `segmenter.py` - rule-based implementation
- Levels map to approximate word counts: 3, 5, 8, 12, whole verse, whole passage, reference-only

**TTS Provider** (`src/tts/`)
Abstract interface: given a full verse or passage text string, return a single audio file (bytes or a file path) of the complete text read naturally.
Providers always generate or fetch audio for the full verse - never for short segments.
Segment slicing is handled downstream by the Session Manager using Aligner timestamps.

- `google_tts_provider.py` - Google Cloud TTS, for WEB text
- `elevenlabs_provider.py` - ElevenLabs API, alternative TTS, also for WEB text
- `esv_audio_provider.py` - fetches MP3 from ESV API audio endpoint (does not store permanently)

**Aligner** (`src/aligner/`)
Core infrastructure used by all audio providers. Takes a full-verse audio file and its transcript, returns word-level timestamps. The Session Manager uses these timestamps to slice the audio into segments of the appropriate length for the user's current proficiency level.
This means both the TTS path and the ESV recorded-audio path go through the same alignment and slicing pipeline. The Aligner is not ESV-specific.

- `whisperx_aligner.py` - uses WhisperX for forced alignment

**Speech Recogniser** (`src/sr/`)
Takes a recorded audio clip from the user, returns a transcript string.

- `whisper_sr.py` - runs a Whisper model locally (on-device)
- Model size is configurable; default is `base` for balance of speed and accuracy

**Scorer** (`src/scorer/`)
Compares a user transcript against the expected segment text. 
The scorer uses sequence alignment (not simple sequential matching) to compare the user's transcript against the expected text. This ensures that extra or inserted words do not cause correct words later in the sentence to be marked wrong. The score reflects only accuracy on the expected words. The diff should distinguish between missing words, correct words, and inserted words so the user gets useful feedback. Returns a score and a word-level diff.

- Target: word-perfect scoring (the goal of Bible memorisation)
- Normalises for case, punctuation, and common transcription noise before scoring
- Returns structured result: `{score: float, expected: str, got: str, diff: list}`

**Progression Engine** (`src/progression/`)
Tracks per-user, per-verse proficiency levels and decides when to advance or drop back.

- Advancing: success on the current level on two consecutive days
- Holding: single failure does not change level
- Dropping: failure on the same level on several consecutive days (configurable, default 3)
- Levels: `[3-word, 5-word, 8-word, 12-word, full-verse, full-passage, reference-only]`

**Session Manager** (`src/session/`)
Orchestrates a single practice session end to end.
Given a user and a verse reference, it drives the loop: fetch segment, play audio, record user, score, update state.
Responsible for slicing full-verse audio into the correct segment for the user's current level, using timestamps from the Aligner. This is the only layer that knows what segment length is appropriate - neither the TTS provider nor the Aligner needs this knowledge.

**Data Layer** (`src/data/`)
SQLite via SQLAlchemy (no external database dependency for local use).

- Tables: `users`, `verse_lists`, `verses`, `sessions`, `proficiency_records`
- Migrations managed with Alembic

**API Layer** (`src/api/`)
FastAPI application. Exposes REST endpoints consumed by the frontend.

- `routers/verses.py` - add/remove/list verses
- `routers/sessions.py` - start and advance a practice session
- `routers/progress.py` - retrieve proficiency history

**Frontend** (`frontend/`)
Browser-based. HTML/CSS/JavaScript, kept deliberately simple.
Handles microphone recording, audio playback, and visual feedback.
No heavy frontend framework in Phase 1 - plain JS is sufficient.

---

## Tech Stack

| Concern          | Choice                                             |
| ---------------- | -------------------------------------------------- |
| Language         | Python 3.11+                                       |
| Package manager  | uv                                                 |
| Web framework    | FastAPI                                            |
| Database         | SQLite + SQLAlchemy                                |
| Migrations       | Alembic                                            |
| SR model         | OpenAI Whisper (via `whisper` or `faster-whisper`) |
| Forced alignment | WhisperX                                           |
| TTS (default)    | Google Cloud TTS or ElevenLabs                     |
| Testing          | pytest                                             |
| Config/secrets   | python-dotenv, `.env` file                         |

---

## Directory Structure

```
bible-mem-app/
    CLAUDE.md
    README.md
    .env                    # never committed
    .env.example            # committed, no real values
    .gitignore
    pyproject.toml
    src/
        text/
        segmenter/
        tts/
        aligner/
        sr/
        scorer/
        progression/
        session/
        data/
        api/
    frontend/
        index.html
        static/
            css/
            js/
    tests/
        test_segmenter.py
        test_scorer.py
        test_progression.py
        ...
    scripts/               # one-off utilities, not part of the app
    docs/
        architecture.md
        esv_notes.md
```

---

## Key Commands

```bash
# Install dependencies
uv sync

# Run the development server
uv run uvicorn src.api.main:app --reload

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src

# Apply database migrations
uv run alembic upgrade head
```

---

## Coding Conventions

- Type hints on all function signatures: `def foo(x: int) -> str:`
- Docstrings on all public functions and classes (one-line minimum)
- Abstract base classes (`abc.ABC`) for all swappable providers (Text, TTS, SR, Aligner)
- Concrete implementations live in the same module directory as the abstract class
- No business logic in API routers - routers call session/service layer only
- All configuration via environment variables, accessed through a central `src/config.py`
- Tests live in `tests/` and mirror the `src/` structure

---

## Environment Variables

See `.env.example` for the full list. Key variables:

```
ESV_API_KEY=          # optional; enables ESV text provider
GOOGLE_TTS_KEY=       # optional; enables Google TTS provider
ELEVENLABS_API_KEY=   # optional; enables ElevenLabs TTS provider
TEXT_PROVIDER=web     # "web" or "esv"
TTS_PROVIDER=google   # "google" or "elevenlabs" or "esv"
WHISPER_MODEL=base    # "tiny", "base", "small", "medium"
DATABASE_URL=sqlite:///./bible_mem.db
```

---

## Notes for Claude Code

- When implementing a new provider, always implement the abstract base class first, then the concrete class
- The Segmenter module is intentionally self-contained and should have the most thorough test coverage of any module - it is the most novel algorithmic contribution
- Do not import from `src.api` inside any other module - the API layer depends on everything else, nothing depends on it
- The ESV provider should be gated: if `ESV_API_KEY` is not set, importing the module should raise a clear error, not fail silently
- The Aligner is core infrastructure, not an ESV-specific component. Every audio path (TTS-generated or ESV recorded) produces a full-verse audio file that is then aligned and sliced. No provider ever generates short segment audio directly
- Audio slicing logic belongs exclusively in the Session Manager. Neither the TTS provider nor the Aligner should know anything about segment lengths or proficiency levels
