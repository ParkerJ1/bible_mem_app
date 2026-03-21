"""SequenceAligner — sequence-alignment-based verse scorer.

Uses difflib.SequenceMatcher (LCS) so that inserted words in the middle of
the user's transcript do not cause correct words further along to be marked
wrong.  The score reflects only accuracy on the *expected* words.

Two modes are supported, controlled by SCORING_MODE in the environment:
  strict  — exact word match after case/punctuation normalisation only.
  lenient — both sides are lemmatised before comparison so that tense and
             plural differences (loved/love, worlds/world) are ignored.
"""

import re
import string
from difflib import SequenceMatcher

import nltk
from nltk.stem import WordNetLemmatizer

import src.config as config
from src.scorer.base import DiffToken, Scorer, ScorerResult

# Ensure WordNet data is available at import time (no-op if already downloaded)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

_lemmatizer = WordNetLemmatizer()


class SequenceAligner(Scorer):
    """Scores a recitation attempt using sequence alignment.

    Normalises both strings (lowercase, strip punctuation, collapse
    whitespace), then optionally lemmatises in lenient mode before running
    SequenceMatcher to produce a word-level diff.

    The scoring mode is read from src.config.SCORING_MODE but can be
    overridden per-instance by passing mode= to the constructor.

    Score formula:
        score = correct_words / max(1, total_expected_words)

    Diff token statuses:
        "correct"  — word was matched in expected and got
        "missing"  — word is in expected but absent from got
        "inserted" — word appears in got but was not expected
    """

    def __init__(self, mode: str | None = None) -> None:
        """Initialise the scorer.

        Args:
            mode: Override the scoring mode. Must be "strict" or "lenient".
                  Defaults to the value of SCORING_MODE in the environment.
        """
        resolved = mode or config.SCORING_MODE
        if resolved not in ("strict", "lenient"):
            raise ValueError(f"Unknown scoring mode: {resolved!r}. Use 'strict' or 'lenient'.")
        self._mode = resolved

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, expected: str, got: str) -> ScorerResult:
        """Score the user's transcript against the expected verse text."""
        norm_expected = _normalise(expected)
        norm_got = _normalise(got)

        expected_words = norm_expected.split()
        got_words = norm_got.split()

        if self._mode == "lenient":
            cmp_expected = [_lemmatize(w) for w in expected_words]
            cmp_got = [_lemmatize(w) for w in got_words]
        else:
            cmp_expected = expected_words
            cmp_got = got_words

        diff = _build_diff(expected_words, got_words, cmp_expected, cmp_got)

        if not expected_words:
            score = 1.0 if not got_words else 0.0
        else:
            correct = sum(1 for t in diff if t.status == "correct")
            score = correct / len(expected_words)

        return ScorerResult(
            score=round(score, 4),
            expected=norm_expected,
            got=norm_got,
            diff=diff,
        )

    @property
    def mode(self) -> str:
        """The active scoring mode: 'strict' or 'lenient'."""
        return self._mode


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower()
    # Remove all punctuation characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lemmatize(word: str) -> str:
    """Return the WordNet lemma of word, trying verb then noun form."""
    # Try verb lemma first (handles tense), fall back to noun (handles plurals)
    verb = _lemmatizer.lemmatize(word, pos="v")
    if verb != word:
        return verb
    return _lemmatizer.lemmatize(word, pos="n")


def _build_diff(
    expected_words: list[str],
    got_words: list[str],
    cmp_expected: list[str],
    cmp_got: list[str],
) -> list[DiffToken]:
    """Build the word-level diff using SequenceMatcher on the comparison lists.

    Display words come from expected_words / got_words (original, normalised)
    while alignment is performed on cmp_expected / cmp_got (optionally lemmatised).
    """
    matcher = SequenceMatcher(None, cmp_expected, cmp_got, autojunk=False)
    diff: list[DiffToken] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for w in expected_words[i1:i2]:
                diff.append(DiffToken(word=w, status="correct"))

        elif tag == "delete":
            # Words in expected that have no counterpart in got
            for w in expected_words[i1:i2]:
                diff.append(DiffToken(word=w, status="missing"))

        elif tag == "insert":
            # Words in got that were not expected
            for w in got_words[j1:j2]:
                diff.append(DiffToken(word=w, status="inserted"))

        elif tag == "replace":
            # Treat as: expected words missing, then got words inserted
            for w in expected_words[i1:i2]:
                diff.append(DiffToken(word=w, status="missing"))
            for w in got_words[j1:j2]:
                diff.append(DiffToken(word=w, status="inserted"))

    return diff
