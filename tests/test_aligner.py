"""Tests for WhisperXAligner.

Unit tests mock the WhisperX library entirely so no model download or GPU
is needed.  The integration test uses the real cached audio file and requires
whisperx and its alignment model to be available.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.aligner.base import Aligner, WordTimestamp
from src.aligner.whisperx_aligner import WhisperXAligner, _extract_timestamps

# ---------------------------------------------------------------------------
# The real audio produced by scripts/test_tts_live.py
# ---------------------------------------------------------------------------
_AUDIO_CACHE = Path("audio_cache")
_LIVE_AUDIO = _AUDIO_CACHE / "cf006606aa876074.mp3"
_LIVE_TRANSCRIPT = "For God so loved the world."


# ---------------------------------------------------------------------------
# Fake WhisperX alignment result helpers
# ---------------------------------------------------------------------------

def _make_word_segment(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end}


def _make_result(word_segments: list[dict]) -> dict:
    return {"word_segments": word_segments, "segments": []}


def _fake_audio_array(seconds: float = 3.0, sample_rate: int = 16_000):
    """Return a list that mimics a numpy audio array of the given duration."""
    import numpy as np
    return np.zeros(int(seconds * sample_rate), dtype="float32")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aligner(monkeypatch: pytest.MonkeyPatch) -> WhisperXAligner:
    """WhisperXAligner with whisperx.load_align_model patched out."""
    mock_model = MagicMock()
    mock_metadata = MagicMock()
    with patch("src.aligner.whisperx_aligner.whisperx.load_align_model",
               return_value=(mock_model, mock_metadata)):
        instance = WhisperXAligner()
    return instance


# ---------------------------------------------------------------------------
# _extract_timestamps helper
# ---------------------------------------------------------------------------

class TestExtractTimestamps:
    def test_well_formed_segments_extracted(self) -> None:
        result = _make_result([
            _make_word_segment("For", 0.0, 0.3),
            _make_word_segment("God", 0.4, 0.7),
            _make_word_segment("so",  0.8, 0.9),
        ])
        out = _extract_timestamps(result)
        assert len(out) == 3
        assert out[0] == WordTimestamp(word="For", start=0.0, end=0.3)
        assert out[1] == WordTimestamp(word="God", start=0.4, end=0.7)

    def test_missing_start_is_dropped(self) -> None:
        result = _make_result([
            {"word": "hello", "end": 0.5},          # no start
            {"word": "world", "start": 0.6, "end": 1.0},
        ])
        out = _extract_timestamps(result)
        assert len(out) == 1
        assert out[0].word == "world"

    def test_missing_end_is_dropped(self) -> None:
        result = _make_result([
            {"word": "hello", "start": 0.0},         # no end
            {"word": "world", "start": 0.6, "end": 1.0},
        ])
        out = _extract_timestamps(result)
        assert len(out) == 1
        assert out[0].word == "world"

    def test_empty_word_is_dropped(self) -> None:
        result = _make_result([
            {"word": "  ", "start": 0.0, "end": 0.2},
            {"word": "world", "start": 0.3, "end": 0.8},
        ])
        out = _extract_timestamps(result)
        assert len(out) == 1
        assert out[0].word == "world"

    def test_empty_word_segments_returns_empty_list(self) -> None:
        assert _extract_timestamps(_make_result([])) == []

    def test_missing_word_segments_key_returns_empty_list(self) -> None:
        assert _extract_timestamps({}) == []

    def test_timestamps_are_floats(self) -> None:
        result = _make_result([_make_word_segment("test", 1, 2)])
        out = _extract_timestamps(result)
        assert isinstance(out[0].start, float)
        assert isinstance(out[0].end, float)

    def test_words_are_stripped(self) -> None:
        result = _make_result([{"word": "  hello  ", "start": 0.0, "end": 0.5}])
        out = _extract_timestamps(result)
        assert out[0].word == "hello"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_implements_aligner_abc(self, aligner: WhisperXAligner) -> None:
        assert isinstance(aligner, Aligner)

    def test_load_align_model_called_with_english(self,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
        with patch("src.aligner.whisperx_aligner.whisperx.load_align_model",
                   return_value=(MagicMock(), MagicMock())) as mock_load:
            WhisperXAligner()
        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs.get("language_code") == "en"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_audio_raises_file_not_found(self,
                                                  aligner: WhisperXAligner,
                                                  tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing.mp3"):
            aligner.align(tmp_path / "missing.mp3", "some transcript")

    def test_empty_transcript_returns_empty_list(self,
                                                  aligner: WhisperXAligner,
                                                  tmp_path: Path) -> None:
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        result = aligner.align(audio, "   ")
        assert result == []

    def test_empty_transcript_does_not_call_whisperx(self,
                                                      aligner: WhisperXAligner,
                                                      tmp_path: Path) -> None:
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        with patch("src.aligner.whisperx_aligner.whisperx.align") as mock_align:
            aligner.align(audio, "")
        mock_align.assert_not_called()


# ---------------------------------------------------------------------------
# align() behaviour (mocked WhisperX)
# ---------------------------------------------------------------------------

class TestAlignBehaviour:
    def _patch_whisperx(self, aligner: WhisperXAligner, word_segments: list[dict]):
        """Context manager that patches whisperx.load_audio and whisperx.align."""
        import numpy as np
        audio = np.zeros(48_000, dtype="float32")  # 3 seconds at 16 kHz

        load_audio_patch = patch(
            "src.aligner.whisperx_aligner.whisperx.load_audio",
            return_value=audio,
        )
        align_patch = patch(
            "src.aligner.whisperx_aligner.whisperx.align",
            return_value=_make_result(word_segments),
        )
        return load_audio_patch, align_patch

    def test_returns_list_of_word_timestamps(self,
                                              aligner: WhisperXAligner,
                                              tmp_path: Path) -> None:
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        segs = [
            _make_word_segment("For",   0.0, 0.3),
            _make_word_segment("God",   0.4, 0.7),
            _make_word_segment("loved", 0.8, 1.2),
        ]
        p1, p2 = self._patch_whisperx(aligner, segs)
        with p1, p2:
            result = aligner.align(audio, "For God loved")

        assert len(result) == 3
        assert all(isinstance(t, WordTimestamp) for t in result)

    def test_word_order_preserved(self,
                                   aligner: WhisperXAligner,
                                   tmp_path: Path) -> None:
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        segs = [
            _make_word_segment("the",   0.0, 0.1),
            _make_word_segment("lord",  0.2, 0.5),
            _make_word_segment("is",    0.6, 0.7),
            _make_word_segment("my",    0.8, 0.9),
            _make_word_segment("shepherd", 1.0, 1.6),
        ]
        p1, p2 = self._patch_whisperx(aligner, segs)
        with p1, p2:
            result = aligner.align(audio, "the lord is my shepherd")

        words = [t.word for t in result]
        assert words == ["the", "lord", "is", "my", "shepherd"]

    def test_timestamps_match_whisperx_output(self,
                                               aligner: WhisperXAligner,
                                               tmp_path: Path) -> None:
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        segs = [_make_word_segment("God", 1.23, 1.89)]
        p1, p2 = self._patch_whisperx(aligner, segs)
        with p1, p2:
            result = aligner.align(audio, "God")

        assert result[0].start == pytest.approx(1.23)
        assert result[0].end == pytest.approx(1.89)

    def test_segment_passed_with_full_audio_duration(self,
                                                      aligner: WhisperXAligner,
                                                      tmp_path: Path) -> None:
        """align() must pass a single segment spanning 0.0 → full duration."""
        import numpy as np
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake")
        # 3 seconds of audio at 16 kHz
        fake_audio = np.zeros(48_000, dtype="float32")

        with patch("src.aligner.whisperx_aligner.whisperx.load_audio",
                   return_value=fake_audio), \
             patch("src.aligner.whisperx_aligner.whisperx.align",
                   return_value=_make_result([])) as mock_align:
            aligner.align(audio_file, "hello world")

        call_args = mock_align.call_args
        segments_arg = call_args[0][0]
        assert len(segments_arg) == 1
        assert segments_arg[0]["text"] == "hello world"
        assert segments_arg[0]["start"] == pytest.approx(0.0)
        assert segments_arg[0]["end"] == pytest.approx(3.0)

    def test_unaligned_words_omitted(self,
                                      aligner: WhisperXAligner,
                                      tmp_path: Path) -> None:
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        segs = [
            _make_word_segment("For", 0.0, 0.3),
            {"word": "God", "start": None, "end": None},   # unaligned
            _make_word_segment("loved", 0.8, 1.2),
        ]
        p1, p2 = self._patch_whisperx(aligner, segs)
        with p1, p2:
            result = aligner.align(audio, "For God loved")

        assert len(result) == 2
        assert result[0].word == "For"
        assert result[1].word == "loved"


# ---------------------------------------------------------------------------
# Integration test — real audio + real WhisperX model
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not _LIVE_AUDIO.exists(),
    reason="Live audio file not found in audio_cache/",
)
class TestWhisperXAlignerIntegration:
    """Runs WhisperX against the real TTS-generated audio file.

    Requires the whisperx alignment model to be downloadable and the file
    audio_cache/cf006606aa876074.mp3 to exist.
    """

    @pytest.fixture(scope="class")
    def timestamps(self) -> list[WordTimestamp]:
        a = WhisperXAligner()
        return a.align(_LIVE_AUDIO, _LIVE_TRANSCRIPT)

    def test_returns_word_timestamps(self, timestamps: list[WordTimestamp]) -> None:
        assert all(isinstance(t, WordTimestamp) for t in timestamps)

    def test_non_empty_result(self, timestamps: list[WordTimestamp]) -> None:
        assert len(timestamps) > 0

    def test_starts_are_non_negative(self, timestamps: list[WordTimestamp]) -> None:
        assert all(t.start >= 0.0 for t in timestamps)

    def test_end_after_start(self, timestamps: list[WordTimestamp]) -> None:
        assert all(t.end >= t.start for t in timestamps)

    def test_timestamps_are_ordered(self, timestamps: list[WordTimestamp]) -> None:
        starts = [t.start for t in timestamps]
        assert starts == sorted(starts)

    def test_key_words_present(self, timestamps: list[WordTimestamp]) -> None:
        words = {t.word.lower().strip(".,") for t in timestamps}
        for expected in ("god", "world"):
            assert expected in words, f"'{expected}' not found in {words}"
