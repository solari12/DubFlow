from __future__ import annotations

import wave
from pathlib import Path

from dubflow_worker.models.dubbing_quality import DubbingQualitySettings, is_severe_overflow
from dubflow_worker.models.tts import TTSResult
from dubflow_worker.pipeline.tts_feedback import synthesize_with_duration_feedback
from dubflow_worker.translation.base import TranslationEngine
from dubflow_worker.tts.base import TTSEngine


class FakeTTS(TTSEngine):
    name = "fake-tts"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, language: str, output_path: Path, speaker: str | None = None) -> TTSResult:
        self.calls.append(text)
        duration = {"Original long sentence.": 2.0, "Short phrase.": 1.0, "Short.": 0.8}.get(text, 2.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x01\x00" * round(duration * 16000))
        return TTSResult(output_path, duration, 16000, self.name, language, speaker, 0.01)


class FakeTranslator(TranslationEngine):
    name = "fake-translation"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        return "Original long sentence."

    def shorten_for_dubbing(
        self, translated_text: str, source_text: str, source_language: str,
        target_language: str, max_expansion_ratio: float,
    ) -> str:
        self.calls += 1
        return "Short phrase." if self.calls == 1 else "Short."


def _segment() -> list[dict]:
    return [{
        "id": 3,
        "start": 1.0,
        "end": 2.0,
        "speaker": "SPEAKER_01",
        "source_text": "A sentence.",
        "target_text": "Original long sentence.",
    }]


def test_duration_feedback_retries_once_and_keeps_speaker(tmp_path: Path) -> None:
    tts = FakeTTS()
    translator = FakeTranslator()
    result = synthesize_with_duration_feedback(
        _segment(), source_language="en", target_language="vi", tts_engine=tts,
        translation_engine=translator, output_dir=tmp_path,
    )[0]
    assert tts.calls == ["Original long sentence.", "Short phrase."]
    assert result.original_translation == "Original long sentence."
    assert result.final_translation == "Short phrase."
    assert result.tts_duration_before_fit == 2.0
    assert result.tts_duration_after_fit == 1.0
    assert result.translation_retry_count == 1
    assert result.overflow is False
    assert result.speaker == "SPEAKER_01"
    assert result.audio_path.is_file()


def test_retry_limit_is_configurable_and_capped_at_two(tmp_path: Path) -> None:
    class SlowFirstRetry(FakeTTS):
        def synthesize(self, text: str, language: str, output_path: Path, speaker: str | None = None) -> TTSResult:
            result = super().synthesize(text, language, output_path, speaker)
            if text == "Short phrase.":
                return TTSResult(result.audio_path, 1.4, result.sample_rate, result.engine, result.language, result.speaker, result.runtime)
            return result

    tts = SlowFirstRetry()
    translator = FakeTranslator()
    result = synthesize_with_duration_feedback(
        _segment(), source_language="en", target_language="vi", tts_engine=tts,
        translation_engine=translator, output_dir=tmp_path,
        settings=DubbingQualitySettings(max_translation_retries=2),
    )[0]
    assert result.translation_retry_count == 2
    assert result.final_translation == "Short."
    assert tts.calls == ["Original long sentence.", "Short phrase.", "Short."]


def test_missing_rephraser_preserves_severe_overflow_without_truncating(tmp_path: Path) -> None:
    class NoRephrase(FakeTranslator):
        def shorten_for_dubbing(self, *args, **kwargs) -> str:
            raise NotImplementedError("no local rephraser")

    tts = FakeTTS()
    result = synthesize_with_duration_feedback(
        _segment(), source_language="en", target_language="vi", tts_engine=tts,
        translation_engine=NoRephrase(), output_dir=tmp_path,
    )[0]
    assert result.translation_retry_count == 0
    assert result.overflow is True
    assert result.severe_overflow is True
    assert result.overflow_duration == 1.0
    assert result.tts_duration_after_fit == 2.0
    assert result.audio_path.stat().st_size > 32000


def test_quality_settings_reject_extreme_speedup_and_unbounded_retries() -> None:
    try:
        DubbingQualitySettings(min_time_stretch_ratio=0.5)
    except ValueError as exc:
        assert "between 0.75 and 1.0" in str(exc)
    else:
        raise AssertionError("extreme speedup must be rejected")
    try:
        DubbingQualitySettings(max_translation_retries=3)
    except ValueError as exc:
        assert "between 0 and 2" in str(exc)
    else:
        raise AssertionError("retry count must remain bounded")


def test_severe_overflow_threshold_is_configurable() -> None:
    settings = DubbingQualitySettings(max_overflow_ratio=1.4)
    assert not is_severe_overflow(1.3, 1.0, settings)
    assert is_severe_overflow(1.5, 1.0, settings)
