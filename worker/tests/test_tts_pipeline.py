from __future__ import annotations

import wave
from pathlib import Path

import pytest

from dubflow_worker.models.tts import TTSResult
from dubflow_worker.pipeline.synthesize import synthesize_translated_transcript
from dubflow_worker.tts.base import TTSEngine


class FakeTTSEngine(TTSEngine):
    name = "fake-tts"
    device = "cpu"

    def __init__(self, *, fail_text: str | None = None) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.fail_text = fail_text

    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Path,
        speaker: str | None = None,
    ) -> TTSResult:
        self.calls.append((text, language, speaker))
        if text == self.fail_text:
            raise RuntimeError("synthetic engine failure")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
        with wave.open(str(output_path), "rb") as wav_file:
            duration = wav_file.getnframes() / wav_file.getframerate()
            sample_rate = wav_file.getframerate()
        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=sample_rate,
            engine=self.name,
            language=language,
            speaker=speaker,
            runtime=0.01,
        )


def _transcript(*segments: dict) -> dict:
    return {"segments": list(segments)}


def _segment(segment_id: int = 7, **overrides) -> dict:
    item = {
        "id": segment_id,
        "start": 1.25,
        "end": 3.5,
        "speaker": "SPEAKER_01",
        "source_text": "Hello.",
        "target_text": "Xin chào.",
        "error": None,
    }
    item.update(overrides)
    return item


def test_basic_tts_invocation_and_audio_file_creation(tmp_path: Path) -> None:
    engine = FakeTTSEngine()
    result = synthesize_translated_transcript(
        _transcript(_segment()), language="vi", output_dir=tmp_path, engine=engine
    )[0]

    assert engine.calls == [("Xin chào.", "vi", "SPEAKER_01")]
    assert result.result is not None
    assert result.result.audio_path.is_file()
    assert result.result.audio_path.stat().st_size > 44


def test_generated_wav_has_valid_metadata_and_positive_duration(tmp_path: Path) -> None:
    result = synthesize_translated_transcript(
        _transcript(_segment()),
        language="vi",
        output_dir=tmp_path,
        engine=FakeTTSEngine(),
    )[0].result

    assert result is not None
    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() > 0
        assert audio.getframerate() == 16000
    assert result.duration == 1.0
    assert result.duration > 0


def test_sample_rate_language_and_speaker_propagate(tmp_path: Path) -> None:
    result = synthesize_translated_transcript(
        _transcript(_segment()),
        language="vi_VN",
        output_dir=tmp_path,
        engine=FakeTTSEngine(),
    )[0].result

    assert result is not None
    assert result.sample_rate == 16000
    assert result.language == "vi_VN"
    assert result.speaker == "SPEAKER_01"


def test_segment_id_and_timestamps_are_preserved(tmp_path: Path) -> None:
    result = synthesize_translated_transcript(
        _transcript(_segment(42, start=4.125, end=8.75)),
        language="vi",
        output_dir=tmp_path,
        engine=FakeTTSEngine(),
    )[0]

    assert (result.id, result.start, result.end) == (42, 4.125, 8.75)
    assert result.speaker == "SPEAKER_01"


def test_empty_text_is_reported_without_invoking_engine(tmp_path: Path) -> None:
    engine = FakeTTSEngine()
    result = synthesize_translated_transcript(
        _transcript(_segment(target_text="")),
        language="vi",
        output_dir=tmp_path,
        engine=engine,
    )[0]

    assert not engine.calls
    assert result.result is None
    assert result.error == "Translated target_text is empty"
    assert result.id == 7


def test_synthesis_failure_is_recorded_and_later_segments_continue(tmp_path: Path) -> None:
    engine = FakeTTSEngine(fail_text="Không tạo được.")
    results = synthesize_translated_transcript(
        _transcript(
            _segment(1, target_text="Không tạo được."),
            _segment(2, target_text="Xin chào."),
        ),
        language="vi",
        output_dir=tmp_path,
        engine=engine,
    )

    assert results[0].result is None
    assert results[0].error == "RuntimeError: synthetic engine failure"
    assert results[1].result is not None
    assert results[1].result.audio_path.is_file()


def test_multiple_segments_create_distinct_wavs(tmp_path: Path) -> None:
    results = synthesize_translated_transcript(
        _transcript(_segment(1), _segment(2, target_text="Tạm biệt.")),
        language="vi",
        output_dir=tmp_path,
        engine=FakeTTSEngine(),
    )

    assert len(results) == 2
    assert [result.id for result in results] == [1, 2]
    assert results[0].result is not None and results[0].result.audio_path.is_file()
    assert results[1].result is not None and results[1].result.audio_path.is_file()
    assert results[0].result.audio_path != results[1].result.audio_path


def test_missing_translation_error_is_preserved(tmp_path: Path) -> None:
    result = synthesize_translated_transcript(
        _transcript(
            _segment(target_text=None, error="UnsupportedLanguageError: no route")
        ),
        language="vi",
        output_dir=tmp_path,
        engine=FakeTTSEngine(),
    )[0]

    assert result.result is None
    assert result.error == "UnsupportedLanguageError: no route"


def test_language_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="language"):
        synthesize_translated_transcript(
            _transcript(_segment()), language=" ", output_dir=tmp_path, engine=FakeTTSEngine()
        )
