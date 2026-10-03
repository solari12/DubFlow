from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from dubflow_worker.tts.benchmark_providers_v1 import PROVIDER_REGISTRY
from dubflow_worker.tts.benchmark_v1 import (
    TTSBenchmarkFixture,
    benchmark_provider,
    validate_wav,
    write_provider_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "output/tts-benchmark-v1/fixture.json"


class WavProvider:
    provider = "fixture-provider"
    model = "fixture-model"
    language = "vi"
    device = "cpu"
    model_size_bytes = 123

    def synthesize(self, text: str, output_path: Path) -> None:
        samples = struct.pack("<1600h", *([0] * 1600))
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(samples)


def test_provider_registration_contains_piper_and_opt_in_korva_only():
    assert set(PROVIDER_REGISTRY) == {"piper", "korva"}


def test_benchmark_fixture_loads_approved_and_tts_only_text():
    fixture = TTSBenchmarkFixture.load(FIXTURE_PATH)
    assert fixture.language == "vi"
    assert "Hiragana" in fixture.original_text
    assert "hi-ra-ga-na" in fixture.normalized_text
    assert fixture.normalization_note
    approved = json.loads((REPO_ROOT / "output/real-validation-v6-review/translation-review.json").read_text(encoding="utf-8"))
    approved_text = next(unit["approved_text"] for unit in approved["units"] if unit["unit_id"] == "tu-0005")
    assert fixture.original_text == approved_text


def test_original_and_normalized_cases_are_separate_and_immutable():
    fixture = TTSBenchmarkFixture.load(FIXTURE_PATH)
    cases = fixture.cases()
    assert cases["original"] != cases["normalized"]
    assert "Hiragana" in cases["original"] and "hi-ra-ga-na" not in cases["original"]
    assert "hi-ra-ga-na" in cases["normalized"]
    assert fixture.pronunciation_terms["hiragana"] == "Hiragana"


def test_benchmark_captures_provider_and_wav_metadata(tmp_path):
    report = benchmark_provider(
        WavProvider,
        provider_name="fixture-provider",
        model_name="fixture-model",
        cases={"original": "xin chào"},
        output_dir=tmp_path,
    )
    case = report["cases"][0]
    assert report["status"] == "COMPLETED"
    assert report["initialization_status"] == "LOADED"
    assert report["model_size_bytes"] == 123
    assert case["status"] == "SUCCESS"
    assert case["sample_rate"] == 16000
    assert case["channels"] == 1 and case["pcm"]
    assert case["duration_seconds"] == pytest.approx(0.1)
    assert case["rtf"] is not None
    assert Path(case["output_wav"]).is_file()
    assert "not process-specific" in report["vram_measurement"]["method"]


def test_failed_provider_initialization_is_recorded_for_every_case(tmp_path):
    def broken_provider():
        raise RuntimeError("model files missing")

    report = benchmark_provider(
        broken_provider,
        provider_name="broken",
        model_name="missing-model",
        cases={"original": "xin chào", "normalized": "xin-cha-o"},
        output_dir=tmp_path,
    )
    assert report["status"] == "FAILED_GENERATION"
    assert report["initialization_status"] == "FAILED"
    assert [case["case"] for case in report["cases"]] == ["original", "normalized"]
    assert all(case["status"] == "FAILED_GENERATION" for case in report["cases"])
    assert all("model files missing" in case["error"] for case in report["cases"])


def test_wav_validation_reopens_pcm_and_rejects_invalid_file(tmp_path):
    path = tmp_path / "valid.wav"
    WavProvider().synthesize("test", path)
    assert validate_wav(path)["pcm"] is True
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wave")
    with pytest.raises((wave.Error, EOFError)):
        validate_wav(bad)


def test_report_writer_serializes_measurements(tmp_path):
    report = {"provider": "test", "cases": [], "status": "COMPLETED"}
    path = write_provider_report(tmp_path, report)
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_benchmark_additions_do_not_touch_production_provider_files():
    protected = [
        REPO_ROOT / "worker/src/dubflow_worker/tts/sherpa_onnx.py",
        REPO_ROOT / "worker/src/dubflow_worker/pipeline/synthesize.py",
        REPO_ROOT / "worker/src/dubflow_worker/pipeline/align_audio.py",
    ]
    assert all(path.is_file() for path in protected)
    assert all("benchmark_v1" not in path.read_text(encoding="utf-8") for path in protected)
