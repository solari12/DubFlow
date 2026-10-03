from __future__ import annotations

import json
import sys
import types
import wave
from pathlib import Path

import pytest

from dubflow_worker.tts.benchmark_v1 import TTSBenchmarkFixture, validate_wav
from dubflow_worker.tts.kokoro_benchmark_v2 import (
    KokoroBenchmarkProvider,
    benchmark_voice_device,
    build_quality_review,
    classify_inference_error,
    normalize_device,
    render_benchmark_report,
    validate_voice,
)


def _fixture() -> TTSBenchmarkFixture:
    return TTSBenchmarkFixture(
        version="1.0",
        language="vi",
        original_text="Câu tiếng Việt gốc.",
        normalized_text="Câu tiếng Việt đã chuẩn hóa.",
        pronunciation_terms={"hiragana": "Hiragana", "katakana": "Katakana", "kanji": "Kanji"},
        normalized_pronunciations={"hiragana": "hi-ra-ga-na", "katakana": "ka-ta-ka-na", "kanji": "kan-ji"},
        normalization_note="unchanged test fixture",
    )


def test_kokoro_device_selection_is_explicit():
    assert normalize_device("CPU") == "cpu"
    assert normalize_device("cuda") == "cuda"
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        normalize_device("auto")


def test_voice_validation_uses_upstream_voice_registry():
    assert validate_voice("diem_trinh", ["diem_trinh", "thanh_dat"]) == "diem_trinh"
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        validate_voice("not_a_voice", ["diem_trinh"])


def test_kokoro_provider_initializes_requested_cpu_voice(monkeypatch, tmp_path):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_package = types.ModuleType("kokoro_vietnamese")
    captured = {}

    class FakeModel:
        def parameters(self):
            return iter([types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"))])

    class FakeTTS:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.device = kwargs["device"]
            self.model = FakeModel()

        def synthesize(self, text):
            return [0.0], "phones"

    fake_package.KokoroVietnamese = FakeTTS
    fake_package.list_voices = lambda: ["diem_trinh", "thanh_dat"]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "kokoro_vietnamese", fake_package)

    provider = KokoroBenchmarkProvider(device="cpu", voice="diem_trinh", assets_dir=tmp_path)
    assert provider.device == "cpu"
    assert captured["device"] == "cpu"
    assert captured["voice"] == "diem_trinh"


def test_cuda_request_never_constructs_cpu_fallback(monkeypatch, tmp_path):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_package = types.ModuleType("kokoro_vietnamese")
    fake_package.KokoroVietnamese = lambda **kwargs: pytest.fail("model must not initialize on CPU")
    fake_package.list_voices = lambda: ["diem_trinh"]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "kokoro_vietnamese", fake_package)

    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        KokoroBenchmarkProvider(device="cuda", voice="diem_trinh", assets_dir=tmp_path)


def test_cuda_request_constructs_model_with_cuda_and_checks_device(monkeypatch, tmp_path):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake_package = types.ModuleType("kokoro_vietnamese")
    captured = {}

    class FakeModel:
        def parameters(self):
            return iter([types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))])

    class FakeTTS:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.device = "cuda"
            self.model = FakeModel()

    fake_package.KokoroVietnamese = FakeTTS
    fake_package.list_voices = lambda: ["diem_trinh"]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "kokoro_vietnamese", fake_package)

    provider = KokoroBenchmarkProvider(device="cuda", voice="diem_trinh", assets_dir=tmp_path)
    assert provider.device == "cuda"
    assert captured["device"] == "cuda"


def test_cuda_failure_status_only_calls_real_oom_an_oom():
    assert classify_inference_error("cuda", "CUDA out of memory allocating tensor") == "CUDA_OOM"
    assert classify_inference_error("cuda", "CUDA kernel launch failed") == "CUDA_FAILED"
    assert classify_inference_error("cpu", "out of memory") == "FAILED"


def test_benchmark_validates_wavs_and_serializes_case_results(tmp_path, monkeypatch):
    class FakeSampler:
        def start(self):
            pass

        def finish(self):
            return {"before_mib": 10, "peak_sampled_mib": 25, "after_mib": 12, "sample_count": 3, "error": None}

    monkeypatch.setattr("dubflow_worker.tts.kokoro_benchmark_v2.SystemVramSampler", FakeSampler)
    monkeypatch.setattr("dubflow_worker.tts.kokoro_benchmark_v2._windows_peak_working_set_mib", lambda: 100.0)

    class FakeProvider:
        def __init__(self, **kwargs):
            self.device = kwargs["device"]

        def synthesize(self, text, output_path):
            if output_path is None:
                return {"phonemes": "warmup"}
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 1600)

    result = benchmark_voice_device(
        fixture=_fixture(),
        device="cpu",
        voice="diem_trinh",
        output_dir=tmp_path / "cpu" / "diem_trinh",
        assets_dir=tmp_path,
        provider_factory=FakeProvider,
    )
    assert result["status"] == "SUCCESS"
    assert result["original_repeated_runs"]["warmup"]["wav_written"] is False
    assert len(result["original_repeated_runs"]["runs"]) == 3
    original = next(case for case in result["cases"] if case["case"] == "original")
    assert original["average_synthesis_seconds"] >= 0
    assert original["file_size_bytes"] > 0
    for case in result["cases"]:
        if case["status"] == "SUCCESS":
            props = validate_wav(Path(case["output_wav"]))
            assert props["pcm"] is True
    assert len(list((tmp_path / "cpu" / "diem_trinh").rglob("*.wav"))) == 7
    encoded = json.dumps(result, ensure_ascii=False)
    assert json.loads(encoded)["voice"] == "diem_trinh"


def test_failure_reporting_quality_review_and_report_generation():
    failed = benchmark_voice_device(
        fixture=_fixture(),
        device="cpu",
        voice="diem_trinh",
        output_dir=Path("missing-model"),
        assets_dir=Path("missing-model"),
        provider_factory=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("weights missing")),
    )
    assert failed["initialization_status"] == "FAILED"
    assert all(case["status"] == "FAILED" for case in failed["cases"])
    quality = build_quality_review([failed])
    assert "NOT_REVIEWED" in quality
    assert "| Voice | Device | Case | Human Review | Notes |" in quality
    assert "weights missing" in json.dumps(failed)

    environment = {
        "packages": {"vig2p": "0.1.2", "torch": "2.11.0+cu130", "transformers": "4.57.6", "onnxruntime": None},
        "torch_cuda": {"runtime_version": "13.0", "cuda_is_available": True},
        "gpu": {"name": "RTX 3050 Laptop GPU", "driver_version": "610.62", "total_vram_mib": 4096},
    }
    report = {
        "status": "COMPLETED_WITH_FAILURES",
        "model": {"package_version": "0.1.0", "source_commit": "sha", "repo_id": "model", "revision": "rev", "resolved_revision": "resolved"},
        "environment": environment,
        "fixture": {"path": "fixture.json", "sha256": "hash"},
        "existing_candidates": {"piper_cpu_rtf": 0.034, "korva_cpu_rtf": 2.6806, "korva_original_duration_seconds": 13.56, "korva_cuda_status": "CUDA_FAILED"},
        "results": [failed],
        "successful_wav_count": 0,
    }
    markdown = render_benchmark_report(report)
    assert "Quality comparison requires human listening review" in markdown
    assert "CUDA_FAILED" in markdown
    assert "NOT_REVIEWED" in markdown

