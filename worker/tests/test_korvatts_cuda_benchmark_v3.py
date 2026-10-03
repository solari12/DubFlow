from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from dubflow_worker.tts.benchmark_v1 import TTSBenchmarkFixture
from dubflow_worker.tts.korvatts_cuda_benchmark_v3 import (
    CPU_PROVIDER,
    PROVIDER,
    _finish_profile,
    build_quality_review,
    calculate_rtf,
    comparison_data,
    render_report,
    synthesize_case,
    verify_session_providers,
)


REPO = Path(__file__).resolve().parents[2]


class FakeSession:
    def __init__(self, providers):
        self.providers = providers

    def get_providers(self):
        return self.providers


def test_existing_fixture_loads_without_rewriting_and_has_five_shared_cases():
    fixture_path = REPO / "output/tts-benchmark-v1/fixture.json"
    fixture = TTSBenchmarkFixture.load(fixture_path)
    assert list(fixture.cases()) == ["original", "normalized", "isolated-hiragana", "isolated-katakana", "isolated-kanji"]
    assert fixture.cases()["original"]


def test_provider_verification_requires_cuda_as_first_provider_for_every_session():
    sessions = {name: FakeSession([PROVIDER, CPU_PROVIDER]) for name in ("duration_predictor", "text_encoder")}
    result = verify_session_providers(sessions)
    assert result["cuda_primary_for_every_session"] is True
    assert result["cpu_ep_registered_as_fallback"] is True
    with pytest.raises(RuntimeError, match="CUDA session verification failed"):
        verify_session_providers({"duration_predictor": FakeSession([CPU_PROVIDER])})


def test_rtf_is_calculated_from_synthesis_wall_time_and_audio_duration():
    assert calculate_rtf(3.0, 6.0) == 0.5
    with pytest.raises(ValueError, match="positive"):
        calculate_rtf(1.0, 0.0)


def test_synthesis_records_total_wall_and_validated_wav_metadata(tmp_path):
    class FakeTTS:
        def synthesize(self, text, **kwargs):
            assert kwargs == {"voice": "gia_bao", "lang": "vi", "total_steps": 32, "speed": 1.05}
            return b"audio", 0.5

        def save_audio(self, wav_data, path):
            assert wav_data == b"audio"
            with wave.open(path, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(b"\0\0" * 12000)

    result = synthesize_case(FakeTTS(), "test", tmp_path / "case.wav")
    assert result["wav_validation"] == "PASS"
    assert result["sample_rate"] == 24000
    assert result["frame_count"] == 12000
    assert result["duration_seconds"] == 0.5
    assert result["file_size_bytes"] > 0
    assert result["rtf"] >= 0
    assert result["total_wall_seconds"] >= result["synthesis_seconds"]


def test_profile_inspection_detects_cpu_provider_fallback(tmp_path):
    class ProfileSession:
        def __init__(self, name, rows):
            self.name, self.rows = name, rows

        def end_profiling(self):
            path = tmp_path / f"{self.name}.json"
            path.write_text(json.dumps(self.rows), encoding="utf-8")
            return str(path)

    rows = [
        {"cat": "Node", "args": {"provider": PROVIDER}},
        {"cat": "Node", "args": {"provider": CPU_PROVIDER}},
        {"cat": "Session", "args": {}},
    ]
    result = _finish_profile({"vocoder": ProfileSession("profile", rows)}, tmp_path)
    assert result["status"] == "CPU_EP_FALLBACK_DETECTED"
    assert result["execution_node_event_counts"][PROVIDER] == 1
    assert result["execution_node_event_counts"][CPU_PROVIDER] == 1


def test_profile_inspection_fails_if_cuda_did_not_execute_any_node(tmp_path):
    class ProfileSession:
        def end_profiling(self):
            path = tmp_path / "cpu-only.json"
            path.write_text(json.dumps([{"cat": "Node", "args": {"provider": CPU_PROVIDER}}]), encoding="utf-8")
            return str(path)

    with pytest.raises(RuntimeError, match="did not record any nodes"):
        _finish_profile({"vocoder": ProfileSession()}, tmp_path)


def _report():
    return {
        "status": "CUDA_SUPPORTED",
        "environment": {"gpu": {"name": "RTX 3050", "total_vram_mib": 4096, "driver_version": "test"}, "python": "3.11", "onnxruntime": {"version": "1.30"}, "packages": {"korvatts": "0.1.3", "onnxruntime-gpu": "1.30"}},
        "cuda_execution_verification": {"status": "PASS", "execution_node_event_counts": {PROVIDER: 100, CPU_PROVIDER: 0}},
        "settings": {"voice": "gia_bao", "steps": 32, "speed": 1.05},
        "fixture": {"path": "fixture.json", "sha256": "sha"},
        "original_repeated_runs": {"warmup": {"synthesis_seconds": 0.5}, "runs": [], "average_synthesis_seconds": 0.6, "min_synthesis_seconds": 0.5, "max_synthesis_seconds": 0.7, "average_total_wall_seconds": 0.61, "average_rtf": 0.5, "min_rtf": 0.4, "max_rtf": 0.6},
        "gpu_memory_measurements": {"system_wide_sampled": {"peak_sampled_mib": 500, "method": "coarse system-wide"}, "process_level_sampled": {"peak_sampled_mib": None, "method": "process-level VRAM unavailable", "error": "WDDM"}},
        "peak_process_working_set_mib": 600,
        "cases": [{"case": "original", "status": "SUCCESS", "wav_validation": "PASS", "output_wav": "original.wav"}],
        "success_count": 1, "failure_count": 0, "cuda_initialization_failure_count": 0, "cpu_device_fallback_occurrences": 0, "wav_validation_failure_count": 0,
        "comparison": {"rows": []},
    }


def test_report_generation_covers_cuda_metrics_wav_review_and_no_winner():
    markdown = render_report(_report())
    assert "CUDAExecutionProvider" in markdown
    assert "coarse system-wide" in markdown
    assert "WDDM" in markdown
    assert "WAV validation: PASS" in markdown
    assert "No ranking or winner selection" in markdown
    sheet = build_quality_review(_report()["cases"])
    assert "NOT_REVIEWED" in sheet
    assert "GOOD" not in sheet and "WINNER" not in sheet


def test_comparison_uses_cuda_three_run_means_and_marks_provider_failure(tmp_path):
    previous = tmp_path / "output/tts-benchmark-v1"
    previous.mkdir(parents=True)
    (previous / "benchmark-results.json").write_text(json.dumps([]), encoding="utf-8")
    report = _report()
    report["status"] = "CUDA_FAILED"
    report["settings"]["voice"] = "gia_bao"
    report["original_repeated_runs"]["average_synthesis_seconds"] = 1.5
    report["original_repeated_runs"]["average_rtf"] = 0.25
    report["original_repeated_runs"]["runs"] = [
        {"success": True, "duration_seconds": 6.0},
        {"success": True, "duration_seconds": 7.0},
        {"success": True, "duration_seconds": 5.0},
    ]
    result = comparison_data(tmp_path, report)
    row = next(r for r in result["rows"] if r["provider"] == "KorvaTTS" and r["device"] == "CUDA")
    assert row["status"] == "CUDA_FAILED"
    assert row["wav_generation_status"] == "SUCCESS"
    assert row["wall_time_seconds"] == 1.5
    assert row["audio_duration_seconds"] == 6.0
    assert row["rtf"] == 0.25
