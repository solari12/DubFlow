from __future__ import annotations

import sys
import types
import pytest

from dubflow_worker.tts.benchmark_providers_v1 import (
    _ort_device_for_request,
    create_cuda_only_session_options,
)
from dubflow_worker.tts.cuda_benchmark_v1 import (
    classify_cuda_failure,
    normalize_korva_device,
    render_cuda_benchmark_markdown,
)


def test_cuda_device_selection_is_explicit():
    assert normalize_korva_device("cuda") == "cuda"
    assert _ort_device_for_request("cuda") == "gpu"
    assert _ort_device_for_request("gpu") == "gpu"
    with pytest.raises(ValueError, match="requires device"):
        normalize_korva_device("cpu")


def test_cuda_request_does_not_silently_fall_back_to_cpu(monkeypatch):
    fake_korva = types.ModuleType("korvatts")
    fake_korva.__spec__ = __import__("importlib.util").util.spec_from_loader("korvatts", loader=None)
    fake_korva.TTS = lambda **kwargs: pytest.fail("TTS must not initialize on a CPU-only runtime")
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "korvatts", fake_korva)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    from dubflow_worker.tts.benchmark_providers_v1 import KorvaBenchmarkProvider

    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        KorvaBenchmarkProvider(device="cuda")


def test_cuda_only_sessions_disable_cpu_ep_fallback():
    class FakeOptions:
        def __init__(self):
            self.entries = []

        def add_session_config_entry(self, key, value):
            self.entries.append((key, value))

    class FakeOrt:
        SessionOptions = FakeOptions

    options = create_cuda_only_session_options(FakeOrt)
    assert options.entries == [("session.disable_cpu_ep_fallback", "1")]


def test_cuda_failure_classification_records_oom_and_unavailable():
    assert classify_cuda_failure("CUDA out of memory allocating tensor") == "CUDA_OOM"
    assert classify_cuda_failure("CUDAExecutionProvider is unavailable") == "CUDA_UNAVAILABLE"
    assert classify_cuda_failure("unsupported CUDA operator") == "CUDA_FAILED"


def test_cuda_report_serializes_gpu_metrics_cpu_baseline_and_repeated_runs():
    report = {
        "status": "CUDA_SUPPORTED",
        "initialization_status": "LOADED",
        "environment": {
            "gpu": {"name": "RTX 3050 Laptop GPU", "total_vram_mib": 4096, "driver_version": "610.62", "nvidia_smi_cuda_runtime": "13.3"},
            "python": "3.11.17",
            "torch": {"version": None, "cuda_runtime": None, "cuda_is_available": None},
            "onnxruntime": {"version": "1.30.0", "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]},
            "cuda_driver_runtime_assessment": "driver ok",
            "venv_prefix": "isolated-venv",
        },
        "initialization_error": None,
        "model_initialization_seconds": 1.25,
        "cpu_baseline": {"initialization_seconds": 2.9, "original": {"synthesis_seconds": 36.3, "duration_seconds": 13.56, "rtf": 2.68}, "peak_process_working_set_mib": 685.1},
        "original_repeated_runs": {
            "warmup": {"success": True},
            "runs": [{"run": 1, "status": "SUCCESS", "success": True, "synthesis_seconds": 2.0, "duration_seconds": 13.5, "rtf": 0.15}],
            "average_synthesis_seconds": 2.0,
            "min_synthesis_seconds": 2.0,
            "max_synthesis_seconds": 2.0,
            "average_rtf": 0.15,
        },
        "gpu_memory_measurements": {
            "process_level_sampled": {"peak_sampled_mib": None, "sample_count": 0, "error": "not exposed by WDDM"},
            "system_wide_sampled": {"before_mib": 200, "peak_sampled_mib": 850, "after_mib": 300},
        },
        "gpu_memory_allocated_peak_mib": None,
        "gpu_memory_reserved_peak_mib": None,
        "pytorch_memory_metrics_note": "Torch unavailable/not used by ONNX Runtime.",
        "peak_process_working_set_mib": 500.0,
        "cases": [{"case": "original", "status": "SUCCESS", "output_wav": "original.wav", "synthesis_seconds": 2.0, "duration_seconds": 13.5, "rtf": 0.15}],
    }
    markdown = render_cuda_benchmark_markdown(report)
    assert "CUDA_SUPPORTED" in markdown
    assert "PyTorch `gpu_memory_allocated_peak_mib` and `gpu_memory_reserved_peak_mib`: null" in markdown
    assert "200, peak 850, after 300 MiB" in markdown
    assert "warm-up success: true" in markdown.lower()
    assert "original.wav" in markdown


def test_cpu_benchmark_device_remains_explicit_cpu():
    assert _ort_device_for_request("cpu") == "cpu"
    assert _ort_device_for_request("CPU") == "cpu"
