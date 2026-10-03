from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

from dubflow_worker.tts.benchmark_v1 import _windows_peak_working_set_mib, validate_wav


CUDA_STATUSES = {"CUDA_SUPPORTED", "CUDA_FAILED", "CUDA_OOM", "CUDA_UNAVAILABLE"}


def normalize_korva_device(device: str) -> str:
    value = device.strip().lower()
    if value not in {"cuda", "gpu"}:
        raise ValueError("CUDA benchmark mode requires device='cuda' or 'gpu'")
    return "cuda"


def classify_cuda_failure(error: str) -> str:
    text = error.lower()
    if any(token in text for token in ("out of memory", "cublas_status_alloc_failed", "cuda error 2")):
        return "CUDA_OOM"
    if any(token in text for token in ("cudaexecutionprovider is unavailable", "cuda dll", "cudnn", "cannot load library", "failed to load library")):
        return "CUDA_UNAVAILABLE"
    return "CUDA_FAILED"


def _run_nvidia_smi(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=3, check=True)
        return completed.stdout.strip()
    except Exception:
        return None


def _gpu_info() -> dict[str, Any]:
    fields = _run_nvidia_smi(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    cuda_header = _run_nvidia_smi([])
    info: dict[str, Any] = {"name": None, "total_vram_mib": None, "driver_version": None, "nvidia_smi_cuda_runtime": None}
    if fields:
        try:
            row = next(csv.reader([fields]))
            info.update({"name": row[0].strip(), "total_vram_mib": int(row[1].strip()), "driver_version": row[2].strip()})
        except (StopIteration, IndexError, ValueError):
            pass
    if cuda_header:
        match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", cuda_header)
        if match:
            info["nvidia_smi_cuda_runtime"] = match.group(1)
    return info


def _environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("korvatts", "onnxruntime-gpu", "onnxruntime", "torch", "numpy", "soundfile", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import onnxruntime as ort
        ort_info = {"version": ort.__version__, "available_providers": ort.get_available_providers(), "device": ort.get_device()}
    except Exception as exc:
        ort_info = {"version": None, "available_providers": [], "device": None, "error": f"{type(exc).__name__}: {exc}"}
    torch_info: dict[str, Any] = {"installed": importlib.util.find_spec("torch") is not None, "version": packages["torch"], "cuda_runtime": None, "cuda_is_available": None}
    if torch_info["installed"]:
        try:
            import torch
            torch_info.update({"cuda_runtime": torch.version.cuda, "cuda_is_available": bool(torch.cuda.is_available())})
        except Exception as exc:
            torch_info["error"] = f"{type(exc).__name__}: {exc}"
    else:
        torch_info["note"] = "PyTorch is absent; KorvaTTS uses ONNX Runtime. Torch allocator metrics would not measure ORT CUDA allocations."
    gpu = _gpu_info()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "venv_prefix": sys.prefix,
        "packages": packages,
        "onnxruntime": ort_info,
        "torch": torch_info,
        "gpu": gpu,
        "cuda_driver_runtime_assessment": (
            "nvidia-smi reports the driver CUDA UMD version; this is not a compatibility guarantee for every bundled CUDA/cuDNN component. The installed CUDAExecutionProvider was discovered and DLL preload completed; strict CUDA-only Korva session initialization is the definitive model compatibility check."
            if gpu.get("nvidia_smi_cuda_runtime") else "Driver/CUDA runtime compatibility could not be determined from nvidia-smi."
        ),
    }


class _GpuSampler:
    """System VRAM plus best-effort nvidia-smi process samples, clearly separated."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval = interval_seconds
        self.pid = os.getpid()
        self.system_samples: list[int] = []
        self.process_samples: list[int] = []
        self.system_error: str | None = None
        self.process_error: str | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        system = _run_nvidia_smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if system:
            try:
                self.system_samples.append(int(system.splitlines()[0].strip()))
            except ValueError as exc:
                self.system_error = str(exc)
        else:
            self.system_error = "nvidia-smi system memory query unavailable"

        processes = _run_nvidia_smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        if processes is None:
            self.process_error = "nvidia-smi per-process query unavailable (WDDM may not expose compute process memory)"
            return
        try:
            for row in csv.reader(processes.splitlines()):
                if len(row) >= 2:
                    pid, used_memory = row[0].strip(), row[1].strip()
                    if pid.upper() == "[N/A]" or used_memory.upper() == "[N/A]":
                        self.process_error = "nvidia-smi returned [N/A] for process telemetry; this driver/WDDM mode does not expose per-process VRAM here"
                        continue
                    if int(pid) == self.pid:
                        self.process_samples.append(int(used_memory))
        except (ValueError, csv.Error) as exc:
            self.process_error = f"Could not parse per-process nvidia-smi data: {exc}"

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample_once()
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        self._sample_once()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def finish(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=4)
        self._sample_once()
        return {
            "system_wide_sampled": {
                "method": "nvidia-smi total memory.used sampled every 0.5 seconds; system-wide, coarse, not process peak.",
                "before_mib": self.system_samples[0] if self.system_samples else None,
                "peak_sampled_mib": max(self.system_samples) if self.system_samples else None,
                "after_mib": self.system_samples[-1] if self.system_samples else None,
                "sample_count": len(self.system_samples),
                "error": self.system_error,
            },
            "process_level_sampled": {
                "method": "nvidia-smi compute-apps pid match sampled every 0.5 seconds; process-specific samples, coarse and not exact peak.",
                "pid": self.pid,
                "peak_sampled_mib": max(self.process_samples) if self.process_samples else None,
                "sample_count": len(self.process_samples),
                "error": self.process_error if not self.process_samples else None,
            },
        }


def _synthesize_one(provider, text: str, path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    provider.synthesize(text, path)
    runtime = time.perf_counter() - started
    metadata = validate_wav(path)
    return {**metadata, "synthesis_seconds": runtime, "rtf": runtime / metadata["duration_seconds"], "output_wav": str(path)}


def _failed_result(name: str, text: str, path: Path, error: str, status: str = "FAILED") -> dict[str, Any]:
    return {"case": name, "input_text": text, "output_wav": str(path), "status": status, "error": error}


def run_korva_cuda_benchmark(fixture, output_dir: Path, cpu_report_path: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repeated_dir = output_dir / "repeated"
    repeated_dir.mkdir(parents=True, exist_ok=True)
    env = _environment_snapshot()
    report: dict[str, Any] = {
        "version": "1.0",
        "phase": "1K.4.1",
        "provider": "korvatts-cuda",
        "model": "dogenthq/KorvaTTS",
        "device": "cuda",
        "gpu": env["gpu"],
        "status": "CUDA_UNAVAILABLE",
        "initialization_status": "NOT_STARTED",
        "environment": env,
        "cases": [],
        "original_repeated_runs": {"warmup": None, "runs": [], "average_synthesis_seconds": None, "min_synthesis_seconds": None, "max_synthesis_seconds": None, "average_rtf": None},
        "gpu_memory_allocated_peak_mib": None,
        "gpu_memory_reserved_peak_mib": None,
        "pytorch_memory_metrics_note": "Null: KorvaTTS is ONNX Runtime-based and PyTorch allocator values would not represent its ORT allocations.",
        "peak_process_working_set_mib": None,
        "gpu_memory_measurements": None,
        "warnings": [],
    }
    try:
        cpu_report = json.loads(Path(cpu_report_path).read_text(encoding="utf-8"))
        original_cpu = next(case for case in cpu_report["cases"] if case["case"] == "original")
        normalized_cpu = next(case for case in cpu_report["cases"] if case["case"] == "normalized")
        report["cpu_baseline"] = {
            "report_path": str(cpu_report_path),
            "initialization_seconds": cpu_report.get("model_initialization_seconds"),
            "model_size_bytes": cpu_report.get("model_size_bytes"),
            "original": {key: original_cpu.get(key) for key in ("duration_seconds", "synthesis_seconds", "rtf")},
            "normalized": {key: normalized_cpu.get(key) for key in ("duration_seconds", "synthesis_seconds", "rtf")},
            "peak_process_working_set_mib": cpu_report.get("peak_process_working_set_mib"),
        }
    except Exception as exc:
        report["warnings"].append(f"CPU baseline report could not be loaded: {type(exc).__name__}: {exc}")
        report["cpu_baseline"] = None

    sampler = _GpuSampler()
    sampler.start()
    provider = None
    try:
        from dubflow_worker.tts.benchmark_providers_v1 import KorvaBenchmarkProvider
        init_started = time.perf_counter()
        provider = KorvaBenchmarkProvider(device="cuda")
        report["model_initialization_seconds"] = time.perf_counter() - init_started
        report["initialization_status"] = "LOADED"
        report["model_size_bytes"] = provider.model_size_bytes
        report["model_size_measurement_method"] = provider.model_size_measurement_method
        report["provider_configuration"] = provider.benchmark_configuration
        report["execution_providers"] = provider.execution_providers
        report["device"] = provider.device
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report["initialization_status"] = "FAILED"
        report["initialization_error"] = error
        report["status"] = classify_cuda_failure(error)

    cases_text = fixture.cases()
    if provider is None:
        for name, text in cases_text.items():
            report["cases"].append(_failed_result(name, text, output_dir / f"{name}.wav", report["initialization_error"]))
        report["original_repeated_runs"]["warmup"] = {"success": False, "error": report["initialization_error"]}
        report["original_repeated_runs"]["runs"] = [
            {"run": index, "success": False, "status": "SKIPPED", "error": "CUDA initialization failed"}
            for index in range(1, 4)
        ]
    else:
        report["status"] = "CUDA_SUPPORTED"
        warmup_path = repeated_dir / "warmup.wav"
        try:
            warmup = _synthesize_one(provider, fixture.original_text, warmup_path)
            report["original_repeated_runs"]["warmup"] = {"success": True, **warmup}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            report["original_repeated_runs"]["warmup"] = {"success": False, "error": error}
            report["status"] = classify_cuda_failure(error)

        original_runs = []
        original_case = None
        if report["status"] == "CUDA_SUPPORTED":
            for index in range(1, 4):
                path = output_dir / "original.wav" if index == 1 else repeated_dir / f"original-run-{index}.wav"
                try:
                    result = _synthesize_one(provider, fixture.original_text, path)
                    result.update({"run": index, "success": True, "status": "SUCCESS"})
                    original_runs.append(result)
                    if index == 1:
                        original_case = {"case": "original", "input_text": fixture.original_text, **result}
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    original_runs.append({"run": index, "success": False, "status": classify_cuda_failure(error), "error": error, "output_wav": str(path)})
                    report["status"] = classify_cuda_failure(error)
                    break
            for skipped_index in range(len(original_runs) + 1, 4):
                original_runs.append({"run": skipped_index, "success": False, "status": "SKIPPED_AFTER_FAILURE", "error": "Previous measured CUDA synthesis failed"})
        else:
            original_runs = [{"run": index, "success": False, "status": "SKIPPED_AFTER_WARMUP_FAILURE", "error": "CUDA warm-up failed"} for index in range(1, 4)]
        report["original_repeated_runs"]["runs"] = original_runs
        successful_original_runs = [run for run in original_runs if run.get("success")]
        if successful_original_runs:
            runtimes = [run["synthesis_seconds"] for run in successful_original_runs]
            report["original_repeated_runs"].update({
                "average_synthesis_seconds": sum(runtimes) / len(runtimes),
                "min_synthesis_seconds": min(runtimes),
                "max_synthesis_seconds": max(runtimes),
                "average_rtf": sum(run["rtf"] for run in successful_original_runs) / len(successful_original_runs),
            })
        report["cases"].append(original_case or _failed_result("original", fixture.original_text, output_dir / "original.wav", "CUDA original synthesis did not succeed"))

        remaining = [(name, text) for name, text in cases_text.items() if name != "original"]
        for name, text in remaining:
            path = output_dir / f"{name}.wav"
            if report["status"] != "CUDA_SUPPORTED":
                report["cases"].append(_failed_result(name, text, path, "Earlier CUDA synthesis failed; CPU fallback is disabled", "SKIPPED_AFTER_FAILURE"))
                continue
            try:
                result = _synthesize_one(provider, text, path)
                result.update({"case": name, "input_text": text, "status": "SUCCESS"})
                if name in {"normalized", "isolated-hiragana", "isolated-katakana", "isolated-kanji"}:
                    cpu_case = next((case for case in cpu_report["cases"] if case["case"] == name), None) if report.get("cpu_baseline") else None
                    result["duration_delta_seconds_vs_cpu"] = (
                        result["duration_seconds"] - cpu_case["duration_seconds"]
                        if cpu_case and cpu_case.get("duration_seconds") is not None else None
                    )
                report["cases"].append(result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                report["cases"].append(_failed_result(name, text, path, error, classify_cuda_failure(error)))
                report["status"] = classify_cuda_failure(error)

    report["peak_process_working_set_mib"] = _windows_peak_working_set_mib()
    report["gpu_memory_measurements"] = sampler.finish()
    if report["status"] == "CUDA_SUPPORTED" and not all(case.get("status") == "SUCCESS" for case in report["cases"]):
        report["status"] = "CUDA_FAILED"
    if report["status"] not in CUDA_STATUSES:
        report["status"] = "CUDA_FAILED"
    return report


def render_cuda_benchmark_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    gpu = env["gpu"]
    cuda = report.get("gpu_memory_measurements") or {}
    process = cuda.get("process_level_sampled", {})
    system = cuda.get("system_wide_sampled", {})
    cpu = report.get("cpu_baseline") or {}
    cpu_original = cpu.get("original") or {}
    original_runs = report["original_repeated_runs"]
    lines = [
        "# Phase 1K.4.1 — KorvaTTS CUDA Benchmark", "", "## Hardware", "",
        f"- GPU: {gpu.get('name') or 'Unavailable'}",
        f"- VRAM: {gpu.get('total_vram_mib') if gpu.get('total_vram_mib') is not None else 'Unknown'} MiB",
        f"- Driver: {gpu.get('driver_version') or 'Unknown'}; nvidia-smi CUDA UMD: {gpu.get('nvidia_smi_cuda_runtime') or 'Unknown'}",
        f"- Python: {env['python']}",
        f"- PyTorch: {env['torch'].get('version') or 'not installed'}; torch CUDA runtime: {env['torch'].get('cuda_runtime') or 'not available'}; torch.cuda.is_available(): {env['torch'].get('cuda_is_available')}",
        f"- ONNX Runtime: {env['onnxruntime'].get('version')}; available providers: {', '.join(env['onnxruntime'].get('available_providers', []))}",
        f"- Driver/runtime note: {env['cuda_driver_runtime_assessment']}",
        "", "## Environment", "",
        f"- venv: `{env['venv_prefix']}`",
        "- Model: `dogenthq/KorvaTTS`",
        f"- CUDA benchmark status: `{report['status']}`",
        f"- Initialization: `{report.get('initialization_status')}`" + (f" ({report.get('initialization_error')})" if report.get("initialization_error") else ""),
        "- CUDA session policy: CUDAExecutionProvider only; ONNX Runtime CPU EP fallback disabled.",
        "", "## CPU baseline", "",
        f"- Initialization: {cpu.get('initialization_seconds', 'unavailable')} s; original {cpu_original.get('synthesis_seconds', 'unavailable')} s, duration {cpu_original.get('duration_seconds', 'unavailable')} s, RTF {cpu_original.get('rtf', 'unavailable')}; peak working set {cpu.get('peak_process_working_set_mib', 'unavailable')} MiB.",
        "", "## CUDA result", "",
        "| Metric | CPU | CUDA |", "|---|---:|---:|",
        f"| Initialization | {cpu.get('initialization_seconds', 'unavailable')} s | {_fmt(report.get('model_initialization_seconds'))} s |",
        f"| Original synthesis | {cpu_original.get('synthesis_seconds', 'unavailable')} s | {_fmt(original_runs.get('average_synthesis_seconds'))} s average of successful measured runs |",
        f"| Original duration | {cpu_original.get('duration_seconds', 'unavailable')} s | {_fmt(_mean_duration(original_runs.get('runs', [])))} s average |",
        f"| Original RTF | {cpu_original.get('rtf', 'unavailable')} | {_fmt(original_runs.get('average_rtf'))} average |",
        f"| Peak process GPU memory | N/A | {process.get('peak_sampled_mib') if process.get('peak_sampled_mib') is not None else 'Unavailable'} MiB sampled by nvidia-smi |",
        f"| Peak CPU working set | {cpu.get('peak_process_working_set_mib', 'unavailable')} MiB | {report.get('peak_process_working_set_mib', 'unavailable')} MiB |",
        "", "PyTorch `gpu_memory_allocated_peak_mib` and `gpu_memory_reserved_peak_mib`: null. " + report["pytorch_memory_metrics_note"],
        f"System-wide VRAM samples (not process peak): before {system.get('before_mib')}, peak {system.get('peak_sampled_mib')}, after {system.get('after_mib')} MiB. Process-level nvidia-smi samples: {process.get('sample_count', 0)}; {process.get('error') or 'sampled value is coarse, not exact peak'}.",
        "", "## Repeated CUDA benchmark", "",
        f"- Warm-up success: {bool(original_runs.get('warmup', {}).get('success'))}",
        "| Run | Result | Synthesis seconds | Duration seconds | RTF |", "|---:|---|---:|---:|---:|",
    ]
    for run in original_runs.get("runs", []):
        lines.append(f"| {run.get('run')} | {run.get('status')} | {_fmt(run.get('synthesis_seconds'))} | {_fmt(run.get('duration_seconds'))} | {_fmt(run.get('rtf'))} |")
    lines.extend(["", "## Case results", "", "| Case | Success | Runtime seconds | Duration seconds | RTF | Output path |", "|---|---|---:|---:|---:|---|"])
    for case in report["cases"]:
        lines.append(f"| {case['case']} | {case['status']} | {_fmt(case.get('synthesis_seconds'))} | {_fmt(case.get('duration_seconds'))} | {_fmt(case.get('rtf'))} | `{case.get('output_wav')}` |")
        if case.get("error"):
            lines.append(f"|  | Error |  |  |  | `{case['error']}` |")
    lines.extend([
        "", "## CUDA compatibility", "", f"`{report['status']}`", "",
        "## Human review", "", "Current CUDA WAVs: `NOT_REVIEWED`. Previously supplied manual listening feedback: KorvaTTS sounded significantly better than Piper; this is qualitative user feedback, not a numeric score or automatic selection.",
        "", "## Limitations", "",
        "- GPU has 4 GB VRAM; system-wide sampled values are not process-level peak.",
        "- ONNX Runtime exposes no PyTorch allocator statistics for these sessions; those fields are null. Process memory uses best-effort nvidia-smi PID samples where WDDM exposes them.",
        "- CUDA was tested for this controlled text only. No full-video, alignment, or render test was run.",
        "- Piper remains the production default; no production TTS behavior changed.", "",
    ])
    return "\n".join(lines)


def _mean_duration(runs: list[dict[str, Any]]) -> float | None:
    durations = [run["duration_seconds"] for run in runs if run.get("success") and run.get("duration_seconds") is not None]
    return sum(durations) / len(durations) if durations else None


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"
