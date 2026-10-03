"""Provider-neutral, benchmark-only TTS helpers. Production provider selection is untouched."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any, Protocol


@dataclass(frozen=True)
class TTSBenchmarkFixture:
    version: str
    language: str
    original_text: str
    normalized_text: str
    pronunciation_terms: dict[str, str]
    normalized_pronunciations: dict[str, str]
    normalization_note: str

    @classmethod
    def load(cls, path: Path) -> "TTSBenchmarkFixture":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fixture = cls(**data)
        if not fixture.original_text.strip() or not fixture.normalized_text.strip():
            raise ValueError("Benchmark texts must not be empty")
        if fixture.original_text == fixture.normalized_text:
            raise ValueError("Original and TTS-normalized inputs must remain distinct")
        return fixture

    def cases(self) -> dict[str, str]:
        cases = {
            "original": self.original_text,
            "normalized": self.normalized_text,
        }
        cases.update({f"isolated-{name}": text for name, text in self.pronunciation_terms.items()})
        return cases


class BenchmarkTTSProvider(Protocol):
    provider: str
    model: str
    language: str
    device: str
    model_size_bytes: int | None

    def synthesize(self, text: str, output_path: Path) -> None: ...


def validate_wav(path: Path) -> dict[str, Any]:
    """Validate readable, nonempty PCM WAV and return measured properties."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        compression = wav_file.getcomptype()
    if channels < 1 or sample_width < 1 or sample_rate < 1 or frame_count < 1:
        raise ValueError(f"WAV has invalid stream metadata: {path}")
    if compression != "NONE":
        raise ValueError(f"WAV must use uncompressed PCM: {path}")
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
        "duration_seconds": frame_count / sample_rate,
        "pcm": True,
    }


def benchmark_provider(
    provider_factory,
    *,
    provider_name: str,
    model_name: str,
    cases: dict[str, str],
    output_dir: Path,
    language: str = "vi",
    hardware: str = "CPU",
) -> dict[str, Any]:
    """Initialize once, synthesize each case separately, preserving per-case failures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "provider": provider_name,
        "model": model_name,
        "language": language,
        "device": hardware,
        "model_size_bytes": None,
        "model_size_measurement_method": None,
        "provider_configuration": {},
        "model_initialization_seconds": None,
        "initialization_status": "NOT_STARTED",
        "cases": [],
        "warnings": [],
    }
    gpu_sampler = _SystemGpuMemorySampler()
    gpu_sampler.start()
    try:
        init_start = time.perf_counter()
        provider = provider_factory()
        result["model_initialization_seconds"] = round(time.perf_counter() - init_start, 6)
        result["initialization_status"] = "LOADED"
        result["model_size_bytes"] = getattr(provider, "model_size_bytes", None)
        result["model_size_measurement_method"] = getattr(provider, "model_size_measurement_method", "Not reported by provider")
        result["provider_configuration"] = getattr(provider, "benchmark_configuration", {})
        result["device"] = getattr(provider, "device", hardware)
    except Exception as exc:
        result["initialization_status"] = "FAILED"
        result["initialization_error"] = f"{type(exc).__name__}: {exc}"
        result["warnings"].append("All test cases skipped because provider initialization failed.")
        provider = None

    for case_name, text in cases.items():
        wav_path = output_dir / f"{case_name}.wav"
        case: dict[str, Any] = {"case": case_name, "input_text": text, "output_wav": str(wav_path), "status": "FAILED_GENERATION"}
        if provider is None:
            case["error"] = result.get("initialization_error", "Provider initialization failed")
            result["cases"].append(case)
            continue
        synth_start = time.perf_counter()
        try:
            # Native providers can write diagnostics directly to the OS stderr
            # handle, so capture fd 2 rather than only redirecting sys.stderr.
            sys.stderr.flush()
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_capture:
                saved_stderr = os.dup(2)
                try:
                    os.dup2(stderr_capture.fileno(), 2)
                    provider.synthesize(text, wav_path)
                finally:
                    sys.stderr.flush()
                    os.dup2(saved_stderr, 2)
                    os.close(saved_stderr)
                stderr_capture.seek(0)
                captured_warnings = stderr_capture.read().strip()
            synth_seconds = time.perf_counter() - synth_start
            properties = validate_wav(wav_path)
            case.update(properties)
            case["synthesis_seconds"] = round(synth_seconds, 6)
            case["rtf"] = round(synth_seconds / properties["duration_seconds"], 6)
            case["status"] = "SUCCESS"
            if captured_warnings:
                case["warnings"] = [line.strip() for line in captured_warnings.splitlines() if line.strip()]
                result["warnings"].extend(case["warnings"])
        except Exception as exc:
            case["synthesis_seconds"] = round(time.perf_counter() - synth_start, 6)
            case["error"] = f"{type(exc).__name__}: {exc}"
            case["warnings"] = ["Partial/invalid WAV, if present, must not be used for listening."]
            if wav_path.exists():
                wav_path.unlink()
        result["cases"].append(case)

    result["total_runtime_seconds"] = round(time.perf_counter() - started, 6)
    successful = [case for case in result["cases"] if case["status"] == "SUCCESS"]
    result["status"] = "COMPLETED" if successful and len(successful) == len(result["cases"]) else (
        "PARTIAL_FAILURE" if successful else "FAILED_GENERATION"
    )
    result["peak_process_working_set_mib"] = _windows_peak_working_set_mib()
    result["vram_measurement"] = gpu_sampler.stop()
    result["vram_measurement"]["device_is_cpu"] = str(result["device"]).lower() == "cpu"
    if result["vram_measurement"]["device_is_cpu"]:
        result["warnings"].append("Provider ran on CPU; observed nvidia-smi values are system-wide background samples, not provider VRAM.")
    return result


def write_provider_report(output_dir: Path, report: dict[str, Any]) -> Path:
    report_path = Path(output_dir) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def _windows_peak_working_set_mib() -> float | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current = kernel.GetCurrentProcess
    get_current.restype = wintypes.HANDLE
    get_info = psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    if not get_info(get_current(), ctypes.byref(counters), counters.cb):
        return None
    return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)


class _SystemGpuMemorySampler:
    """Coarse system-wide nvidia-smi samples; never represented as process peak."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.values_mib: list[int] = []
        self.errors: list[str] = []
        self._stop = Event()
        self._thread: Thread | None = None

    def _sample(self) -> None:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
            )
            first = completed.stdout.strip().splitlines()[0]
            self.values_mib.append(int(first.strip()))
        except Exception as exc:
            if not self.errors:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._sample()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self.values_mib:
            self._sample()
        return {
            "method": "System-wide nvidia-smi memory.used sampled every 0.5s; coarse and not process-specific or exact peak.",
            "before_mib": self.values_mib[0] if self.values_mib else None,
            "peak_sampled_mib": max(self.values_mib) if self.values_mib else None,
            "after_mib": self.values_mib[-1] if self.values_mib else None,
            "sample_count": len(self.values_mib),
            "available": bool(self.values_mib),
            "error": self.errors[0] if self.errors else None,
        }
