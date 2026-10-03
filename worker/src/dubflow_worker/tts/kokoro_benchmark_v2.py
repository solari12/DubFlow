"""Benchmark-only Kokoro Vietnamese adapter and report helpers."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

from dubflow_worker.tts.benchmark_v1 import (
    TTSBenchmarkFixture,
    _windows_peak_working_set_mib,
    validate_wav,
)


MODEL_REPO_ID = "contextboxai/Kokoro-Vietnamese"
MODEL_REVISION = "9f210d6"
SOURCE_REPO = "https://github.com/iamdinhthuan/Kokoro-Vietnamese"
SOURCE_COMMIT = "a249afe5555aec6c435165c2f61ec0f71284812f"
SAMPLE_RATE = 24000
SELECTED_VOICES = {
    "diem_trinh": {"label": "Diễm Trinh", "selection_note": "Candidate female voice; upstream provides a name, not verified gender metadata."},
    "thanh_dat": {"label": "Thành Đạt", "selection_note": "Candidate male voice; upstream provides a name, not verified gender metadata."},
    "mai_linh": {"label": "Mai Linh", "selection_note": "Additional distinct voicepack; any perceived gender/timbral difference awaits listening."},
}
CASE_NAMES = ("original", "normalized", "isolated-hiragana", "isolated-katakana", "isolated-kanji")


def normalize_device(device: str) -> str:
    value = device.strip().lower()
    if value not in {"cpu", "cuda"}:
        raise ValueError("Kokoro benchmark device must be 'cpu' or 'cuda'")
    return value


def validate_voice(voice: str, available_voices: list[str] | tuple[str, ...] | set[str] | None = None) -> str:
    choices = set(available_voices or SELECTED_VOICES)
    if voice not in choices:
        raise ValueError(f"Unknown Kokoro voice {voice!r}; available voices: {', '.join(sorted(choices))}")
    return voice


def classify_inference_error(device: str, error: str) -> str:
    text = error.lower()
    if any(token in text for token in ("out of memory", "cublas_status_alloc_failed", "cuda error 2")):
        return "CUDA_OOM" if device == "cuda" else "FAILED"
    return "CUDA_FAILED" if device == "cuda" else "FAILED"


class KokoroBenchmarkProvider:
    """Thin wrapper around the upstream PyTorch API; CUDA requests never fall back."""

    def __init__(self, *, device: str, voice: str, assets_dir: Path):
        self.device = normalize_device(device)
        self.voice = validate_voice(voice)
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE: torch.cuda.is_available() is false; refusing CPU fallback")

        from kokoro_vietnamese import KokoroVietnamese, list_voices

        validate_voice(self.voice, list_voices())
        root = Path(assets_dir)
        tts = KokoroVietnamese(
            device=self.device,
            voice=self.voice,
            model_path=root / "kokoro_vi.pth",
            voicepack_path=root / "voicepacks" / f"{self.voice}.pt",
            config_path=root / "config.json",
        )
        actual_device = str(getattr(tts, "device", "unknown")).lower()
        if actual_device != self.device:
            raise RuntimeError(
                f"Kokoro selected {actual_device!r} for requested {self.device!r}; refusing device fallback"
            )
        try:
            model_device = next(tts.model.parameters()).device.type
        except (AttributeError, StopIteration, TypeError):
            model_device = actual_device
        if model_device != self.device:
            raise RuntimeError(
                f"Kokoro model parameters are on {model_device!r}, requested {self.device!r}; refusing device fallback"
            )
        self.tts = tts

    def synthesize(self, text: str, output_path: Path | None = None) -> dict[str, Any]:
        import soundfile as sf

        audio, phonemes = self.tts.synthesize(text)
        result = {"phonemes": phonemes, "sample_rate": SAMPLE_RATE}
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            result["output_wav"] = str(output_path)
        return result


class SystemVramSampler:
    """Coarse whole-device sampling, reported separately from torch allocator data."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval = interval_seconds
        self.samples: list[int] = []
        self.errors: list[str] = []
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            self.samples.append(int(result.stdout.strip().splitlines()[0]))
        except Exception as exc:
            if not self.errors:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        while not self.stopped.is_set():
            self._sample()
            self.stopped.wait(self.interval)

    def start(self) -> None:
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def finish(self) -> dict[str, Any]:
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=4)
        self._sample()
        return {
            "method": "Whole-device nvidia-smi memory.used, sampled every 0.5s; coarse, not process-specific or exact peak.",
            "before_mib": self.samples[0] if self.samples else None,
            "peak_sampled_mib": max(self.samples) if self.samples else None,
            "after_mib": self.samples[-1] if self.samples else None,
            "sample_count": len(self.samples),
            "error": self.errors[0] if self.errors else None,
        }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi_info() -> dict[str, Any]:
    info: dict[str, Any] = {"name": None, "driver_version": None, "total_vram_mib": None}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        row = next(csv.reader([result.stdout.strip()]))
        info.update({"name": row[0].strip(), "driver_version": row[1].strip(), "total_vram_mib": int(row[2].strip())})
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def environment_metadata(*, before: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_runtime = torch.version.cuda
        cuda_gpu = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception as exc:
        cuda_available, cuda_runtime, cuda_gpu = None, None, None
        torch_error = f"{type(exc).__name__}: {exc}"
    else:
        torch_error = None
    try:
        import onnxruntime as ort

        ort_info = {"version": ort.__version__, "available_providers": ort.get_available_providers()}
    except Exception:
        ort_info = {"version": None, "available_providers": []}
    try:
        import kokoro_vietnamese  # noqa: F401

        package_version = _package_version("kokoro-vietnamese")
        direct_url = importlib.metadata.distribution("kokoro-vietnamese").read_text("direct_url.json")
        source_info = json.loads(direct_url) if direct_url else None
    except Exception:
        package_version, source_info = _package_version("kokoro-vietnamese"), None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "venv_prefix": sys.prefix,
        "before_install": before,
        "packages": {
            "kokoro-vietnamese": package_version,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "vig2p": _package_version("vig2p"),
            "huggingface-hub": _package_version("huggingface-hub"),
            "numpy": _package_version("numpy"),
            "soundfile": _package_version("soundfile"),
            "onnxruntime": _package_version("onnxruntime"),
            "onnxruntime-gpu": _package_version("onnxruntime-gpu"),
        },
        "source_install_metadata": source_info,
        "torch_cuda": {
            "cuda_is_available": cuda_available,
            "runtime_version": cuda_runtime,
            "device_name": cuda_gpu,
            "error": torch_error,
        },
        "gpu": _nvidia_smi_info(),
        "onnx_runtime": ort_info,
    }


def _torch_memory_snapshot(torch, device: str) -> dict[str, Any] | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "measurement": "Process-level PyTorch CUDA allocator statistics; includes model and temporary tensors.",
        "allocated_mib": round(torch.cuda.memory_allocated() / (1024 * 1024), 3),
        "max_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
        "reserved_mib": round(torch.cuda.memory_reserved() / (1024 * 1024), 3),
        "max_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 3),
    }


def _write_case(provider, text: str, path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    provider.synthesize(text, path)
    synthesis_seconds = time.perf_counter() - started
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("Kokoro returned without creating a nonempty WAV")
    audio = validate_wav(path)
    return {
        **audio,
        "synthesis_seconds": synthesis_seconds,
        "rtf": synthesis_seconds / audio["duration_seconds"],
        "file_size_bytes": path.stat().st_size,
        "output_wav": str(path),
    }


def _failed_case(name: str, text: str, path: Path | None, error: str, status: str = "FAILED") -> dict[str, Any]:
    return {
        "case": name,
        "input_text": text,
        "output_wav": str(path) if path else None,
        "file_size_bytes": None,
        "success": False,
        "status": status,
        "error": error,
    }


def benchmark_voice_device(
    *,
    fixture: TTSBenchmarkFixture,
    device: str,
    voice: str,
    output_dir: Path,
    assets_dir: Path,
    provider_factory: Callable[..., Any] = KokoroBenchmarkProvider,
) -> dict[str, Any]:
    device = normalize_device(device)
    voice = validate_voice(voice)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_text = fixture.cases()
    torch = None
    try:
        import torch as torch_module

        torch = torch_module
    except Exception:
        pass
    sampler = SystemVramSampler()
    sampler.start()
    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        error = "RuntimeError: CUDA_UNAVAILABLE: torch.cuda.is_available() is false; refusing CPU fallback"
        return {
            "device": device,
            "voice": voice,
            "status": "CUDA_FAILED",
            "initialization_status": "FAILED",
            "model_initialization_seconds": None,
            "initialization_error": error,
            "original_repeated_runs": {"warmup": {"success": False, "error": error}, "runs": []},
            "cases": [_failed_case(n, t, output_dir / f"{n}.wav", error, "CUDA_FAILED") for n, t in cases_text.items()],
            "torch_cuda_memory": None,
            "whole_device_vram_samples": sampler.finish(),
            "process_peak_cpu_working_set_mib": _windows_peak_working_set_mib(),
        }

    memory_before = None
    init_started = time.perf_counter()
    try:
        if device == "cuda" and torch is not None:
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        provider = provider_factory(device=device, voice=voice, assets_dir=assets_dir)
        init_seconds = time.perf_counter() - init_started
        init_status = "LOADED"
        init_error = None
    except Exception as exc:
        init_seconds = time.perf_counter() - init_started
        provider = None
        init_status = "FAILED"
        init_error = f"{type(exc).__name__}: {exc}"

    case_results: list[dict[str, Any]] = []
    repeated: dict[str, Any] = {"warmup": None, "runs": []}
    status = "SUCCESS"
    if provider is None:
        status = classify_inference_error(device, init_error or "Provider initialization failed")
        repeated["warmup"] = {"success": False, "error": init_error}
        repeated["runs"] = [
            {"run": i, "success": False, "status": "SKIPPED", "error": "Provider initialization failed"}
            for i in range(1, 4)
        ]
        for name, text in cases_text.items():
            case_results.append(_failed_case(name, text, output_dir / f"{name}.wav", init_error or "Initialization failed", status))
    else:
        warmup_started = time.perf_counter()
        try:
            provider.synthesize(cases_text["original"], None)
            repeated["warmup"] = {"success": True, "synthesis_seconds": time.perf_counter() - warmup_started, "wav_written": False}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            repeated["warmup"] = {"success": False, "error": error, "synthesis_seconds": time.perf_counter() - warmup_started}
            status = classify_inference_error(device, error)

        original_runs = []
        original_paths = [output_dir / "original.wav", output_dir / "repeated" / "original-run-2.wav", output_dir / "repeated" / "original-run-3.wav"]
        if repeated["warmup"]["success"]:
            for run_number, path in enumerate(original_paths, start=1):
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    result = _write_case(provider, cases_text["original"], path)
                    result.update({"run": run_number, "success": True, "status": "SUCCESS"})
                    original_runs.append(result)
                except Exception as exc:
                    if path.exists():
                        path.unlink()
                    error = f"{type(exc).__name__}: {exc}"
                    original_runs.append({"run": run_number, "success": False, "status": classify_inference_error(device, error), "error": error, "output_wav": str(path)})
                    status = classify_inference_error(device, error)
                    if status == "CUDA_OOM":
                        break
        else:
            original_runs = [{"run": i, "success": False, "status": "SKIPPED_AFTER_WARMUP_FAILURE", "error": repeated["warmup"]["error"]} for i in range(1, 4)]
        while len(original_runs) < 3:
            original_runs.append({"run": len(original_runs) + 1, "success": False, "status": "SKIPPED_AFTER_OOM", "error": "Previous measured run exhausted CUDA memory"})
        repeated["runs"] = original_runs
        good_original = [run for run in original_runs if run.get("success")]
        if good_original:
            durations = [run["duration_seconds"] for run in good_original]
            times = [run["synthesis_seconds"] for run in good_original]
            rtfs = [run["rtf"] for run in good_original]
            case_results.append({
                "case": "original",
                "input_text": cases_text["original"],
                "status": "SUCCESS" if len(good_original) == 3 else "PARTIAL_SUCCESS",
                "success": len(good_original) == 3,
                "output_wav": good_original[0]["output_wav"],
                "file_size_bytes": good_original[0]["file_size_bytes"],
                "duration_seconds": sum(durations) / len(durations),
                "synthesis_seconds": sum(times) / len(times),
                "rtf": sum(rtfs) / len(rtfs),
                "min_synthesis_seconds": min(times),
                "max_synthesis_seconds": max(times),
                "average_synthesis_seconds": sum(times) / len(times),
                "min_rtf": min(rtfs),
                "max_rtf": max(rtfs),
                "average_rtf": sum(rtfs) / len(rtfs),
                "measured_runs": good_original,
                "channels": good_original[0]["channels"],
                "sample_rate": good_original[0]["sample_rate"],
            })
        else:
            case_results.append(_failed_case("original", cases_text["original"], original_paths[0], "No successful measured original synthesis", status))

        for name, text in cases_text.items():
            if name == "original":
                continue
            path = output_dir / f"{name}.wav"
            try:
                case = {"case": name, "input_text": text, **_write_case(provider, text, path), "success": True, "status": "SUCCESS"}
            except Exception as exc:
                if path.exists():
                    path.unlink()
                error = f"{type(exc).__name__}: {exc}"
                case = _failed_case(name, text, path, error, classify_inference_error(device, error))
                if case["status"] == "CUDA_OOM":
                    status = "CUDA_OOM"
            case_results.append(case)

    try:
        cuda_memory = _torch_memory_snapshot(torch, device) if torch is not None else None
    except Exception as exc:
        cuda_memory = {"error": f"{type(exc).__name__}: {exc}", "measurement": "PyTorch CUDA allocator metrics could not be read."}
    if cuda_memory is not None:
        cuda_memory["allocated_before_mib"] = round((memory_before or 0) / (1024 * 1024), 3)
    del provider
    del torch
    if device == "cuda":
        gc.collect()
        try:
            torch_module.cuda.synchronize()
            cuda_memory["allocated_after_mib"] = round(torch_module.cuda.memory_allocated() / (1024 * 1024), 3) if cuda_memory is not None else None
            torch_module.cuda.empty_cache()
        except Exception:
            pass
    process_peak = _windows_peak_working_set_mib()
    system_vram = sampler.finish()
    if device == "cuda" and status not in {"CUDA_OOM", "CUDA_FAILED"}:
        if all(case.get("status") == "SUCCESS" for case in case_results):
            status = "CUDA_SUPPORTED"
        else:
            status = "CUDA_FAILED"
    elif device == "cpu" and status == "SUCCESS" and not all(case.get("status") == "SUCCESS" for case in case_results):
        status = "PARTIAL_FAILURE"

    return {
        "device": device,
        "voice": voice,
        "voice_label": SELECTED_VOICES[voice]["label"],
        "voice_selection_note": SELECTED_VOICES[voice]["selection_note"],
        "status": status,
        "initialization_status": init_status,
        "model_initialization_seconds": init_seconds,
        "initialization_error": init_error,
        "process_peak_cpu_working_set_mib": process_peak,
        "process_peak_cpu_memory_method": "Windows process lifetime PeakWorkingSetSize; it is process-level and cumulative across this benchmark, not attributed solely to this voice/device.",
        "torch_cuda_memory": cuda_memory,
        "whole_device_vram_samples": system_vram,
        "original_repeated_runs": repeated,
        "cases": case_results,
    }


def prepare_model_assets(cache_root: Path, voices: list[str] | tuple[str, ...] = tuple(SELECTED_VOICES)) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    patterns = ["kokoro_vi.pth", "config.json", *(f"voicepacks/{voice}.pt" for voice in voices)]
    path = Path(snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        allow_patterns=patterns,
        cache_dir=str(cache_root),
    ))
    return path, path.name


def fixture_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_quality_review(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Kokoro Vietnamese quality review", "",
        "Human listening only. No automatic quality score or winner is produced.", "",
        "Pay attention to naturalness, robotic artifacts, Vietnamese tones, punctuation and pauses, sentence rhythm, Hiragana/Katakana/Kanji pronunciation, mixed Vietnamese/English, and suitability for video dubbing.", "",
        "| Voice | Device | Case | Human Review | Notes |", "|---|---|---|---|---|",
    ]
    for result in results:
        for case in result.get("cases", []):
            lines.append(f"| {result.get('voice_label', result['voice'])} (`{result['voice']}`) | {result['device']} | {case['case']} | NOT_REVIEWED |  |")
    lines.append("")
    return "\n".join(lines)


def render_benchmark_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 1K.4.2 - Kokoro Vietnamese TTS benchmark", "",
        f"Overall status: `{report['status']}`. Human quality review: `NOT_REVIEWED`.", "",
        "## Model and environment", "",
        f"- Repository package: `kokoro-vietnamese` {report['model']['package_version']} ({report['model']['source_commit']}).",
        f"- Model: `{report['model']['repo_id']}` at revision `{report['model']['revision']}` (resolved `{report['model'].get('resolved_revision') or 'unavailable'}`).",
        f"- Vietnamese G2P: `vig2p` {report['environment']['packages'].get('vig2p') or 'unavailable'}.",
        f"- PyTorch: {report['environment']['packages'].get('torch') or 'unavailable'}, CUDA runtime {report['environment']['torch_cuda'].get('runtime_version') or 'unavailable'}; `torch.cuda.is_available()` = {report['environment']['torch_cuda'].get('cuda_is_available')}.",
        f"- GPU: {report['environment']['gpu'].get('name') or 'unavailable'}; driver {report['environment']['gpu'].get('driver_version') or 'unavailable'}; total VRAM {report['environment']['gpu'].get('total_vram_mib') or 'unavailable'} MiB.",
        f"- Transformers: {report['environment']['packages'].get('transformers') or 'unavailable'}; ONNX Runtime used: {report['environment']['packages'].get('onnxruntime') or 'no'}.",
        f"- Fixture: `{report['fixture']['path']}` (SHA-256 `{report['fixture']['sha256']}`); source fixture was used unchanged.", "",
        "## Voice selection", "",
        "Only three names from the upstream voice registry were tested: Diễm Trinh and Mai Linh as candidate female voices, and Thành Đạt as a candidate male voice. The upstream registry provides voice names but no gender/timbre metadata; perceived presentation and audible distinction remain for human review.", "",
        "## CPU and CUDA results", "",
        "| Device | Voice | Status | Init seconds | Original synthesis avg / min / max s | Original RTF avg / min / max | PyTorch max allocated / reserved MiB | Whole-device VRAM peak sampled MiB | Process peak working set MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        original = next((case for case in result.get("cases", []) if case.get("case") == "original"), {})
        memory = result.get("torch_cuda_memory") or {}
        vram = result.get("whole_device_vram_samples") or {}
        lines.append(
            f"| {result['device']} | {result.get('voice_label', result['voice'])} (`{result['voice']}`) | {result['status']} | {_fmt(result.get('model_initialization_seconds'))} | {_fmt(original.get('average_synthesis_seconds'))} / {_fmt(original.get('min_synthesis_seconds'))} / {_fmt(original.get('max_synthesis_seconds'))} | {_fmt(original.get('average_rtf'))} / {_fmt(original.get('min_rtf'))} / {_fmt(original.get('max_rtf'))} | {_fmt(memory.get('max_allocated_mib'))} / {_fmt(memory.get('max_reserved_mib'))} | {_fmt(vram.get('peak_sampled_mib'))} | {_fmt(result.get('process_peak_cpu_working_set_mib'))} |"
        )
        if result.get("initialization_error"):
            lines.append(f"|  | Initialization error | `{result['initialization_error']}` |  |  |  |  |  |  |  |")
    lines.extend(["", "## Per-case WAV results", "", "| Device | Voice | Case | Status | Synthesis seconds | Duration seconds | RTF | Sample rate | Channels | File bytes | WAV |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"])
    wav_count = 0
    for result in report["results"]:
        for case in result.get("cases", []):
            if case.get("file_size_bytes"):
                wav_count += len(case.get("measured_runs", [])) if case["case"] == "original" else 1
            lines.append(
                f"| {result['device']} | {result.get('voice_label', result['voice'])} | {case['case']} | {case['status']} | {_fmt(case.get('synthesis_seconds'))} | {_fmt(case.get('duration_seconds'))} | {_fmt(case.get('rtf'))} | {case.get('sample_rate', '—')} | {case.get('channels', '—')} | {case.get('file_size_bytes', '—')} | `{case.get('output_wav') or 'not generated'}` |"
            )
            if case.get("error"):
                lines.append(f"|  | Error | `{case['error']}` |  |  |  |  |  |  |  |  |")
    lines.extend([
        "", f"Successful WAV count: {report.get('successful_wav_count', wav_count)}.",
        "", "## Existing candidate comparison", "",
        f"- Piper VIVOS x_low CPU RTF: {report['existing_candidates']['piper_cpu_rtf']} (Phase 1K.4).",
        f"- KorvaTTS CPU RTF: {report['existing_candidates']['korva_cpu_rtf']}; generated duration: {report['existing_candidates']['korva_original_duration_seconds']} s.",
        f"- KorvaTTS CUDA: `{report['existing_candidates']['korva_cuda_status']}`; CPU-assigned ONNX nodes with CPU fallback disabled, not OOM.",
        "- Kokoro results are listed above; failed CUDA cases are not retried on CPU.", "",
        "Quality comparison requires human listening review. Performance results are measured locally and are not sufficient by themselves to determine the preferred production TTS.", "",
        "## Memory interpretation", "",
        "PyTorch allocated/reserved values are process-level allocator statistics for Kokoro and are separate from the whole-device nvidia-smi samples. Whole-device values include any other GPU use and are coarse sampled values, not process peaks. CPU peak working set is a process-lifetime Windows high-water mark and is not attributable to each voice configuration independently.", "",
        "## Production boundary and limitations", "",
        "- This is an offline controlled-text benchmark. No production provider/pipeline, alignment, translation, rendering, or full-video behavior was changed or run.",
        "- Three voices and five short cases do not establish broad voice quality or dubbing suitability.",
        "- Pronunciation-normalized input is a TTS-only experimental fixture and was not edited.",
        "- Human quality review remains `NOT_REVIEWED`; no automatic score or winner is selected.", "",
    ])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"
