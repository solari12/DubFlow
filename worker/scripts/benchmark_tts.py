from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.pipeline.synthesize import synthesize_translated_transcript
from dubflow_worker.tts.sherpa_onnx import SherpaOnnxPiperEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local translated speech synthesis")
    parser.add_argument("input", type=Path, help="Phase 1C translated transcript JSON")
    parser.add_argument("--language", default="vi", help="Target language; default: vi")
    parser.add_argument("--output", type=Path, required=True, help="Directory for WAV and report")
    parser.add_argument("--model-dir", type=Path, help="Sherpa-ONNX Piper model directory")
    parser.add_argument("--num-threads", type=int, default=2)
    return parser


def _peak_working_set_mib() -> float | None:
    """Read the Windows process peak working set when the OS API is available."""
    if sys.platform != "win32":
        return None

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_memory_info = psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_memory_info.restype = wintypes.BOOL
    ok = get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    if not ok:
        return None
    return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "tts-benchmark.json"
    started = time.perf_counter()
    report: dict = {
        "version": "1.0",
        "success": False,
        "engine": "sherpa-onnx-piper-vivos-x_low",
        "model": "vi_VN-vivos-x_low",
        "model_size_bytes": None,
        "language": args.language,
        "device": "cpu",
        "model_initialization_seconds": None,
        "synthesis_runtime_seconds": 0.0,
        "total_runtime_seconds": None,
        "generated_audio_duration_seconds": 0.0,
        "rtf": None,
        "segment_count": 0,
        "successful_segments": 0,
        "failed_segments": 0,
        "sample_rate": None,
        "peak_process_working_set_mib": None,
        "vram_usage_mib": 0,
        "segments": [],
    }

    try:
        input_transcript = json.loads(args.input.read_text(encoding="utf-8"))
        input_segments = input_transcript.get("segments", [])
        report["segment_count"] = len(input_segments)
        model_dir = args.model_dir or (
            Path(__file__).resolve().parents[1]
            / ".model-cache"
            / "tts"
            / "vits-piper-vi_VN-vivos-x_low"
        )
        model_path = model_dir / "vi_VN-vivos-x_low.onnx"
        report["model_size_bytes"] = model_path.stat().st_size if model_path.is_file() else None
        init_started = time.perf_counter()
        engine = SherpaOnnxPiperEngine(model_dir, num_threads=args.num_threads)
        report["model_initialization_seconds"] = round(
            time.perf_counter() - init_started, 6
        )
        results = synthesize_translated_transcript(
            input_transcript,
            language=args.language,
            output_dir=args.output / "audio",
            engine=engine,
        )
        report["segments"] = [result.to_dict() for result in results]
        successes = [result.result for result in results if result.result is not None]
        report["successful_segments"] = len(successes)
        report["failed_segments"] = len(results) - len(successes)
        report["synthesis_runtime_seconds"] = round(
            sum(result.runtime for result in successes), 6
        )
        report["generated_audio_duration_seconds"] = round(
            sum(result.duration for result in successes), 6
        )
        sample_rates = sorted({result.sample_rate for result in successes})
        report["sample_rate"] = sample_rates[0] if len(sample_rates) == 1 else sample_rates or None
        if report["generated_audio_duration_seconds"] > 0:
            report["rtf"] = round(
                report["synthesis_runtime_seconds"]
                / report["generated_audio_duration_seconds"],
                6,
            )
        report["success"] = (
            report["failed_segments"] == 0 and report["successful_segments"] > 0
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        report["total_runtime_seconds"] = round(time.perf_counter() - started, 6)
        report["peak_process_working_set_mib"] = _peak_working_set_mib()
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
