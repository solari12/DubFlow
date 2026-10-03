from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from dubflow_worker.audio.extractor import AudioExtractionError, AudioExtractor
from dubflow_worker.diarization.base import DiarizationResult
from dubflow_worker.diarization.pyannote import PyannoteDiarizer
from dubflow_worker.config.settings import Settings


class NvidiaSmiSampler:
    """Sample whole-device VRAM. This is not process allocation or an exact peak."""

    def __init__(self) -> None:
        self.executable = shutil.which("nvidia-smi")
        self.maximum_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set() and self.executable:
            try:
                result = subprocess.run(
                    [self.executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=False,
                )
                if result.returncode == 0:
                    readings = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                    if readings:
                        self.maximum_mib = max(self.maximum_mib or 0, max(readings))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(0.25)

    def start(self) -> None:
        if self.executable:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the local pyannote diarization pipeline")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _gpu_info() -> dict[str, str | int | None]:
    info: dict[str, str | int | None] = {"name": None, "total_memory_mib": None}
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                name, memory = (part.strip() for part in result.stdout.splitlines()[0].split(",", 1))
                info.update(name=name, total_memory_mib=int(memory))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return info


def _torch_memory(device: str) -> dict[str, int | None]:
    result = {
        "process_allocated_mib": None,
        "process_reserved_mib": None,
        "process_peak_allocated_mib": None,
        "process_peak_reserved_mib": None,
    }
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            result["process_allocated_mib"] = round(torch.cuda.memory_allocated() / 1048576)
            result["process_reserved_mib"] = round(torch.cuda.memory_reserved() / 1048576)
            result["process_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1048576)
            result["process_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1048576)
    except (ImportError, RuntimeError):
        pass
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.min_speakers is not None and args.min_speakers < 1:
        _parser().error("--min-speakers must be positive")
    if args.max_speakers is not None and args.max_speakers < 1:
        _parser().error("--max-speakers must be positive")
    if args.min_speakers and args.max_speakers and args.min_speakers > args.max_speakers:
        _parser().error("--min-speakers cannot exceed --max-speakers")
    if not args.output.parent.is_dir():
        print(f"Output directory does not exist: {args.output.parent}", file=sys.stderr)
        return 1

    settings = Settings.from_env()
    extractor = AudioExtractor(settings.ffmpeg_path)
    report: dict = {
        "version": "1.0",
        "environment": {
            "python": sys.version.split()[0],
            "pyannote_audio": importlib.metadata.version("pyannote.audio")
            if importlib.util.find_spec("pyannote") else None,
            "device_requested": args.device,
            "gpu": _gpu_info(),
        },
        "success": False,
        "audio_duration_seconds": None,
        "model_load_seconds": None,
        "diarization_seconds": None,
        "processing_seconds": None,
        "total_runtime_seconds": None,
        "rtf": None,
        "sampled_whole_gpu_memory_mib": None,
        "process_allocated_mib": None,
        "process_reserved_mib": None,
        "process_peak_allocated_mib": None,
        "process_peak_reserved_mib": None,
        "speaker_count": None,
        "segment_count": None,
        "error": None,
        "result": None,
    }
    sampler = NvidiaSmiSampler()
    diarizer = None
    runtime_started: float | None = None
    try:
        with extractor.extract(args.input) as audio:
            report["audio_duration_seconds"] = round(audio.duration, 3)
            sampler.start()
            try:
                import torch
                if args.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except (ImportError, RuntimeError):
                pass
            runtime_started = time.perf_counter()
            load_started = time.perf_counter()
            try:
                diarizer = PyannoteDiarizer(
                    args.model, args.device,
                    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                )
            finally:
                report["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
            started = time.perf_counter()
            output = diarizer.diarize(audio.path, args.min_speakers, args.max_speakers)
            processing = time.perf_counter() - started
            report["diarization_seconds"] = round(processing, 3)
            report["processing_seconds"] = round(processing, 3)
            report["total_runtime_seconds"] = round(time.perf_counter() - runtime_started, 3)
            report["rtf"] = round(processing / audio.duration, 4) if audio.duration else None
            annotation = getattr(output, "speaker_diarization", output)
            tracks = (
                (turn.start, turn.end, speaker)
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            )
            result = DiarizationResult.from_tracks(
                tracks, filename=args.input.name, duration=audio.duration,
                engine=PyannoteDiarizer.name, model=args.model, device=args.device,
            )
            report["speaker_count"] = len(result.speakers)
            report["segment_count"] = len(result.segments)
            report["result"] = result.to_dict()
            report["success"] = True
            report.update(_torch_memory(args.device))
    except Exception as exc:
        if runtime_started is not None:
            report["total_runtime_seconds"] = round(time.perf_counter() - runtime_started, 3)
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Diarization benchmark failed: {report['error']}", file=sys.stderr)
    finally:
        sampler.stop()
        report["sampled_whole_gpu_memory_mib"] = sampler.maximum_mib
        if diarizer:
            diarizer.close()
    try:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"Could not write benchmark report: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
