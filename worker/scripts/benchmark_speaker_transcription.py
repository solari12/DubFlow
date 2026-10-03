from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.pipeline.speaker_transcription import merge_speaker_transcript


ROOT = Path(__file__).resolve().parents[2]
ASR_SCRIPT = ROOT / "worker" / "scripts" / "benchmark_asr.py"
DIARIZATION_SCRIPT = ROOT / "worker" / "scripts" / "benchmark_diarization.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run faster-whisper and pyannote sequentially, then assign speakers by overlap"
    )
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=ROOT / "input" / "phase1b1" / "two-voice-alternating.wav",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output" / "phase1b2" / "speaker-transcription-benchmark.json",
    )
    parser.add_argument(
        "--asr-python", type=Path,
        default=ROOT / "worker" / ".venv" / "Scripts" / "python.exe",
    )
    parser.add_argument(
        "--diarization-python", type=Path,
        default=ROOT / "worker" / ".venv-diarization" / "Scripts" / "python.exe",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_oom(*values: Any) -> bool:
    text = " ".join(str(value) for value in values if value is not None).lower()
    return any(marker in text for marker in ("out of memory", "cuda_error_out_of_memory", "cuda oom"))


def _run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "success": False,
        "asr_runtime_seconds": None,
        "diarization_runtime_seconds": None,
        "merge_runtime_seconds": None,
        "total_runtime_seconds": None,
        "asr_rtf": None,
        "diarization_rtf": None,
        "speaker_count": None,
        "transcript_segment_count": None,
        "assigned_speaker_segment_count": None,
        "unassigned_speaker_segment_count": None,
        "sampled_whole_gpu_memory_mib": None,
        "oom": False,
        "segments": [],
    }
    failures: list[str] = []
    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"Audio input does not exist: {args.input}")
        if not args.asr_python.is_file():
            raise FileNotFoundError(f"ASR Python environment not found: {args.asr_python}")
        if not args.diarization_python.is_file():
            raise FileNotFoundError(
                f"Diarization Python environment not found: {args.diarization_python}"
            )

        with tempfile.TemporaryDirectory(prefix="dubflow-phase1b2-", dir=output_path.parent) as work:
            work_dir = Path(work)
            asr_report_path = work_dir / "asr-report.json"
            transcripts_dir = work_dir / "transcripts"
            asr_env = os.environ.copy()
            source_root = str(ROOT / "worker" / "src")
            asr_env["PYTHONPATH"] = source_root + os.pathsep + asr_env.get("PYTHONPATH", "")

            asr_command = [
                str(args.asr_python), str(ASR_SCRIPT), str(args.input.resolve()),
                "--models", "base", "--device", "cuda", "--compute-type", "float16",
                "--transcripts-dir", str(transcripts_dir), "--output", str(asr_report_path),
            ]
            asr_process = _run(asr_command, env=asr_env)
            asr_report = _read_json(asr_report_path) if asr_report_path.is_file() else {}
            asr_result = (
                asr_report.get("cases", [{}])[0].get("results", [{}])[0]
                if asr_report.get("cases") else {}
            )
            report["asr_runtime_seconds"] = asr_result.get("processing_seconds")
            report["asr_rtf"] = asr_result.get("rtf")
            asr_memory = asr_result.get("sampled_whole_gpu_memory_mib")
            oom = _is_oom(asr_result.get("error"), asr_process.stdout, asr_process.stderr)
            if asr_process.returncode != 0 or not asr_result.get("success"):
                failures.append(
                    "ASR failed: " + (asr_result.get("error") or asr_process.stderr.strip() or "unknown error")
                )
            transcript_path = Path(asr_result["transcript_file"]) if asr_result.get("transcript_file") else None

            # ASR ran in a separate process. subprocess.run has returned, so its model and CUDA
            # context have been torn down before pyannote starts.
            if not failures and transcript_path and transcript_path.is_file():
                diar_report_path = work_dir / "diarization-report.json"
                diar_command = [
                    str(args.diarization_python), str(DIARIZATION_SCRIPT),
                    str(args.input.resolve()),
                    "--model", "pyannote/speaker-diarization-community-1",
                    "--device", "cuda", "--min-speakers", "2", "--max-speakers", "2",
                    "--output", str(diar_report_path),
                ]
                diar_process = _run(diar_command, env=os.environ.copy())
                diar_report = _read_json(diar_report_path) if diar_report_path.is_file() else {}
                report["diarization_runtime_seconds"] = diar_report.get("diarization_seconds")
                diar_memory = diar_report.get("sampled_whole_gpu_memory_mib")
                oom = oom or _is_oom(diar_report.get("error"), diar_process.stdout, diar_process.stderr)
                if diar_process.returncode != 0 or not diar_report.get("success"):
                    failures.append(
                        "Diarization failed: "
                        + (diar_report.get("error") or diar_process.stderr.strip() or "unknown error")
                    )
                if not failures:
                    transcript = _read_json(transcript_path)
                    diarization = diar_report.get("result") or {}
                    duration = float(transcript["source"]["duration"])
                    report["speaker_count"] = len(diarization.get("speakers", []))
                    report["transcript_segment_count"] = len(transcript.get("segments", []))
                    merge_started = time.perf_counter()
                    report["segments"] = merge_speaker_transcript(
                        transcript.get("segments", []),
                        diarization.get("segments", []),
                        threshold=0.20,
                    )
                    report["merge_runtime_seconds"] = round(time.perf_counter() - merge_started, 6)
                    report["assigned_speaker_segment_count"] = sum(
                        segment["speaker"] is not None for segment in report["segments"]
                    )
                    report["unassigned_speaker_segment_count"] = (
                        report["transcript_segment_count"] - report["assigned_speaker_segment_count"]
                    )
                    if report["asr_runtime_seconds"] is not None and duration:
                        report["asr_rtf"] = round(report["asr_runtime_seconds"] / duration, 4)
                    if report["diarization_runtime_seconds"] is not None and duration:
                        report["diarization_rtf"] = round(
                            report["diarization_runtime_seconds"] / duration, 4
                        )
                    memories = [value for value in (asr_memory, diar_memory) if value is not None]
                    report["sampled_whole_gpu_memory_mib"] = max(memories) if memories else None
                    report["success"] = True
            elif not failures:
                failures.append("ASR transcript JSON was not produced.")

            if asr_memory is not None and report["sampled_whole_gpu_memory_mib"] is None:
                report["sampled_whole_gpu_memory_mib"] = asr_memory
            if oom:
                report["oom"] = True
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        report["oom"] = _is_oom(exc)
    finally:
        report["total_runtime_seconds"] = round(time.perf_counter() - started, 3)

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print("\n".join(failures), file=sys.stderr)
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
