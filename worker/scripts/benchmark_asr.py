from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dubflow_worker.asr.base import ASRError
from dubflow_worker.asr.factory import create_asr_engine
from dubflow_worker.audio.extractor import AudioExtractionError, AudioExtractor
from dubflow_worker.config.settings import Settings


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    path: Path
    language: str | None


class NvidiaSmiSampler:
    """Sample whole-device VRAM usage; this is not per-process attribution."""

    def __init__(self) -> None:
        self.executable = shutil.which("nvidia-smi")
        self.maximum_mb: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        if not self.executable:
            return
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [self.executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if result.returncode == 0:
                    values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
                    if values:
                        sampled = max(values)
                        self.maximum_mb = max(self.maximum_mb or sampled, sampled)
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


def _normalize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _normalize_characters(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return "".join(char for char in normalized if not char.isspace() and char.isalnum())


def _edit_distance(left: str | list[str], right: str | list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _quality_metrics(reference: str, hypothesis: str) -> dict[str, int | float | None]:
    reference_words = _normalize_words(reference)
    hypothesis_words = _normalize_words(hypothesis)
    reference_chars = _normalize_characters(reference)
    hypothesis_chars = _normalize_characters(hypothesis)
    word_errors = _edit_distance(reference_words, hypothesis_words)
    char_errors = _edit_distance(reference_chars, hypothesis_chars)
    return {
        "reference_word_count": len(reference_words),
        "hypothesis_word_count": len(hypothesis_words),
        "word_errors": word_errors,
        "wer": word_errors / len(reference_words) if reference_words else None,
        "reference_character_count": len(reference_chars),
        "hypothesis_character_count": len(hypothesis_chars),
        "character_errors": char_errors,
        "cer": char_errors / len(reference_chars) if reference_chars else None,
    }


def _timestamp_quality(transcript, duration: float) -> dict[str, int | float | bool | None]:
    segments = transcript.segments
    invalid_ranges = sum(
        segment.start < 0 or segment.end <= segment.start or segment.end > duration + 0.05
        for segment in segments
    )
    out_of_order = 0
    overlaps = 0
    gaps: list[float] = []
    for previous, current in zip(segments, segments[1:]):
        if current.start < previous.start:
            out_of_order += 1
        if current.start < previous.end - 0.05:
            overlaps += 1
        elif current.start > previous.end:
            gaps.append(current.start - previous.end)

    invalid_word_ranges = 0
    non_monotonic_words = 0
    word_timestamp_count = 0
    for segment in segments:
        previous_word_start: float | None = None
        for word in segment.words:
            word_timestamp_count += 1
            if word.start < segment.start - 0.05 or word.end <= word.start or word.end > segment.end + 0.05:
                invalid_word_ranges += 1
            if previous_word_start is not None and word.start < previous_word_start:
                non_monotonic_words += 1
            previous_word_start = word.start

    return {
        "segment_ranges_valid": invalid_ranges == 0,
        "invalid_segment_ranges": invalid_ranges,
        "segment_starts_monotonic": out_of_order == 0,
        "out_of_order_segment_pairs": out_of_order,
        "segment_overlaps_over_50ms": overlaps,
        "maximum_segment_gap_seconds": round(max(gaps), 3) if gaps else 0.0,
        "word_timestamp_count": word_timestamp_count,
        "invalid_word_ranges": invalid_word_ranges,
        "out_of_order_word_pairs": non_monotonic_words,
    }


def _gpu_info() -> dict[str, str | int | None]:
    info: dict[str, str | int | None] = {
        "name": None,
        "driver_version": None,
        "total_memory_mib": None,
        "cuda_device_count": None,
    }
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                name, driver, memory = (part.strip() for part in result.stdout.splitlines()[0].split(",", 2))
                info.update(name=name, driver_version=driver, total_memory_mib=int(memory))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    try:
        import ctranslate2

        info["cuda_device_count"] = ctranslate2.get_cuda_device_count()
    except Exception:
        pass
    return info


def _environment() -> dict:
    def version(package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "faster_whisper": version("faster-whisper"),
        "ctranslate2": version("ctranslate2"),
        "imageio_ffmpeg": version("imageio-ffmpeg"),
        "cuda_runtime_package": version("nvidia-cublas-cu12"),
        "gpu": _gpu_info(),
    }


def _parse_case(value: str) -> BenchmarkCase:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("case must have the form NAME=LANGUAGE=PATH (use '-' for auto-detect)")
    name, language, path = parts
    return BenchmarkCase(name=name, language=None if language == "-" else language, path=Path(path))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark faster-whisper on local speech samples")
    parser.add_argument("input", nargs="?", type=Path, help="legacy single-input form")
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        default=[],
        metavar="NAME=LANGUAGE=PATH",
        help="named case; repeat for different samples/languages, use '-' for language auto-detection",
    )
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"])
    parser.add_argument("--language", help="language for the legacy single-input form")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--compute-type")
    parser.add_argument("--references-dir", type=Path, help="optional NAME.txt references for WER/CER")
    parser.add_argument("--transcripts-dir", type=Path, help="optional directory for per-case/model JSON transcripts")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)
    if bool(args.input) == bool(args.case):
        parser.error("provide either one INPUT or one or more --case options")
    if args.input:
        args.cases = [BenchmarkCase(args.input.stem, args.input, args.language)]
    else:
        args.cases = args.case
    return args


def _save_transcript(directory: Path | None, case: BenchmarkCase, model_name: str, source: Path, transcript) -> str | None:
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case.name)
    path = directory / f"{safe_name}-{model_name}.json"
    payload = transcript.to_dict(filename=source.name, model=model_name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    base_settings = Settings.from_env()
    extractor = AudioExtractor(base_settings.ffmpeg_path)
    report = {
        "version": "1.0",
        "environment": _environment(),
        "methodology": {
            "timed_operation": "model load plus transcription; media extraction excluded",
            "model_order": "models run sequentially in one process per case; CUDA context is reused after first inference",
            "rtf": "processing_seconds / audio_duration_seconds; lower is faster",
            "gpu_memory": "maximum sampled whole-device nvidia-smi memory.used; not process-attributed and may miss peaks",
            "wer": "case-folded, punctuation-insensitive whitespace-delimited word edit rate",
            "cer": "case-folded, punctuation/whitespace-insensitive Unicode character edit rate; Vietnamese diacritics retained",
        },
        "cases": [],
    }
    all_success = True

    try:
        for case in args.cases:
            source = extractor.validate_input(case.path)
            case_report = {
                "name": case.name,
                "language_requested": case.language,
                "source": str(source),
                "duration_seconds": None,
                "results": [],
            }
            reference = None
            if args.references_dir:
                reference_path = args.references_dir / f"{case.name}.txt"
                if reference_path.is_file():
                    reference = reference_path.read_text(encoding="utf-8").strip()
                    case_report["reference_file"] = str(reference_path)

            with extractor.extract(source) as audio:
                case_report["duration_seconds"] = round(audio.duration, 3)
                for model_name in args.models:
                    settings = base_settings.with_overrides(
                        asr_model=model_name,
                        asr_device=args.device,
                        asr_compute_type=args.compute_type,
                    )
                    engine = create_asr_engine(settings)
                    sampler = NvidiaSmiSampler()
                    sampler.start()
                    start = time.perf_counter()
                    try:
                        transcript = engine.transcribe(audio.path, language=case.language)
                        elapsed = time.perf_counter() - start
                        text = " ".join(segment.text for segment in transcript.segments)
                        result = {
                            "model": model_name,
                            "device": settings.asr_device,
                            "compute_type": settings.asr_compute_type,
                            "language_requested": case.language,
                            "language_detected": transcript.language,
                            "audio_duration_seconds": round(audio.duration, 3),
                            "processing_seconds": round(elapsed, 3),
                            "rtf": round(elapsed / audio.duration, 4) if audio.duration else None,
                            "segment_count": len(transcript.segments),
                            "transcript_character_count": len(text),
                            "sampled_whole_gpu_memory_mib": sampler.maximum_mb,
                            "timestamps": _timestamp_quality(transcript, audio.duration),
                            "success": True,
                            "error": None,
                        }
                        transcript_path = _save_transcript(
                            args.transcripts_dir, case, model_name, source, transcript
                        )
                        if transcript_path:
                            result["transcript_file"] = transcript_path
                        if reference is not None:
                            result["quality"] = _quality_metrics(reference, text)
                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        result = {
                            "model": model_name,
                            "device": settings.asr_device,
                            "compute_type": settings.asr_compute_type,
                            "language_requested": case.language,
                            "language_detected": None,
                            "audio_duration_seconds": round(audio.duration, 3),
                            "processing_seconds": round(elapsed, 3),
                            "rtf": round(elapsed / audio.duration, 4) if audio.duration else None,
                            "segment_count": None,
                            "transcript_character_count": None,
                            "sampled_whole_gpu_memory_mib": sampler.maximum_mb,
                            "success": False,
                            "error": str(exc),
                        }
                    finally:
                        sampler.stop()
                        engine.close()
                    case_report["results"].append(result)
                    all_success = all_success and result["success"]
                    print(json.dumps({"case": case.name, **result}, ensure_ascii=False))
            report["cases"].append(case_report)
    except (FileNotFoundError, AudioExtractionError, ASRError) as exc:
        print(f"Benchmark setup error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        if not args.output.parent.exists():
            print(f"Output directory does not exist: {args.output.parent}", file=sys.stderr)
            return 1
        try:
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Could not write benchmark report: {exc}", file=sys.stderr)
            return 1
    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
