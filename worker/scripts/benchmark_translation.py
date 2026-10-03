from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.pipeline.translate import detected_source_language, translate_speaker_transcript
from dubflow_worker.translation.factory import create_translation_engine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local transcript translation")
    parser.add_argument("input", type=Path, help="speaker-aware transcript JSON")
    parser.add_argument("--source-language", help="explicit override; otherwise use detected ASR language")
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("argos", "nllb"), default="argos")
    parser.add_argument("--model", default=None, help="NLLB model name or local model path")
    parser.add_argument(
        "--device", default=None, help="NLLB device; defaults to cpu or DUBFLOW_NLLB_DEVICE"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report: dict = {
        "version": "1.0",
        "success": False,
        "translation": {
            "engine": "argos-translate",
            "detected_language": None,
            "translation_source_language": args.source_language,
            "translation_target_language": args.target_language,
            "source_language_overridden": False,
        },
        "benchmark": {
            "model_load_runtime_seconds": None,
            "total_runtime_seconds": None,
            "translation_runtime_seconds": None,
            "input_segment_count": 0,
            "translated_segment_count": 0,
            "failed_segment_count": 0,
            "detected_language": None,
            "translation_source_language": args.source_language,
            "translation_target_language": args.target_language,
            "target_language": args.target_language,
            "engine": args.provider,
            "model": args.model,
            "device": (args.device or "cpu") if args.provider == "nllb" else "cpu",
        },
        "segments": [],
    }
    try:
        speaker_transcript = json.loads(args.input.read_text(encoding="utf-8"))
        detected = detected_source_language(speaker_transcript)
        report["translation"]["detected_language"] = detected
        report["translation"]["translation_source_language"] = args.source_language or detected
        report["translation"]["source_language_overridden"] = bool(
            args.source_language and args.source_language.lower() != detected
        )
        report["benchmark"]["detected_language"] = detected
        report["benchmark"]["translation_source_language"] = args.source_language or detected
        report["benchmark"]["source_language_overridden"] = bool(
            args.source_language and args.source_language.lower() != detected
        )
        report["benchmark"]["input_segment_count"] = len(speaker_transcript.get("segments", []))
        started = time.perf_counter()
        engine = create_translation_engine(args.provider, model=args.model, device=args.device)
        report["benchmark"]["engine"] = engine.name
        report["benchmark"]["device"] = engine.device
        report["benchmark"]["model"] = getattr(engine, "model_name", engine.name)
        load_started = time.perf_counter()
        if hasattr(engine, "load"):
            engine.load()
        report["benchmark"]["model_load_runtime_seconds"] = round(
            time.perf_counter() - load_started, 6
        )
        translation_started = time.perf_counter()
        translated = translate_speaker_transcript(
            speaker_transcript,
            source_language=args.source_language,
            target_language=args.target_language,
            engine=engine,
        )
        report["benchmark"]["translation_runtime_seconds"] = round(
            time.perf_counter() - translation_started, 6
        )
        report["benchmark"]["total_runtime_seconds"] = round(time.perf_counter() - started, 6)
        report["translation"].update(translated.to_dict()["translation"])
        report["translation"]["engine"] = translated.engine
        report["benchmark"]["engine"] = translated.engine
        report["benchmark"]["translation_source_language"] = translated.source_language
        report["benchmark"]["translation_target_language"] = translated.target_language
        report["benchmark"]["source_language_overridden"] = translated.source_language_overridden
        report["segments"] = [segment.to_dict() for segment in translated.segments]
        report["benchmark"]["translated_segment_count"] = sum(
            segment.target_text is not None and bool(segment.source_text.strip())
            for segment in translated.segments
        )
        report["benchmark"]["failed_segment_count"] = sum(
            segment.target_text is None for segment in translated.segments
        )
        report["success"] = report["benchmark"]["failed_segment_count"] == 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["benchmark"]["total_runtime_seconds"] is None:
            report["benchmark"]["total_runtime_seconds"] = (
                round(time.perf_counter() - started, 6) if "started" in locals() else None
            )
        report["benchmark"]["failed_segment_count"] = report["benchmark"][
            "input_segment_count"
        ]
        print(report["error"], file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
