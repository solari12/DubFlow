from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.pipeline.translate import translate_speaker_transcript
from dubflow_worker.translation.argos import ArgosTranslationEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local transcript translation")
    parser.add_argument("input", type=Path, help="speaker-aware transcript JSON")
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("argos",), default="argos")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report: dict = {
        "version": "1.0",
        "success": False,
        "translation": {
            "engine": "argos-translate",
            "source_language": args.source_language,
            "target_language": args.target_language,
        },
        "benchmark": {
            "translation_runtime_seconds": None,
            "input_segment_count": 0,
            "translated_segment_count": 0,
            "failed_segment_count": 0,
            "source_language": args.source_language,
            "target_language": args.target_language,
            "engine": "argos-translate",
            "device": "cpu",
        },
        "segments": [],
    }
    try:
        speaker_transcript = json.loads(args.input.read_text(encoding="utf-8"))
        report["benchmark"]["input_segment_count"] = len(speaker_transcript.get("segments", []))
        started = time.perf_counter()
        engine = ArgosTranslationEngine()
        translated = translate_speaker_transcript(
            speaker_transcript,
            source_language=args.source_language,
            target_language=args.target_language,
            engine=engine,
        )
        report["benchmark"]["translation_runtime_seconds"] = round(
            time.perf_counter() - started, 6
        )
        report["translation"]["engine"] = translated.engine
        report["benchmark"]["engine"] = translated.engine
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
        print(report["error"], file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
