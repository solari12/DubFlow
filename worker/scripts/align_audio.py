from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.models.alignment import AudioAlignmentSettings
from dubflow_worker.pipeline.align_audio import align_translated_transcript


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align Phase 1D TTS segments to transcript timing")
    parser.add_argument("transcript", type=Path, help="Phase 1C or Phase 1D transcript JSON")
    parser.add_argument("tts_output_dir", type=Path, help="Directory containing Phase 1D WAVs")
    parser.add_argument("--output", type=Path, required=True, help="Phase 1E output directory")
    parser.add_argument("--min-stretch", type=float, default=0.90)
    parser.add_argument("--max-stretch", type=float, default=1.10)
    parser.add_argument("--overflow-policy", choices=("preserve", "trim", "fail"), default="preserve")
    parser.add_argument("--no-pad-short", action="store_true", help="Leave short audio unpadded")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--format", choices=("wav",), default="wav")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def run(transcript_path: Path, tts_output_dir: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    settings = AudioAlignmentSettings(
        min_stretch_factor=args.min_stretch,
        max_stretch_factor=args.max_stretch,
        pad_short_audio=not args.no_pad_short,
        overflow_policy=args.overflow_policy,
        sample_rate=args.sample_rate,
        channels=args.channels,
        output_format=args.format,
        ffmpeg_path=args.ffmpeg,
    )
    aligned = align_translated_transcript(
        transcript,
        tts_output_dir=tts_output_dir,
        output_dir=output_dir,
        settings=settings,
        transcript_path=transcript_path.resolve(),
    )
    input_info = {
        "path": str(transcript_path),
        "version": transcript.get("version"),
        "translation": transcript.get("translation"),
        "tts": {
            "engine": transcript.get("engine"),
            "model": transcript.get("model"),
            "device": transcript.get("device"),
        },
    }
    report = aligned.to_dict(input_transcript=input_info)
    (output_dir / "alignment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args.transcript, args.tts_output_dir, args.output, args)
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
