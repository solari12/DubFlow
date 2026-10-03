from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dubflow_worker.asr.base import ASRError
from dubflow_worker.asr.factory import create_asr_engine
from dubflow_worker.audio.extractor import AudioExtractionError
from dubflow_worker.config.settings import Settings
from dubflow_worker.pipeline.transcribe import TranscriptionPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dubflow-worker", description="DubFlow local AI worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    transcribe = subparsers.add_parser("transcribe", help="transcribe a local audio/video file")
    transcribe.add_argument("input", type=Path, help="input media path")
    transcribe.add_argument("--language", help="language code, e.g. en or vi; omit to detect")
    transcribe.add_argument("--model", help="faster-whisper model (default: DUBFLOW_ASR_MODEL or base)")
    transcribe.add_argument("--device", choices=("cuda", "cpu"), help="execution device")
    transcribe.add_argument("--compute-type", help="CTranslate2 compute type, e.g. float16 or int8")
    transcribe.add_argument("--output", type=Path, default=Path("output/transcript.json"))
    return parser


def _write_json(output_path: Path, payload: dict) -> None:
    parent = output_path.expanduser().parent
    if not parent.exists() or not parent.is_dir():
        raise OSError(f"Output directory does not exist: {parent}")
    try:
        output_path.expanduser().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise OSError(f"Could not write transcript to '{output_path}': {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "transcribe":
        return 2

    try:
        print("[1/4] Validating input...", flush=True)
        settings = Settings.from_env().with_overrides(
            asr_model=args.model,
            asr_device=args.device,
            asr_compute_type=args.compute_type,
        )
        engine = create_asr_engine(settings)
        from dubflow_worker.audio.extractor import AudioExtractor

        extractor = AudioExtractor(settings.ffmpeg_path)
        pipeline = TranscriptionPipeline(engine, extractor)

        def progress(message: str) -> None:
            if message == "Extracting audio...":
                print("[2/4] Extracting audio...", flush=True)
            elif message == "Running ASR...":
                print("[3/4] Running ASR...", flush=True)

        transcript = pipeline.run(args.input, language=args.language, progress=progress)
        payload = transcript.to_dict(filename=args.input.name, model=settings.asr_model)
        print("[4/4] Writing transcript...", flush=True)
        _write_json(args.output, payload)
        print(f"Done. Transcript written to {args.output}")
        return 0
    except (FileNotFoundError, AudioExtractionError, ASRError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # Keep unexpected runtime failures concise for CLI users.
        print(f"Error: transcription failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
