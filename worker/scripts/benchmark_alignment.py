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
    parser = argparse.ArgumentParser(description="Benchmark transcript-to-audio alignment")
    parser.add_argument("transcript", type=Path, help="Phase 1C or Phase 1D transcript JSON")
    parser.add_argument("tts_output_dir", type=Path, help="Directory containing Phase 1D WAVs")
    parser.add_argument("--output", type=Path, required=True, help="Phase 1E output directory")
    parser.add_argument("--min-stretch", type=float, default=0.90)
    parser.add_argument("--max-stretch", type=float, default=1.10)
    parser.add_argument("--overflow-policy", choices=("preserve", "trim", "fail"), default="preserve")
    parser.add_argument("--no-pad-short", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--format", choices=("wav",), default="wav")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
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
        tts_output_dir=args.tts_output_dir,
        output_dir=args.output,
        settings=settings,
        transcript_path=args.transcript.resolve(),
    )
    input_info = {
        "path": str(args.transcript),
        "version": transcript.get("version"),
        "translation": transcript.get("translation"),
        "tts_engine": transcript.get("engine"),
        "tts_model": transcript.get("model"),
    }
    alignment_report = aligned.to_dict(input_transcript=input_info)
    (args.output / "alignment.json").write_text(
        json.dumps(alignment_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    duration_errors = [
        abs(item.duration_error)
        for item in aligned.segments
        if item.duration_error is not None
    ]
    original_tts_durations = [
        item.original_tts_duration
        for item in aligned.segments
        if item.original_tts_duration is not None
    ]
    stretch_factors = [
        item.stretch_factor
        for item in aligned.segments
        if item.alignment_action == "time_stretch" and item.stretch_factor is not None
    ]
    benchmark = {
        "version": "1.0",
        "success": aligned.failed_count == 0,
        "segment_count": len(aligned.segments),
        "total_source_timeline_duration_seconds": aligned.target_timeline_duration,
        "total_tts_duration_seconds": round(sum(original_tts_durations), 6),
        "alignment_runtime_seconds": round(aligned.runtime_seconds, 6),
        "rtf": round(aligned.rtf, 6) if aligned.rtf is not None else None,
        "successful_segments": aligned.successful_count,
        "failed_segments": aligned.failed_count,
        "overflow_segments": aligned.overflow_count,
        "maximum_absolute_duration_error_seconds": round(max(duration_errors), 6)
        if duration_errors
        else None,
        "average_absolute_duration_error_seconds": round(sum(duration_errors) / len(duration_errors), 6)
        if duration_errors
        else None,
        "stretch_factor_statistics": {
            "count": len(stretch_factors),
            "minimum": min(stretch_factors) if stretch_factors else None,
            "maximum": max(stretch_factors) if stretch_factors else None,
            "average": round(sum(stretch_factors) / len(stretch_factors), 6)
            if stretch_factors
            else None,
        },
        "final_output_duration_seconds": aligned.total_duration,
        "output_sample_rate": aligned.sample_rate,
        "output_channels": aligned.channels,
        "peak_amplitude": aligned.peak_amplitude,
        "clipping_count": aligned.clipping_count,
        "overlap_policy": aligned.overlap_policy,
        "output_audio_path": str(aligned.timeline_path),
    }
    (args.output / "alignment-benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(benchmark, ensure_ascii=True))
    return 0 if aligned.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
