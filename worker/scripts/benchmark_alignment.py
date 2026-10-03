from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.models.alignment import AudioAlignmentSettings
from dubflow_worker.models.dubbing_quality import DubbingQualitySettings, is_severe_overflow
from dubflow_worker.pipeline.align_audio import align_translated_transcript


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark transcript-to-audio alignment")
    parser.add_argument("transcript", type=Path, help="Phase 1C or Phase 1D transcript JSON")
    parser.add_argument("tts_output_dir", type=Path, help="Directory containing Phase 1D WAVs")
    parser.add_argument("--output", type=Path, required=True, help="Phase 1E output directory")
    parser.add_argument("--min-stretch", type=float, default=0.85)
    parser.add_argument("--max-stretch", type=float, default=1.10)
    parser.add_argument("--max-overflow-ratio", type=float, default=1.50)
    parser.add_argument("--overflow-policy", choices=("preserve", "trim", "fail"), default="trim")
    parser.add_argument("--no-pad-short", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--format", choices=("wav",), default="wav")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--source-media", type=Path, help="optional original media for pause detection")
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
    quality_settings = DubbingQualitySettings(
        min_time_stretch_ratio=args.min_stretch,
        max_time_stretch_ratio=args.max_stretch,
        max_overflow_ratio=args.max_overflow_ratio,
    )
    aligned = align_translated_transcript(
        transcript,
        tts_output_dir=args.tts_output_dir,
        output_dir=args.output,
        settings=settings,
        transcript_path=args.transcript.resolve(),
        source_media=args.source_media,
    )
    input_info = {
        "path": str(args.transcript),
        "version": transcript.get("version"),
        "translation": transcript.get("translation"),
        "tts_engine": transcript.get("engine"),
        "tts_model": transcript.get("model"),
    }
    alignment_report = aligned.to_dict(input_transcript=input_info)

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
    input_segments = {
        int(segment.get("id", segment.get("segment_id", -1))): segment
        for segment in transcript.get("segments", [])
    }
    fitted_durations = [item.final_audio_duration for item in aligned.segments if item.final_audio_duration is not None]
    source_durations = [item.original_tts_duration for item in aligned.segments if item.original_tts_duration is not None]
    severe_overflows = [
        item for item in aligned.segments
        if item.original_tts_duration is not None
        and is_severe_overflow(item.original_tts_duration, item.target_duration, quality_settings)
    ]
    fit_segments = [
        item for item in aligned.segments
        if item.status == "success"
        and item.stretch_factor is not None
        and item.stretch_factor >= quality_settings.min_time_stretch_ratio
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
        "overflow_segments_before_fitting": sum(item.overflow_before_fitting for item in aligned.segments),
        "overflow_segments_after_fitting": sum(item.overflow_after_fitting for item in aligned.segments),
        "actual_timeline_overflow_segments": sum(
            item.final_audio_duration is not None
            and item.final_audio_duration > item.allowed_duration + 0.5 / aligned.sample_rate
            for item in aligned.segments
        ),
        "severe_overflow_segments": len(severe_overflows),
        "planned_overlaps": aligned.planned_overlap_count,
        "actual_overlaps": aligned.actual_overlap_count,
        "preserved_pauses": sum(item.preserved_pause_after > 0 for item in aligned.segments),
        "shortening_attempted_segments": sum(item.shortening_attempted for item in aligned.segments),
        "shortened_segments": sum(item.shortened for item in aligned.segments),
        "forcibly_truncated_segments": sum(item.forcibly_truncated for item in aligned.segments),
        "any_segment_shortened": any(item.shortened for item in aligned.segments),
        "any_segment_forcibly_truncated": any(item.forcibly_truncated for item in aligned.segments),
        "max_overflow_ratio": quality_settings.max_overflow_ratio,
        "segments_fitting_without_severe_stretch_percent": round(
            100 * len(fit_segments) / len(aligned.segments), 2
        ) if aligned.segments else None,
        "total_tts_segment_duration_before_fit_seconds": round(sum(source_durations), 6),
        "total_tts_segment_duration_after_fit_seconds": round(sum(fitted_durations), 6),
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
        "segments": [
            {
                **item.to_dict(),
                "original_translation": input_segments.get(item.segment_id, {}).get(
                    "original_translation", input_segments.get(item.segment_id, {}).get("target_text")
                ),
                "final_translation": input_segments.get(item.segment_id, {}).get(
                    "final_translation", input_segments.get(item.segment_id, {}).get("target_text")
                ),
                "translation_retry_count": input_segments.get(item.segment_id, {}).get("translation_retry_count", 0),
                "tts_duration_before_fit": item.original_tts_duration,
                "tts_duration_after_fit": item.final_audio_duration,
                "time_stretch_ratio": item.stretch_factor,
                "overflow": item.status == "overflow",
                "overflow_before_fitting": item.overflow_before_fitting,
                "overflow_after_fitting": item.overflow_after_fitting,
                "requires_concise_rephrasing": item.requires_concise_rephrasing,
                "shortening_attempted": item.shortening_attempted or bool(
                    input_segments.get(item.segment_id, {}).get("shortening_attempted")
                ),
                "shortened": item.shortened or bool(
                    input_segments.get(item.segment_id, {}).get("shortened")
                ),
                "forcibly_truncated": item.forcibly_truncated,
                "truncated_duration": item.truncated_duration,
                "overflow_duration": max(
                    0.0,
                    (item.final_audio_duration or 0.0) - item.target_duration,
                ) if item.status == "overflow" else 0.0,
                "severe_overflow": item in severe_overflows,
            }
            for item in aligned.segments
        ],
    }
    alignment_report.update({
        "severe_overflow_count": len(severe_overflows),
        "max_overflow_ratio": quality_settings.max_overflow_ratio,
        "segments_fitting_without_severe_stretch_percent": benchmark[
            "segments_fitting_without_severe_stretch_percent"
        ],
        "total_tts_segment_duration_before_fit_seconds": benchmark[
            "total_tts_segment_duration_before_fit_seconds"
        ],
        "total_tts_segment_duration_after_fit_seconds": benchmark[
            "total_tts_segment_duration_after_fit_seconds"
        ],
        "segments": benchmark["segments"],
    })
    (args.output / "alignment.json").write_text(
        json.dumps(alignment_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "alignment-benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(benchmark, ensure_ascii=True))
    return 0 if aligned.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
