from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.video.ffmpeg import FFmpegVideoRenderer, RenderSettings, write_render_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Phase 1F video rendering")
    parser.add_argument("video", type=Path, help="original video file")
    parser.add_argument("dubbed_audio", type=Path, help="Phase 1E dubbed timeline WAV")
    parser.add_argument("--output-dir", type=Path, default=Path("output/phase1f"))
    parser.add_argument("--video-mode", choices=("copy", "h264"), default="copy")
    parser.add_argument("--audio-codec", choices=("aac",), default="aac")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path = args.output_dir / "dubbed_video.mp4"
    result = FFmpegVideoRenderer(
        RenderSettings(
            audio_codec=args.audio_codec,
            audio_bitrate=args.audio_bitrate,
            video_mode=args.video_mode,
            ffmpeg_path=args.ffmpeg,
        )
    ).render(args.video, args.dubbed_audio, output_path)
    write_render_result(args.output_dir / "render-result.json", result)
    video_size = args.video.stat().st_size if args.video.is_file() else None
    benchmark = {
        "success": result.status == "success",
        "input_video": result.input_video,
        "input_audio": result.input_audio,
        "output_video": result.output_video,
        "source_video_duration_seconds": result.source_video_duration,
        "source_audio_duration_seconds": result.source_audio_duration,
        "dubbed_audio_duration_seconds": result.dubbed_audio_duration,
        "output_duration_seconds": result.output_duration,
        "render_runtime_seconds": result.runtime_seconds,
        "render_rtf": (
            round(result.runtime_seconds / result.source_video_duration, 6)
            if result.source_video_duration
            else None
        ),
        "input_size_bytes": video_size,
        "dubbed_audio_size_bytes": args.dubbed_audio.stat().st_size if args.dubbed_audio.is_file() else None,
        "output_size_bytes": result.output_size_bytes,
        "video_codec": result.video_codec,
        "audio_codec": result.audio_codec,
        "video_stream_copied": result.video_stream_copied,
        "audio_reencoded": result.audio_reencoded,
        "duration_mismatch_seconds": result.duration_mismatch,
        "duration_policy": result.duration_policy,
        "output_has_video": result.output_has_video,
        "output_has_audio": result.output_has_audio,
        "status": result.status,
        "error": result.error,
    }
    (args.output_dir / "render-benchmark.json").write_text(
        json.dumps(benchmark, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(benchmark, indent=2))
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
