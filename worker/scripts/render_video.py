from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.video.ffmpeg import FFmpegVideoRenderer, RenderSettings, write_render_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replace a video's audio with DubFlow dubbed audio")
    parser.add_argument("video", type=Path, help="original video file")
    parser.add_argument("dubbed_audio", type=Path, help="Phase 1E dubbed timeline WAV")
    parser.add_argument("--output", type=Path, required=True, help="output MP4 path")
    parser.add_argument("--audio-codec", choices=("aac",), default="aac")
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--video-mode", choices=("copy", "h264"), default="copy")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = RenderSettings(
        audio_codec=args.audio_codec,
        audio_bitrate=args.audio_bitrate,
        video_mode=args.video_mode,
        ffmpeg_path=args.ffmpeg,
    )
    result = FFmpegVideoRenderer(settings).render(args.video, args.dubbed_audio, args.output)
    result_path = args.output.with_name("render-result.json")
    write_render_result(result_path, result)
    print(json.dumps(result.to_dict(), indent=2))
    if result.status != "success":
        print(f"Render failed; details written to {result_path}", file=sys.stderr)
        return 1
    print(f"Rendered MP4: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
