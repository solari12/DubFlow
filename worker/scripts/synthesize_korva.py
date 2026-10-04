from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path


PROVIDER = "CUDAExecutionProvider"
SESSION_NAMES = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
_DLL_HANDLES: list[object] = []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize translated DubFlow segments with KorvaTTS")
    parser.add_argument("input", type=Path, help="Translated transcript JSON")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--voice", default="gia_bao")
    parser.add_argument("--device", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path)
    return parser


def _prepare_cuda_dlls() -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    worker_root = Path(__file__).resolve().parents[1]
    site = worker_root / ".venv-tts-korva" / "Lib" / "site-packages" / "nvidia"
    for path in (site / "cu13" / "bin" / "x86_64", site / "cudnn" / "bin"):
        if not path.is_dir():
            raise FileNotFoundError(f"KorvaTTS CUDA DLL directory is missing: {path}")
        _DLL_HANDLES.append(os.add_dll_directory(str(path.resolve())))


def _metadata(path: Path) -> dict:
    with wave.open(str(path), "rb") as audio:
        frames, rate = audio.getnframes(), audio.getframerate()
        channels, width, compression = audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()
    if frames <= 0 or rate <= 0 or channels <= 0 or width != 2 or compression != "NONE":
        raise RuntimeError(f"KorvaTTS did not create a non-empty PCM WAV: {path}")
    return {
        "duration": frames / rate,
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "frame_count": frames,
        "file_size_bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.steps <= 32:
        raise SystemExit("--steps must be between 1 and 32")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "korva-report.json"
    report = {
        "provider": "KorvaTTS",
        "voice": args.voice,
        "language": args.language,
        "device": args.device,
        "steps": args.steps,
        "segment_count": 0,
        "successful_segments": 0,
        "failed_segments": 0,
        "tts_runtime_seconds": 0.0,
        "audio_duration_seconds": 0.0,
        "rtf": None,
        "onnx_available_providers": [],
        "onnx_provider_by_session": {},
        "segments": [],
    }
    try:
        _prepare_cuda_dlls()
        import onnxruntime as ort
        from korvatts import TTS
        from korvatts.assets import resolve_assets_dir

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        report["onnx_runtime_version"] = ort.__version__
        report["onnx_available_providers"] = ort.get_available_providers()
        transcript = json.loads(args.input.read_text(encoding="utf-8-sig"))
        segments = transcript.get("segments", [])
        report["segment_count"] = len(segments)
        assets_dir = args.assets_dir
        if assets_dir is None:
            try:
                assets_dir = resolve_assets_dir(auto_download=False)
            except FileNotFoundError:
                from huggingface_hub import snapshot_download

                assets_dir = Path(snapshot_download(
                    repo_id="dogenthq/KorvaTTS",
                    allow_patterns=["onnx/*", "voice_styles/*"],
                    local_files_only=True,
                ))
        report["assets_dir"] = str(Path(assets_dir).resolve())
        tts = TTS(assets_dir=str(assets_dir), device=args.device, auto_download=False)
        report["onnx_provider_by_session"] = {
            name: list(getattr(tts.sessions, name).get_providers()) for name in SESSION_NAMES
        }
        if args.device == "gpu" and any(
            not providers or providers[0] != PROVIDER
            for providers in report["onnx_provider_by_session"].values()
        ):
            raise RuntimeError(
                "KorvaTTS GPU verification failed; CUDAExecutionProvider is not primary "
                f"for every ONNX session: {report['onnx_provider_by_session']}"
            )
        if args.voice not in tts.list_voices():
            raise ValueError(f"KorvaTTS voice {args.voice!r} is unavailable")

        audio_dir = args.output / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for segment in segments:
            segment_id = int(segment["id"])
            text = segment.get("target_text")
            path = audio_dir / f"segment-{segment_id:04d}.wav"
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Segment {segment_id} has no translated target_text")
            started = time.perf_counter()
            samples, _duration = tts.synthesize(
                text, voice=args.voice, lang=args.language, total_steps=args.steps
            )
            runtime = time.perf_counter() - started
            tts.save_audio(samples, str(path))
            metadata = _metadata(path)
            row = {
                "id": segment_id,
                "audio_path": str(path.resolve()),
                "engine": "KorvaTTS",
                "voice": args.voice,
                "device": args.device,
                "runtime": runtime,
                **metadata,
            }
            report["segments"].append(row)
            report["tts_runtime_seconds"] += runtime
            report["audio_duration_seconds"] += metadata["duration"]
        report["successful_segments"] = len(report["segments"])
        report["failed_segments"] = report["segment_count"] - report["successful_segments"]
        report["rtf"] = (
            report["tts_runtime_seconds"] / report["audio_duration_seconds"]
            if report["audio_duration_seconds"] > 0 else None
        )
        if report["failed_segments"]:
            raise RuntimeError(f"KorvaTTS failed for {report['failed_segments']} segment(s)")
        report["success"] = True
    except Exception as exc:
        report["success"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
