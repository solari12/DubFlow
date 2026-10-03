from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.models.dubbing_quality import DubbingQualitySettings
from dubflow_worker.translation.base import TranslationEngine
from dubflow_worker.pipeline.tts_feedback import synthesize_with_duration_feedback
from dubflow_worker.tts.sherpa_onnx import SherpaOnnxPiperEngine


class _NoRephraseProvider(TranslationEngine):
    name = "no-rephrase-provider"
    device = "cpu"

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        raise NotImplementedError("TTS feedback does not translate source text")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize speech with bounded translation feedback")
    parser.add_argument("input", type=Path, help="translated transcript JSON")
    parser.add_argument("--output", type=Path, required=True, help="output directory for WAV and reports")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--min-time-stretch-ratio", type=float, default=0.85)
    parser.add_argument("--max-time-stretch-ratio", type=float, default=1.10)
    parser.add_argument("--max-translation-expansion-ratio", type=float, default=1.25)
    parser.add_argument("--max-overflow-ratio", type=float, default=1.50)
    parser.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    report: dict = {"success": False, "segments": [], "error": None}
    try:
        transcript = json.loads(args.input.read_text(encoding="utf-8"))
        metadata = transcript.get("translation", {})
        source_language = metadata.get("translation_source_language")
        target_language = metadata.get("translation_target_language") or args.language
        detected_language = metadata.get("detected_language")
        if not source_language:
            raise ValueError("Translated transcript does not record its resolved source language")
        translator = _NoRephraseProvider()
        model_dir = args.model_dir or (
            Path(__file__).resolve().parents[1]
            / ".model-cache"
            / "tts"
            / "vits-piper-vi_VN-vivos-x_low"
        )
        tts_engine = SherpaOnnxPiperEngine(model_dir, num_threads=2)
        settings = DubbingQualitySettings(
            min_time_stretch_ratio=args.min_time_stretch_ratio,
            max_time_stretch_ratio=args.max_time_stretch_ratio,
            max_translation_expansion_ratio=args.max_translation_expansion_ratio,
            max_overflow_ratio=args.max_overflow_ratio,
            max_translation_retries=args.max_retries,
        )
        results = synthesize_with_duration_feedback(
            transcript.get("segments", []),
            source_language=source_language,
            target_language=target_language,
            tts_engine=tts_engine,
            translation_engine=translator,
            output_dir=args.output / "audio",
            settings=settings,
        )
        result_by_id = {item.segment_id: item for item in results}
        for segment in transcript.get("segments", []):
            fit = result_by_id[int(segment["id"])]
            segment["original_translation"] = fit.original_translation
            segment["final_translation"] = fit.final_translation
            segment["target_text"] = fit.final_translation
            segment["translation_retry_count"] = fit.translation_retry_count
            segment["shortening_attempted"] = fit.shortening_attempted
            segment["shortened"] = fit.shortened
            segment["requires_concise_rephrasing"] = fit.requires_concise_rephrasing
            segment["forcibly_truncated"] = fit.forcibly_truncated
            segment["duration_fit"] = fit.to_dict()
            audio_info = None
            if fit.audio_path:
                with wave.open(str(fit.audio_path), "rb") as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    sample_rate = wav_file.getframerate()
                audio_info = {
                    "audio_path": str(fit.audio_path),
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "engine": "sherpa-onnx-piper-vivos-x_low",
                    "language": target_language,
                    "speaker": fit.speaker,
                }
            segment["audio"] = audio_info

        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "translated.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        successful = [item for item in results if item.audio_path]
        report = {
            "success": bool(results) and len(successful) == len(results),
            "detected_language": detected_language,
            "translation_source_language": source_language,
            "translation_target_language": target_language,
            "settings": {
                "min_time_stretch_ratio": settings.min_time_stretch_ratio,
                "max_time_stretch_ratio": settings.max_time_stretch_ratio,
                "max_translation_expansion_ratio": settings.max_translation_expansion_ratio,
                "max_overflow_ratio": settings.max_overflow_ratio,
            "max_translation_retries": settings.max_translation_retries,
            "translation_rephrasing_available": False,
            },
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "segment_count": len(results),
            "successful_segments": len(successful),
            "failed_segments": len(results) - len(successful),
            "overflow_segments": sum(item.overflow for item in results),
            "severe_overflow_segments": sum(item.severe_overflow for item in results),
            "translation_retry_count": sum(item.translation_retry_count for item in results),
            "shortening_attempted_segments": sum(item.shortening_attempted for item in results),
            "shortened_segments": sum(item.shortened for item in results),
            "segments_requiring_concise_rephrasing": sum(
                item.requires_concise_rephrasing for item in results
            ),
            "segments": [item.to_dict() for item in results],
            "error": None,
        }
    except Exception as exc:
        report = {
            **report,
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "quality-tts-benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
