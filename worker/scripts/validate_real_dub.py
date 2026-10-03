from __future__ import annotations

import argparse
import array
import json
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.pipeline.speaker_transcription import merge_speaker_transcript


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate completed real-source dubbing artifacts and write validation-report.json"
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir
    asr = _read(output / "asr-benchmark.json")["cases"][0]["results"][0]
    diarization = _read(output / "diarization-benchmark.json")
    translation = _read(output / "translation-benchmark.json")
    tts = _read(output / "tts" / "quality-tts-benchmark.json")
    alignment = _read(output / "alignment-benchmark.json")
    render = _read(output / "render-benchmark.json")
    transcript = _read(output / "transcript.json")
    translated = _read(output / "translated.json")
    asr_transcript = _read(output / "asr" / "transcripts" / "sample-base.json")

    merge_started = time.perf_counter()
    merged = merge_speaker_transcript(
        asr_transcript["segments"], diarization["result"]["segments"], threshold=0.20
    )
    merge_runtime = round(time.perf_counter() - merge_started, 6)

    import imageio_ffmpeg

    final_video = output / "dubbed_video.mp4"
    decode_started = time.perf_counter()
    decode = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(final_video), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    decode_runtime = round(time.perf_counter() - decode_started, 3)

    with wave.open(str(output / "dubbed_timeline.wav"), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        samples = array.array("h")
        samples.frombytes(wav_file.readframes(wav_file.getnframes()))
    pause_checks = []
    for segment in alignment["segments"]:
        pause = float(segment.get("preserved_pause_after", 0.0) or 0.0)
        if pause <= 0:
            continue
        first = round(float(segment["planned_end"]) * sample_rate) * channels
        last = round((float(segment["planned_end"]) + pause) * sample_rate) * channels
        pause_checks.append(
            {
                "segment_id": segment["segment_id"],
                "duration_seconds": pause,
                "verified_silent": not any(samples[first:last]),
            }
        )

    source_readable = args.input_video.is_file() and args.input_video.stat().st_size > 0
    statuses = {
        "source_video_readable": source_readable,
        "asr": bool(asr.get("success")),
        "diarization": bool(diarization.get("success")),
        "speaker_transcript": len(transcript.get("segments", [])) == len(merged)
        and all(item.get("speaker") is not None for item in transcript.get("segments", [])),
        "translation": bool(translation.get("success")),
        "tts": bool(tts.get("success")),
        "alignment": bool(alignment.get("success"))
        and alignment.get("planned_overlaps") == 0
        and alignment.get("actual_overlaps") == 0,
        "render": bool(render.get("success"))
        and bool(render.get("output_has_video"))
        and bool(render.get("output_has_audio")),
        "mp4_decode": decode.returncode == 0,
        "preserved_pauses": bool(pause_checks) and all(pause["verified_silent"] for pause in pause_checks),
        "no_clipping": alignment.get("clipping_count") == 0,
    }
    total_runtime = round(
        (asr.get("processing_seconds") or 0)
        + (diarization.get("total_runtime_seconds") or 0)
        + merge_runtime
        + (translation.get("benchmark", {}).get("total_runtime_seconds") or 0)
        + (tts.get("runtime_seconds") or 0)
        + (alignment.get("alignment_runtime_seconds") or 0)
        + (render.get("render_runtime_seconds") or 0)
        + decode_runtime,
        3,
    )
    truncated_segment_count = int(alignment.get("forcibly_truncated_segments") or 0)
    report = {
        "version": "1.0",
        "overall_result": "PASS" if all(statuses.values()) else "FAIL",
        "success": all(statuses.values()),
        "source_video": str(args.input_video),
        "source_video_readable": source_readable,
        "source_duration_seconds": asr.get("audio_duration_seconds"),
        "detected_language": transcript.get("detected_language"),
        "translation_source_language": translated.get("translation", {}).get(
            "translation_source_language"
        ),
        "translation_target_language": translated.get("translation", {}).get(
            "translation_target_language"
        ),
        "segment_count": len(transcript.get("segments", [])),
        "speaker_count": len(transcript.get("speakers", [])),
        "speakers": transcript.get("speakers", []),
        "planned_overlap_count": alignment.get("planned_overlaps"),
        "actual_overlap_count": alignment.get("actual_overlaps"),
        "preserved_pause_count": len(pause_checks),
        "preserved_pauses_silent_in_output": sum(item["verified_silent"] for item in pause_checks),
        "pause_checks": pause_checks,
        "overflow_segments_before_fitting": alignment.get("overflow_segments_before_fitting"),
        "overflow_segments_after_fitting": alignment.get("overflow_segments_after_fitting"),
        "actual_timeline_overflow_segments_after_truncation": alignment.get(
            "actual_timeline_overflow_segments"
        ),
        "severe_overflow_count": alignment.get("severe_overflow_segments"),
        "tts_duration_before_fitting_seconds": round(
            sum(item.get("tts_duration_before_fit") or 0 for item in tts.get("segments", [])), 6
        ),
        "tts_duration_after_fitting_seconds": alignment.get(
            "total_tts_segment_duration_after_fit_seconds"
        ),
        "final_dubbed_duration_seconds": alignment.get("final_output_duration_seconds"),
        "clipping_count": alignment.get("clipping_count"),
        "render_success": statuses["render"],
        "mp4_decode_success": statuses["mp4_decode"],
        "any_segment_shortened": bool(tts.get("shortened_segments")),
        "shortening_attempted_segments": tts.get("shortening_attempted_segments"),
        "segments_requiring_concise_rephrasing": alignment.get(
            "overflow_segments_after_fitting"
        ),
        "any_segment_forcibly_truncated": bool(alignment.get("any_segment_forcibly_truncated")),
        "forcibly_truncated_segment_count": alignment.get("forcibly_truncated_segments"),
        "stages": {
            "asr": {
                "status": "success" if statuses["asr"] else "failed",
                "runtime_seconds": asr.get("processing_seconds"),
                "model": "base",
                "device": "cuda",
                "segments": asr.get("segment_count"),
            },
            "diarization": {
                "status": "success" if statuses["diarization"] else "failed",
                "runtime_seconds": diarization.get("total_runtime_seconds"),
                "inference_seconds": diarization.get("diarization_seconds"),
                "device": diarization.get("environment", {}).get("device_requested"),
                "speakers": diarization.get("speaker_count"),
            },
            "speaker_transcript_merge": {
                "status": "success" if statuses["speaker_transcript"] else "failed",
                "runtime_seconds": merge_runtime,
                "assigned_segments": sum(item.get("speaker") is not None for item in transcript["segments"]),
            },
            "translation": {
                "status": "success" if statuses["translation"] else "failed",
                "provider": translation.get("translation", {}).get("engine"),
                "device": translation.get("benchmark", {}).get("device"),
                "runtime_seconds": translation.get("benchmark", {}).get("translation_runtime_seconds"),
                "total_runtime_seconds": translation.get("benchmark", {}).get("total_runtime_seconds"),
                "translated_segments": translation.get("benchmark", {}).get("translated_segment_count"),
                "failed_segments": translation.get("benchmark", {}).get("failed_segment_count"),
            },
            "tts": {
                "status": "success" if statuses["tts"] else "failed",
                "runtime_seconds": tts.get("runtime_seconds"),
                "successful_segments": tts.get("successful_segments"),
                "failed_segments": tts.get("failed_segments"),
                "shortening_attempted_segments": tts.get("shortening_attempted_segments"),
                "shortened_segments": tts.get("shortened_segments"),
            },
            "alignment": {
                "status": "success" if statuses["alignment"] else "failed",
                "runtime_seconds": alignment.get("alignment_runtime_seconds"),
                "planned_overlaps": alignment.get("planned_overlaps"),
                "actual_overlaps": alignment.get("actual_overlaps"),
                "preserved_pauses": len(pause_checks),
                "overflow_before": alignment.get("overflow_segments_before_fitting"),
                "overflow_after": alignment.get("overflow_segments_after_fitting"),
                "severe_overflow": alignment.get("severe_overflow_segments"),
                "forcibly_truncated": alignment.get("forcibly_truncated_segments"),
                "clipping_count": alignment.get("clipping_count"),
            },
            "render": {
                "status": "success" if statuses["render"] else "failed",
                "runtime_seconds": render.get("render_runtime_seconds"),
                "output_duration_seconds": render.get("output_duration_seconds"),
                "output_size_bytes": render.get("output_size_bytes"),
                "video_codec": render.get("video_codec"),
                "audio_codec": render.get("audio_codec"),
            },
            "mp4_decode": {
                "status": "success" if statuses["mp4_decode"] else "failed",
                "runtime_seconds": decode_runtime,
                "return_code": decode.returncode,
                "error": decode.stderr.strip() or None,
            },
            "pause_verification": {
                "status": "success" if statuses["preserved_pauses"] else "failed",
                "silent_pauses": sum(item["verified_silent"] for item in pause_checks),
                "pause_count": len(pause_checks),
            },
        },
        "total_runtime_seconds": total_runtime,
        "known_limitations": [
            "No shortening strategy is installed; the provider-neutral hook returns original text after recording an attempt. "
            f"{truncated_segment_count} overlong segments required explicit tail truncation and remain flagged for concise rephrasing.",
            "ASR reports one global language for this mixed Japanese/English source; translation quality was not reviewed.",
            "No subjective speech-quality evaluation was performed.",
        ],
        "errors": {
            "asr": asr.get("error"),
            "diarization": diarization.get("error"),
            "translation": translation.get("error"),
            "tts": tts.get("error"),
            "render": render.get("error"),
            "mp4_decode": decode.stderr.strip() or None,
        },
        "segments": alignment.get("segments", []),
    }
    report_path = output / "validation-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": report["success"], "total_runtime_seconds": total_runtime,
                      "planned_overlaps": report["planned_overlap_count"],
                      "actual_overlaps": report["actual_overlap_count"],
                      "preserved_pauses": report["preserved_pause_count"],
                      "overflow_before": report["overflow_segments_before_fitting"],
                      "overflow_after": report["overflow_segments_after_fitting"],
                      "actual_overflow_after_truncation": report["actual_timeline_overflow_segments_after_truncation"],
                      "shortened": report["any_segment_shortened"],
                      "forcibly_truncated": report["forcibly_truncated_segment_count"],
                      "clipping_count": report["clipping_count"],
                      "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
