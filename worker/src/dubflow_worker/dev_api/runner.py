from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from dubflow_worker.asr.factory import create_asr_engine
from dubflow_worker.audio.extractor import AudioExtractor
from dubflow_worker.config.settings import Settings
from dubflow_worker.diarization.audio import load_audio_waveform
from dubflow_worker.diarization.pyannote import PyannoteDiarizer
from dubflow_worker.models.alignment import AudioAlignmentSettings
from dubflow_worker.pipeline.align_audio import align_translated_transcript
from dubflow_worker.pipeline.speaker_transcription import merge_speaker_transcript
from dubflow_worker.pipeline.transcribe import TranscriptionPipeline
from dubflow_worker.pipeline.translate import translate_speaker_transcript
from dubflow_worker.translation.factory import create_translation_engine
from dubflow_worker.video.ffmpeg import FFmpegVideoRenderer, RenderSettings


STAGES = {
    "ASR": 15,
    "DIARIZATION": 30,
    "TRANSLATION": 45,
    "TTS": 65,
    "ALIGNMENT": 80,
    "RENDERING": 95,
    "COMPLETED": 100,
}
Progress = Callable[[str, int], None]


def run_dubbing(
    video: Path,
    output_dir: Path,
    *,
    source_language: str,
    target_language: str,
    voice: str,
    device: str = "gpu",
    steps: int = 32,
    progress: Progress,
) -> Path:
    """Run the existing Phase 1 modules in sequence; no stage is simulated."""
    if device not in {"gpu", "cpu", "auto"}:
        raise ValueError("TTS device must be gpu, cpu, or auto")
    if not 1 <= steps <= 32:
        raise ValueError("KorvaTTS steps must be between 1 and 32")

    output_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env()
    extractor = AudioExtractor(settings.ffmpeg_path)
    progress("ASR", STAGES["ASR"])
    transcript = TranscriptionPipeline(create_asr_engine(settings), extractor).run(
        video, language=None if source_language == "auto" else source_language
    )
    asr_data = transcript.to_dict(filename=video.name, model=settings.asr_model)
    (output_dir / "asr.json").write_text(json.dumps(asr_data, ensure_ascii=False, indent=2), encoding="utf-8")

    progress("DIARIZATION", STAGES["DIARIZATION"])
    diarizer = PyannoteDiarizer(
        model=os.getenv("DUBFLOW_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"),
        device=os.getenv("DUBFLOW_DIARIZATION_DEVICE", "cpu"),
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
    )
    try:
        diarization = diarizer.diarize(load_audio_waveform(video, extractor))
    finally:
        diarizer.close()
    # pyannote.audio 4 returns a DiarizeOutput wrapper; earlier versions may
    # return the Annotation directly. Match the existing benchmark adapter.
    annotation = getattr(diarization, "speaker_diarization", diarization)
    turns = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
        for turn, _track, label in annotation.itertracks(yield_label=True)
    ]
    segments = merge_speaker_transcript(asr_data["segments"], turns)
    speaker_data = {**asr_data, "segments": segments}

    progress("TRANSLATION", STAGES["TRANSLATION"])
    provider = os.getenv("DUBFLOW_TRANSLATION_PROVIDER", "nllb")
    translation = create_translation_engine(provider)
    if hasattr(translation, "load"):
        translation.load()
    translated = translate_speaker_transcript(
        speaker_data,
        source_language=None if source_language == "auto" else source_language,
        target_language=target_language,
        engine=translation,
    )
    translated_data = translated.to_dict()
    translated_data["segments"] = [s.to_dict() for s in translated.segments]
    translated_path = output_dir / "translated.json"
    translated_path.write_text(json.dumps(translated_data, ensure_ascii=False, indent=2), encoding="utf-8")

    progress("TTS", STAGES["TTS"])
    # Korva runs in its own environment so its ONNX Runtime and CUDA packages
    # stay separate from the API/diarization environment.
    worker_root = Path(__file__).resolve().parents[3]
    tts_python = Path(os.getenv("DUBFLOW_TTS_PYTHON", "")) if os.getenv("DUBFLOW_TTS_PYTHON") else (
        worker_root / ".venv-tts-korva" / "Scripts" / "python.exe"
    )
    if not tts_python.is_file():
        raise FileNotFoundError(f"KorvaTTS Python environment was not found: {tts_python}")
    tts_output = output_dir / "tts"
    tts_process = subprocess.run(
        [
            str(tts_python),
            str(worker_root / "scripts" / "synthesize_korva.py"),
            str(translated_path),
            "--language",
            target_language,
            "--voice",
            voice,
            "--device",
            device,
            "--steps",
            str(steps),
            "--output",
            str(tts_output),
        ],
        cwd=worker_root.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if tts_process.returncode:
        details = (tts_process.stderr or tts_process.stdout)[-3000:]
        raise RuntimeError(f"TTS stage failed (exit {tts_process.returncode}): {details}")
    tts_report = json.loads((tts_output / "korva-report.json").read_text(encoding="utf-8"))
    provider_by_session = tts_report.get("onnx_provider_by_session", {})
    if (
        not tts_report.get("success")
        or tts_report.get("provider") != "KorvaTTS"
        or tts_report.get("voice") != voice
        or tts_report.get("device") != device
        or tts_report.get("steps") != steps
        or tts_report.get("successful_segments") != tts_report.get("segment_count")
        or len(tts_report.get("segments", [])) != tts_report.get("segment_count")
        or any(segment.get("engine") != "KorvaTTS" for segment in tts_report.get("segments", []))
        or (device == "gpu" and any(
            not providers or providers[0] != "CUDAExecutionProvider"
            for providers in provider_by_session.values()
        ))
        or (device == "gpu" and not provider_by_session)
    ):
        raise RuntimeError("KorvaTTS report is incomplete; refusing to align unverified audio")
    (output_dir / "tts-report.json").write_text(
        json.dumps(tts_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    progress("ALIGNMENT", STAGES["ALIGNMENT"])
    aligned = align_translated_transcript(
        translated_data,
        tts_output_dir=tts_output / "audio",
        output_dir=output_dir / "alignment",
        settings=AudioAlignmentSettings(ffmpeg_path=settings.ffmpeg_path),
        transcript_path=translated_path.resolve(),
        source_media=video,
    )
    alignment_report = aligned.to_dict(
        input_transcript={
            "translated_transcript": str(translated_path.resolve()),
            "source_media": str(video.resolve()),
        }
    )
    (output_dir / "alignment-report.json").write_text(
        json.dumps(alignment_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if aligned.quality_fail_count:
        raise RuntimeError(
            f"QUALITY_FAIL reason=INSUFFICIENT_SPEECH_WINDOW; "
            f"alignment has {aligned.quality_fail_count} failed segment(s)."
        )

    progress("RENDERING", STAGES["RENDERING"])
    result = FFmpegVideoRenderer(RenderSettings(ffmpeg_path=settings.ffmpeg_path)).render(
        video, aligned.timeline_path, output_dir / "dubbed_video.mp4"
    )
    if result.status != "success":
        raise RuntimeError(f"Video rendering failed: {result.error or 'unknown render error'}")
    return output_dir / "dubbed_video.mp4"
