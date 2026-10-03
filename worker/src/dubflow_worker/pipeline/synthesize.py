from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dubflow_worker.models.tts import SynthesizedSegment
from dubflow_worker.tts.base import TTSEngine


def synthesize_translated_transcript(
    transcript: Mapping[str, Any],
    *,
    language: str,
    output_dir: Path,
    engine: TTSEngine,
) -> list[SynthesizedSegment]:
    """Synthesize each translated segment while copying its identity and timing."""
    if not language.strip():
        raise ValueError("TTS language is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    output: list[SynthesizedSegment] = []
    for segment in transcript.get("segments", []):
        segment_id = int(segment["id"])
        speaker = segment.get("speaker")
        target_text = segment.get("target_text")
        common = {
            "id": segment_id,
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "speaker": speaker,
            "source_text": str(segment.get("source_text", "")),
            "target_text": target_text,
        }
        if not isinstance(target_text, str) or not target_text.strip():
            failure = segment.get("error") or "Translated target_text is empty"
            output.append(SynthesizedSegment(**common, result=None, error=failure))
            continue
        try:
            result = engine.synthesize(
                target_text,
                language,
                output_dir / f"segment-{segment_id:04d}.wav",
                speaker=speaker,
            )
            output.append(SynthesizedSegment(**common, result=result))
        except Exception as exc:
            output.append(
                SynthesizedSegment(
                    **common,
                    result=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return output
