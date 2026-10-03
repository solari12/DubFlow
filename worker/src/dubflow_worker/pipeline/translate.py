from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dubflow_worker.models.translation import TranslatedSegment, TranslatedTranscript
from dubflow_worker.translation.base import TranslationEngine


def translate_speaker_transcript(
    speaker_transcript: Mapping[str, Any],
    *,
    source_language: str,
    target_language: str,
    engine: TranslationEngine,
) -> TranslatedTranscript:
    """Translate each speaker-aware segment while retaining its identity and timing."""
    if not source_language.strip() or not target_language.strip():
        raise ValueError("Source and target languages are required")
    segments: list[TranslatedSegment] = []
    for segment in speaker_transcript.get("segments", []):
        source_text = str(segment.get("text", ""))
        if not source_text.strip():
            segments.append(
                TranslatedSegment(
                    id=int(segment["id"]),
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    speaker=segment.get("speaker"),
                    source_text=source_text,
                    target_text="",
                )
            )
            continue
        try:
            target_text = engine.translate(source_text, source_language, target_language)
            if not isinstance(target_text, str) or not target_text.strip():
                raise ValueError("Translation engine returned empty output for non-empty text")
            segments.append(
                TranslatedSegment(
                    id=int(segment["id"]),
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    speaker=segment.get("speaker"),
                    source_text=source_text,
                    target_text=target_text,
                )
            )
        except Exception as exc:
            segments.append(
                TranslatedSegment(
                    id=int(segment["id"]),
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    speaker=segment.get("speaker"),
                    source_text=source_text,
                    target_text=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return TranslatedTranscript(
        engine=engine.name,
        source_language=source_language,
        target_language=target_language,
        segments=segments,
    )
