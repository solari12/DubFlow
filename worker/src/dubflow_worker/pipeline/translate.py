from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dubflow_worker.models.translation import TranslatedSegment, TranslatedTranscript
from dubflow_worker.translation.base import TranslationEngine


def _normalize_language(value: str) -> str:
    return value.strip().lower().replace("_", "-").split("-", 1)[0]


def detected_source_language(transcript: Mapping[str, Any]) -> str | None:
    """Read detected ASR language from the Phase 1A and speaker transcript shapes."""
    value = transcript.get("detected_language")
    if value is None and isinstance(transcript.get("asr"), Mapping):
        value = transcript["asr"].get("language")
    if value is None and isinstance(transcript.get("source"), Mapping):
        value = transcript["source"].get("language")
    if not isinstance(value, str) or not value.strip():
        return None
    return _normalize_language(value)


def translate_speaker_transcript(
    speaker_transcript: Mapping[str, Any],
    *,
    target_language: str,
    engine: TranslationEngine,
    source_language: str | None = None,
    detected_language: str | None = None,
) -> TranslatedTranscript:
    """Translate speech using ASR's detected source unless the caller overrides it."""
    detected = detected_source_language(speaker_transcript)
    if detected is None and detected_language and detected_language.strip():
        detected = _normalize_language(detected_language)
    explicit = _normalize_language(source_language) if source_language and source_language.strip() else None
    resolved_source = explicit or detected
    if not resolved_source or not target_language.strip():
        raise ValueError("Detected or explicitly configured source language and target language are required")
    resolved_target = _normalize_language(target_language)

    # Preflight outside per-segment failure isolation. Unsupported language pairs
    # fail the stage instead of becoming empty translations or using the wrong pair.
    engine.validate_language_pair(resolved_source, resolved_target)

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
            target_text = engine.translate_for_dubbing(source_text, resolved_source, resolved_target)
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
        source_language=resolved_source,
        target_language=resolved_target,
        segments=segments,
        detected_language=detected,
        source_language_overridden=explicit is not None and explicit != detected,
    )
