from __future__ import annotations

import hashlib
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from dubflow_worker.translation.base import TranslationEngine
from dubflow_worker.translation.glossary import Glossary
from dubflow_worker.translation.naturalization_v3 import ContextAwareVietnameseNaturalizer
from dubflow_worker.translation.units import group_translation_units, validate_source_coverage


def _audit_by_segment(source_audit: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in source_audit.get("items", []):
        for segment_id in item.get("segment_ids", []):
            result[int(segment_id)].append(item)
    return result


def _timing_estimate(text: str, duration: float, rate: float = 4.5) -> dict[str, Any]:
    syllable_like_units = len(re.findall(r"\S+", text))
    allowable_units = max(1, int(duration * rate))
    return {
        "estimated_vietnamese_syllable_units": syllable_like_units,
        "available_duration_seconds": round(duration, 3),
        "screening_rate_units_per_second": rate,
        "screening_capacity_units": allowable_units,
        "screening_only": True,
        "needs_timing_review": syllable_like_units > allowable_units,
    }


def translate_context_units_v6(
    transcript: Mapping[str, Any],
    *,
    engine: TranslationEngine,
    naturalizer: ContextAwareVietnameseNaturalizer,
    glossary: Glossary,
    source_audit: Mapping[str, Any],
    target_language: str = "vi",
) -> dict[str, Any]:
    detected = str(
        transcript.get("detected_language") or transcript.get("asr", {}).get("language") or ""
    ).lower().split("-", 1)[0]
    if not detected or target_language.lower() != "vi":
        raise ValueError("Phase 1K.2 requires a detected source language and Vietnamese target")
    segments = list(transcript.get("segments", []))
    units = group_translation_units(segments, fallback_language=detected)
    coverage = validate_source_coverage(segments, units)
    by_id = {int(row["id"]): row for row in segments}
    uncertainty_by_segment = _audit_by_segment(source_audit)

    load_started = time.perf_counter()
    load = getattr(engine, "load", None)
    if callable(load):
        load()
    model_load_seconds = time.perf_counter() - load_started
    translation_seconds = 0.0
    naturalization_seconds = 0.0
    unit_rows: list[dict[str, Any]] = []
    segment_mapping: list[dict[str, Any]] = []

    for unit in units:
        source_ids = list(unit.source_segment_ids)
        source_text = unit.source_text
        source_segments = [by_id[segment_id] for segment_id in source_ids]
        start = min(float(row["start"]) for row in source_segments)
        end = max(float(row["end"]) for row in source_segments)
        duration = max(0.0, end - start)
        findings = [
            item for segment_id in source_ids for item in uncertainty_by_segment.get(segment_id, [])
        ]
        finding_ids = list(dict.fromkeys(item["item_id"] for item in findings))
        uncertainty_reasons = list(dict.fromkeys(
            f"{item['suspicious_phrase']}: {item['reason_for_suspicion']}" for item in findings
        ))
        model_input, protected = glossary.mask(unit.translation_input_text)
        translation_started = time.perf_counter()
        provider_raw: str | None = None
        translation_raw: str | None = None
        translation_base_for_naturalization: str | None = None
        fallback_translations: list[dict[str, Any]] = []
        coverage_repair_applied = False
        coverage_note: str | None = None
        translation_final: str | None = None
        naturalization_rules: list[str] = []
        naturalization_review: str | None = None
        error: str | None = None
        try:
            engine.validate_language_pair(unit.source_language, target_language)
            provider_raw = engine.translate_for_dubbing(model_input, unit.source_language, target_language)
            if not isinstance(provider_raw, str) or not provider_raw.strip():
                raise ValueError("NLLB returned empty output for non-empty context unit")
            translation_raw = Glossary.restore(provider_raw.strip(), protected)
            translation_base_for_naturalization = translation_raw

            # NLLB occasionally returns only the first sentence of a long context unit.
            # Keep that exact context output, but for a clearly truncated multi-segment
            # unit, translate the clauses as a recovery path and retain both versions.
            source_word_count = len(re.findall(r"\b[\w’'-]+\b", source_text))
            target_word_count = len(re.findall(r"\S+", translation_raw))
            obvious_truncation = (
                len(source_ids) > 1
                and unit.source_language == "en"
                and source_word_count >= 16
                and target_word_count < source_word_count * 0.45
            )
            if obvious_truncation:
                piece_rows: list[dict[str, Any]] = []
                for segment in source_segments:
                    piece_source = str(segment.get("text", "")).strip()
                    piece_input, piece_protected = glossary.mask(piece_source)
                    piece_provider_raw = engine.translate_for_dubbing(
                        piece_input, unit.source_language, target_language
                    )
                    if not isinstance(piece_provider_raw, str) or not piece_provider_raw.strip():
                        raise ValueError(f"NLLB returned empty supplemental output for segment {segment['id']}")
                    piece_translation = Glossary.restore(piece_provider_raw.strip(), piece_protected)
                    piece_rows.append({
                        "source_segment_id": int(segment["id"]),
                        "source_text": piece_source,
                        "translation_raw": piece_translation,
                    })
                fallback_translations = piece_rows
                translation_base_for_naturalization = ". ".join(
                    row["translation_raw"].rstrip(" .!?。！？…") for row in piece_rows
                )
                coverage_repair_applied = True
                coverage_note = (
                    "The complete-context NLLB response was unusually short relative to its source. "
                    "The naturalized output uses supplemental clause translations to retain source coverage; "
                    "review fluency, repetition, and overlap against the complete-context response."
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        unit_translation_seconds = time.perf_counter() - translation_started
        translation_seconds += unit_translation_seconds

        timing = _timing_estimate(translation_base_for_naturalization or translation_raw or "", duration)
        if translation_raw is not None:
            naturalization_started = time.perf_counter()
            naturalized = naturalizer.naturalize_with_report(
                translation_base_for_naturalization or translation_raw,
                language=target_language,
                source_context=source_text,
            )
            unit_naturalization_seconds = time.perf_counter() - naturalization_started
            naturalization_seconds += unit_naturalization_seconds
            translation_final = naturalized.text
            naturalization_rules = list(naturalized.rules_applied)
            naturalization_review = naturalized.review_reason
            missing_terms = [term for term in protected.values() if term.casefold() not in translation_final.casefold()]
            if missing_terms:
                translation_final = translation_base_for_naturalization or translation_raw
                naturalization_review = "Naturalizer changed a protected term; kept raw translation: " + ", ".join(missing_terms)

        review_reasons = list(uncertainty_reasons)
        if unit.review_reasons:
            review_reasons.extend(unit.review_reasons)
        if naturalization_review:
            review_reasons.append(naturalization_review)
        if error:
            review_reasons.append("Translation failed: " + error)
        if coverage_note:
            review_reasons.append(coverage_note)
        # Translation is machine-generated and has no human semantic sign-off in this phase.
        review_reasons.append("Machine translation has not received human bilingual review.")
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        unit_rows.append({
            "unit_id": unit.translation_unit_id,
            "translation_unit_id": unit.translation_unit_id,
            "source_segment_ids": source_ids,
            "speaker_ids": list(unit.speaker_ids),
            "start": start,
            "end": end,
            "source_text": source_text,
            "translation_input_text": unit.translation_input_text,
            "translation_model_input": model_input,
            "source_text_sha256": source_hash,
            "source_language": unit.source_language,
            "target_language": target_language,
            "provider_output_raw": provider_raw,
            "translation_raw": translation_raw,
            "translation_fallback_segment_translations": fallback_translations,
            "translation_base_for_naturalization": translation_base_for_naturalization,
            "coverage_repair_applied": coverage_repair_applied,
            "coverage_note": coverage_note,
            "translation_final": translation_final,
            "naturalizer": naturalizer.name,
            "naturalization_rules_applied": naturalization_rules,
            "translation_seconds": round(unit_translation_seconds, 6),
            "naturalization_seconds": round(
                unit_naturalization_seconds if translation_raw is not None else 0.0, 6
            ),
            "source_uncertain": bool(findings),
            "source_uncertainty_item_ids": finding_ids,
            "uncertainty_reason": "; ".join(uncertainty_reasons) if uncertainty_reasons else None,
                "needs_review": True,
                "review_reasons": list(dict.fromkeys(review_reasons)),
                "needs_timing_review": timing["needs_timing_review"],
                "timing_review": timing,
                "naturalization_review_reason": naturalization_review,
                "incomplete_source_segment_ids": list(unit.incomplete_segment_ids),
                "translation_error": error,
                "translation_unit_runtime_seconds": round(unit_translation_seconds, 6),
                "naturalization_unit_runtime_seconds": round(
                    unit_naturalization_seconds if translation_raw is not None else 0.0, 6
                ),
            })
        for segment in source_segments:
            segment_mapping.append({
                "id": int(segment["id"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "speaker": segment.get("speaker"),
                "source_text": str(segment.get("text", "")),
                "translation_unit_id": unit.translation_unit_id,
            })

    totals = Counter()
    totals["needs_review"] = sum(row["needs_review"] for row in unit_rows)
    totals["source_uncertain"] = sum(row["source_uncertain"] for row in unit_rows)
    totals["needs_timing_review"] = sum(row["needs_timing_review"] for row in unit_rows)
    totals["failed_units"] = sum(row["translation_raw"] is None for row in unit_rows)
    return {
        "engine": engine.name,
        "naturalizer": naturalizer.name,
        "source_language": detected,
        "target_language": target_language,
        "model_load_seconds": round(model_load_seconds, 6),
        "translation_seconds": round(translation_seconds, 6),
        "naturalization_seconds": round(naturalization_seconds, 6),
        "source_coverage": coverage,
        "translation_units": unit_rows,
        "segments": segment_mapping,
        "counts": dict(totals),
        "success": totals["failed_units"] == 0,
    }
