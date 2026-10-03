from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from dubflow_worker.translation.base import TranslationEngine
from dubflow_worker.translation.glossary import Glossary
from dubflow_worker.translation.naturalization import NaturalizationProvider
from dubflow_worker.translation.units import group_translation_units, validate_source_coverage


def _translation_integrity(unit_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "translation_unit_id": unit_row["translation_unit_id"],
        "source_segment_ids": unit_row["source_segment_ids"],
        "source_character_count": unit_row["source_character_count"],
        "source_text_sha256": unit_row["source_text_sha256"],
        "source_text_preserved_in_model_input": unit_row["source_text_preserved_in_model_input"],
    }


def translate_contextual_transcript(
    transcript: Mapping[str, Any],
    *,
    engine: TranslationEngine,
    naturalizer: NaturalizationProvider,
    target_language: str,
    glossary: Glossary,
    baseline_segments: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    detected = str(
        transcript.get("detected_language")
        or transcript.get("asr", {}).get("language")
        or transcript.get("source", {}).get("language")
        or ""
    ).lower().split("-", 1)[0]
    if not detected or not target_language.strip():
        raise ValueError("Source and target language are required")
    segments = list(transcript.get("segments", []))
    units = group_translation_units(segments, fallback_language=detected)
    source_coverage = validate_source_coverage(segments, units)
    unit_rows: list[dict[str, Any]] = []
    segment_rows: dict[int, dict[str, Any]] = {}
    by_id = {int(segment["id"]): segment for segment in segments}

    for unit in units:
        engine.validate_language_pair(unit.source_language, target_language)
        masked, protected_terms = glossary.mask(unit.translation_input_text)
        try:
            # Always send the complete canonical source to NLLB first.
            context_literal = engine.translate_for_dubbing(masked, unit.source_language, target_language)
            if not isinstance(context_literal, str) or not context_literal.strip():
                raise ValueError("Translation engine returned empty output for non-empty context")
            context_literal = Glossary.restore(context_literal, protected_terms)

            # For complete ASR segments grouped by discourse context, compose translations
            # made from the original source segments. This retains each clause when a long
            # NLLB context decode drops a later clause. Incomplete fragments stay contextual.
            segment_literals: list[str] = []
            use_segment_fallback = len(unit.source_segment_ids) > 1 and not unit.incomplete_segment_ids
            if use_segment_fallback:
                for segment_id in unit.source_segment_ids:
                    source_piece = str(by_id[segment_id].get("text", "")).strip()
                    piece_masked, piece_terms = glossary.mask(source_piece)
                    piece = engine.translate_for_dubbing(
                        piece_masked, unit.source_language, target_language
                    )
                    if not isinstance(piece, str) or not piece.strip():
                        raise ValueError(
                            f"Translation engine returned empty output for source segment {segment_id}"
                        )
                    segment_literals.append(Glossary.restore(piece, piece_terms).strip())
                literal = ". ".join(
                    piece.rstrip(" .!?\u3002\uff01\uff1f\u2026")
                    for piece in segment_literals
                )
            else:
                literal = context_literal

            naturalized_candidate = naturalizer.naturalize(
                literal,
                language=target_language,
                source_context=unit.source_text,
            )
            missing_protected_terms = [
                term for term in protected_terms.values() if term not in naturalized_candidate
            ]
            naturalization_review = naturalizer.review_reason(
                literal,
                naturalized_text=naturalized_candidate,
                language=target_language,
            )
            if missing_protected_terms:
                naturalization_review = (
                    "Naturalization changed protected term spelling: "
                    + ", ".join(missing_protected_terms)
                    + "."
                )
            configured_term_review = [
                term for term in glossary.review_required_terms
                if term.casefold() in literal.casefold()
            ]
            review_reasons = list(unit.review_reasons)
            if naturalization_review:
                review_reasons.append(naturalization_review)
            if configured_term_review:
                review_reasons.append(
                    "Verify preserved ASR-sensitive name(s): " + ", ".join(configured_term_review) + "."
                )
            final = literal if naturalization_review else naturalized_candidate
            naturalization_applied = final != literal
            if use_segment_fallback:
                review_reasons.append(
                    "A long-context translation was composed from source-segment translations to retain every source clause; review fluency and overlap."
                )
            if naturalization_applied:
                review_reasons.append(
                    "Deterministic naturalization is heuristic and has not received human semantic review."
                )
            error = None
        except Exception as exc:
            literal = None
            context_literal = None
            segment_literals = []
            use_segment_fallback = False
            final = None
            naturalization_applied = False
            review_reasons = list(unit.review_reasons)
            error = f"{type(exc).__name__}: {exc}"

        source_hash = hashlib.sha256(unit.source_text.encode("utf-8")).hexdigest()
        unit_row = {
            "translation_unit_id": unit.translation_unit_id,
            "source_segment_ids": list(unit.source_segment_ids),
            "source_text": unit.source_text,
            "translation_input_text": unit.translation_input_text,
            "source_language": unit.source_language,
            "speaker_ids": list(unit.speaker_ids),
            "target_language": target_language,
            "source_character_count": len(unit.source_text),
            "source_text_sha256": source_hash,
            "source_text_preserved_in_model_input": source_coverage["source_text_characters_preserved"],
            "literal_translation": literal,
            "context_translation": context_literal,
            "segment_literal_translations": segment_literals,
            "translation_method": "source_segment_composition" if use_segment_fallback else "complete_unit",
            "final_translation": final,
            "naturalization_applied": naturalization_applied,
            "glossary_terms": list(protected_terms.values()),
            "translation_provider": engine.name,
            "naturalization_provider": naturalizer.name,
            "translation_error": error,
            "review_required": bool(review_reasons or error or not naturalization_applied),
            "review_reason": "; ".join(review_reasons) if review_reasons else (
                "No naturalization was applied; verify literal translation." if not naturalization_applied else error
            ),
            "incomplete_segment_ids": list(unit.incomplete_segment_ids),
        }
        unit_rows.append(unit_row)

        for segment_id in unit.source_segment_ids:
            segment = by_id[segment_id]
            segment_rows[segment_id] = {
                "id": segment_id,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "speaker": segment.get("speaker"),
                "translation_unit_id": unit.translation_unit_id,
            }

    terms_preserved = sorted(
        {term for row in unit_rows for term in row["glossary_terms"]}, key=str.casefold
    )
    translated_units = sum(row["literal_translation"] is not None for row in unit_rows)
    failed_units = len(unit_rows) - translated_units
    review_required_units = [row["translation_unit_id"] for row in unit_rows if row["review_required"]]
    incomplete_unresolved = sorted(
        segment_id
        for unit in units
        for segment_id in unit.incomplete_segment_ids
        if segment_id in unit.source_segment_ids[-1:]
    )
    contextual_segments = sorted(
        segment_id for unit in units if len(unit.source_segment_ids) > 1 for segment_id in unit.source_segment_ids
    )
    translated = {
        "version": "3.0",
        "success": failed_units == 0,
        "translation": {
            "engine": engine.name,
            "detected_language": detected,
            "translation_source_language": detected,
            "translation_target_language": target_language,
            "naturalization_provider": naturalizer.name,
            "glossary_terms": list(glossary.terms),
            "segment_mapping": "segments contain timing/speaker/unit references only; canonical translations exist once in translation_units",
        },
        "source_coverage": source_coverage,
        "translation_units": unit_rows,
        "segments": [segment_rows[int(segment["id"])] for segment in segments],
        "quality": {
            "input_segment_count": len(segments),
            "translation_unit_count": len(unit_rows),
            "translated_segment_count": sum(len(row["source_segment_ids"]) for row in unit_rows if row["literal_translation"] is not None),
            "mapped_source_segment_count": len(segment_rows),
            "failed_unit_count": failed_units,
            "naturalized_unit_count": sum(row["naturalization_applied"] for row in unit_rows),
            "review_required_unit_count": len(review_required_units),
            "review_required_units": review_required_units,
            "glossary_terms_preserved": terms_preserved,
            "contextual_segment_ids": contextual_segments,
            "unresolved_incomplete_segments": incomplete_unresolved,
            "subjective_human_review_required": True,
        },
    }
    return translated
