from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any


EXPECTED_UNIT_IDS = [f"tu-{index:04d}" for index in range(7)]
EXPECTED_STATUSES = {"tu-0000": "D", "tu-0001": "C", "tu-0002": "C", "tu-0003": "C", "tu-0004": "C", "tu-0005": "B", "tu-0006": "C"}
APPROVED_TU_0005 = (
    "Trước tiên, bạn nên học Hiragana, Katakana và một chút Kanji. "
    "Đây là ba hệ chữ viết chính, nên hãy học chúng trong quá trình sử dụng ứng dụng. "
    "Khi tiếp tục học, hãy đặt câu ví dụ và áp dụng những gì mình đang học."
)

REVIEW_NOTES = {
    "tu-0000": "Source segment ends on an incomplete phrase ('in one') at a language boundary. Do not finalize for TTS.",
    "tu-0001": "'Juissancei' is unresolved and may be a proper noun or ASR error. Preserve the source token.",
    "tu-0002": "'ダロノード' is unresolved and the source ends with continuing 'と思ってて' at a speaker boundary. Do not guess or add a continuation.",
    "tu-0003": "'Cuban Japanese' is unclear and tofugood.com must remain exactly as sourced. Do not guess the intended phrase.",
    "tu-0004": "'Good blogs and errors' is unusual and may reflect ASR uncertainty. Do not rewrite 'errors' speculatively.",
    "tu-0005": "Source is complete and semantically clear. Vietnamese was revised for natural spoken delivery without adding information.",
    "tu-0006": "'Bumpal' and the title-like 'Takem's Guide to Learning Japanese' are unverified. Preserve both exactly.",
}

_COPIED_FIELDS = (
    "unit_id", "source_segment_ids", "speaker_ids", "start", "end", "source_text",
    "translation_raw", "translation_final", "source_uncertain",
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_translation_review(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    """Create the fixed 1K.3 human review result without mutating source rows."""
    if [row.get("unit_id") for row in comparison] != EXPECTED_UNIT_IDS:
        raise ValueError("Comparison must contain exactly tu-0000 through tu-0006 in order")

    units = []
    for row in comparison:
        unit_id = row["unit_id"]
        unit = {key: row[key] for key in _COPIED_FIELDS}
        unit.update({
            "review_status": EXPECTED_STATUSES[unit_id],
            "approved_text": APPROVED_TU_0005 if unit_id == "tu-0005" else None,
            "tts_ready": unit_id == "tu-0005",
            "review_note": REVIEW_NOTES[unit_id],
        })
        units.append(unit)
    return {
        "phase": "1K.3",
        "status": "COMPLETED_REVIEW_REQUIRED",
        "source_artifact": "output/real-validation-v6/translation-comparison.json",
        "units": units,
    }


def validate_translation_review(
    comparison: list[dict[str, Any]],
    review: dict[str, Any],
    transcript: dict[str, Any],
    transcript_sha256: str,
) -> dict[str, Any]:
    source_by_id = {segment["id"]: segment for segment in transcript.get("segments", [])}
    units = review.get("units", [])
    expected_source_ids = [segment["id"] for segment in transcript.get("segments", [])]
    assigned_source_ids = [segment_id for unit in units for segment_id in unit["source_segment_ids"]]
    coverage_ok = (Counter(assigned_source_ids) == Counter(expected_source_ids))

    input_by_id = {row["unit_id"]: row for row in comparison}
    source_exact = len(units) == len(comparison) and all(
        unit.get("source_text") == input_by_id.get(unit.get("unit_id"), {}).get("source_text")
        and all(unit.get(field) == input_by_id.get(unit.get("unit_id"), {}).get(field) for field in _COPIED_FIELDS)
        for unit in units
    )
    expected_ids_ok = [unit.get("unit_id") for unit in units] == EXPECTED_UNIT_IDS
    segment_text_matches = all(
        unit["source_text"] == " ".join(source_by_id[sid]["text"] for sid in unit["source_segment_ids"])
        for unit in units
    )
    statuses = Counter(unit["review_status"] for unit in units)
    approved_ok = all(
        (unit["review_status"] in {"A", "B"} and isinstance(unit["approved_text"], str) and bool(unit["approved_text"].strip()) and unit["tts_ready"])
        or (unit["review_status"] in {"C", "D"} and unit["approved_text"] is None and not unit["tts_ready"])
        for unit in units
    )
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    uncertain_tokens = {
        "tu-0001": ["Juissancei"],
        "tu-0002": ["ダロノード", "と思ってて"],
        "tu-0003": ["Cuban Japanese", "tofugood.com"],
        "tu-0004": ["good blogs and errors"],
        "tu-0006": ["Bumpal", "Takem's Guide to Learning Japanese"],
    }
    uncertain_ok = all(
        all(token in unit_by_id[unit_id]["source_text"] for token in tokens)
        and unit_by_id[unit_id]["review_status"] == "C"
        and unit_by_id[unit_id]["approved_text"] is None
        for unit_id, tokens in uncertain_tokens.items()
    )
    if "tofugood.com" in unit_by_id["tu-0003"]["source_text"]:
        uncertain_ok = uncertain_ok and "tofugood.com" in unit_by_id["tu-0003"]["translation_final"]
    if "Bumpal" in unit_by_id["tu-0006"]["source_text"]:
        uncertain_ok = uncertain_ok and all(
            term in unit_by_id["tu-0006"]["translation_final"]
            for term in ("Bumpal", "Takem's Guide to Learning Japanese")
        )
    immutable_ok = expected_ids_ok and source_exact and segment_text_matches and transcript_sha256 != ""
    return {
        "total_units": len(units),
        "status_counts": {status: statuses.get(status, 0) for status in "ABCD"},
        "tts_ready_count": sum(bool(unit["tts_ready"]) for unit in units),
        "blocked_count": sum(not unit["tts_ready"] for unit in units),
        "source_coverage": {
            "transcript_segment_ids": expected_source_ids,
            "assigned_segment_ids": assigned_source_ids,
            "each_segment_assigned_exactly_once": coverage_ok,
            "passed": coverage_ok,
        },
        "source_integrity": {
            "expected_unit_ids_in_order": expected_ids_ok,
            "metadata_and_source_fields_match_comparison": source_exact,
            "unit_source_text_matches_transcript_segments": segment_text_matches,
            "passed": expected_ids_ok and source_exact and segment_text_matches,
        },
        "immutable_source_check": {
            "passed": immutable_ok,
            "transcript_sha256": transcript_sha256,
            "source_texts_match_comparison": source_exact,
            "source_texts_match_transcript": segment_text_matches,
        },
        "approved_text_check": {
            "status_and_tts_rules_passed": approved_ok,
            "only_tu_0005_has_approved_text": [u["unit_id"] for u in units if u["approved_text"] is not None] == ["tu-0005"],
            "tu_0005_approved_text": unit_by_id["tu-0005"]["approved_text"],
            "approved_text_matches_user_review_candidate": unit_by_id["tu-0005"]["approved_text"] == APPROVED_TU_0005,
            "no_new_source_claims_check": "approved text is the supplied reviewed candidate; no other unit received approved text",
            "passed": approved_ok and [u["unit_id"] for u in units if u["approved_text"] is not None] == ["tu-0005"] and unit_by_id["tu-0005"]["approved_text"] == APPROVED_TU_0005,
        },
        "uncertain_source_check": {
            "preserved_source_tokens_by_unit": uncertain_tokens,
            "uncertain_terms_preserved_and_blocked": uncertain_ok,
            "url_preserved_in_translation": "tofugood.com" in unit_by_id["tu-0003"]["translation_final"],
            "passed": uncertain_ok,
        },
    }


def render_translation_review_markdown(review: dict[str, Any], report: dict[str, Any]) -> str:
    lines = [
        "# Phase 1K.3 Translation Review", "", "## Summary", "",
        "| Status | Count |", "|---|---:|",
    ]
    for status in "ABCD":
        labels = {"A": "A — TTS_READY", "B": "B — NEEDS_VI_EDIT", "C": "C — SOURCE_UNCERTAIN", "D": "D — INSUFFICIENT_SOURCE"}
        lines.append(f"| {labels[status]} | {report['status_counts'][status]} |")
    lines.extend([
        f"| TTS-ready units | {report['tts_ready_count']} |",
        f"| Blocked units | {report['blocked_count']} |", "", "## Unit Review", "",
    ])
    labels = {"A": "TTS_READY", "B": "NEEDS_VI_EDIT", "C": "SOURCE_UNCERTAIN", "D": "INSUFFICIENT_SOURCE"}
    for unit in review["units"]:
        approved = unit["approved_text"] if unit["approved_text"] is not None else "None (blocked)"
        lines.extend([
            f"### {unit['unit_id']}", "",
            f"Source: {unit['source_text']}", "",
            f"NLLB: {unit['translation_raw']}", "",
            f"Current final: {unit['translation_final']}", "",
            f"Status: {unit['review_status']} — {labels[unit['review_status']]}", "",
            f"Approved: {approved}", "",
            f"Reason: {unit['review_note']}", "",
        ])
    lines.extend(["## TTS Gate", "", "Allowed into TTS:", "", "- tu-0005", "", "Blocked from TTS:", ""])
    for unit in review["units"]:
        if not unit["tts_ready"]:
            lines.append(f"- {unit['unit_id']}: {unit['review_note']}")
    lines.append("")
    return "\n".join(lines)


def assert_review_valid(report: dict[str, Any]) -> None:
    checks = ("source_coverage", "source_integrity", "immutable_source_check", "approved_text_check", "uncertain_source_check")
    failed = [name for name in checks if not report[name]["passed"]]
    if report["total_units"] != 7 or failed:
        raise ValueError(f"Review validation failed: units={report['total_units']}, failed_checks={failed}")
