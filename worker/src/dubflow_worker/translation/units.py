from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any


_SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f\u2026][\"'\u2019\u201d\uff09\u300d\u300f\u3011]*$")
_EN_INCOMPLETE_END = re.compile(
    r"(?:\b(?:a|an|the|in|on|at|to|and|or|but|because|which|of|for|with|one|my|your))$",
    re.IGNORECASE,
)
_EN_CONTINUATION_START = re.compile(
    r"^(?:and|but|so|then|because|which|while|when|as|that's where|and using that)\b", re.IGNORECASE
)
_JA_INCOMPLETE_END = re.compile(r"(?:\u3066|\u3063\u3066|\u3051\u3069|\u304b\u3089|\u306e\u3067|\u306e\u306b|\u304c|\u306f|\u3092|\u306b|\u3068|\u305f\u3081|\u306e|\u5168\u90e8)$")
_JA_CONTINUATION_START = re.compile(r"^(?:\u305d\u3057\u3066|\u305d\u308c\u3067|\u307e\u305f|\u305d\u306e|\u306a\u306e\u3067|\u3060\u304b\u3089|\u3051\u308c\u3069|\u3051\u3069)")


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    translation_unit_id: str
    source_segment_ids: tuple[int, ...]
    source_language: str
    speaker_ids: tuple[str, ...]
    source_text: str
    translation_input_text: str
    incomplete_segment_ids: tuple[int, ...]
    review_reasons: tuple[str, ...] = ()


def infer_segment_language(text: str, fallback: str) -> str:
    """Conservatively distinguish Japanese script from predominantly Latin ASR text."""
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return fallback
    japanese = sum(
        "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"
        for char in letters
    )
    latin = sum("LATIN" in unicodedata.name(char, "") for char in letters)
    if japanese and japanese / len(letters) >= 0.2:
        return "ja"
    if latin / len(letters) >= 0.65:
        return "en"
    return fallback


def is_incomplete(text: str, language: str) -> bool:
    value = text.strip().rstrip(",\uff0c\u3001;:\uff1a")
    if not value or _SENTENCE_END.search(value):
        return False
    if language == "ja":
        return bool(_JA_INCOMPLETE_END.search(value))
    return bool(_EN_INCOMPLETE_END.search(value))


def _starts_continuation(text: str, language: str) -> bool:
    value = text.strip()
    if language == "ja":
        return bool(_JA_CONTINUATION_START.search(value))
    return bool(_EN_CONTINUATION_START.search(value))


def _translation_input(group: list[dict[str, Any]], language: str) -> str:
    """Join source segments without adding punctuation or dropping source content."""
    del language
    return " ".join(str(segment.get("text", "")).strip() for segment in group)


def _source_characters(text: str) -> str:
    return "".join(character for character in text if character.isalnum())


def validate_source_coverage(
    segments: list[dict[str, Any]], units: list[TranslationUnit]
) -> dict[str, Any]:
    """Check exact once-only segment assignment and that formatting drops no source characters."""
    expected = [int(segment["id"]) for segment in sorted(
        segments, key=lambda item: (float(item["start"]), int(item["id"]))
    )]
    if len(set(expected)) != len(expected):
        raise ValueError("Translation-unit source coverage failed: input segment IDs are not unique")
    actual = [segment_id for unit in units for segment_id in unit.source_segment_ids]
    counts = Counter(actual)
    duplicated = sorted(segment_id for segment_id, count in counts.items() if count != 1)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    by_id = {int(segment["id"]): segment for segment in segments}
    text_mismatches: list[str] = []
    speaker_boundary_violations: list[str] = []
    for unit in units:
        source_parts = [str(by_id[segment_id].get("text", "")).strip() for segment_id in unit.source_segment_ids]
        expected_source = " ".join(source_parts)
        if expected_source != unit.source_text:
            text_mismatches.append(unit.translation_unit_id)
        if unit.translation_input_text != expected_source:
            text_mismatches.append(unit.translation_unit_id)
        speaker_values = tuple(dict.fromkeys(
            str(by_id[segment_id]["speaker"])
            for segment_id in unit.source_segment_ids
            if by_id[segment_id].get("speaker") is not None
        ))
        distinct_speakers = {by_id[segment_id].get("speaker") for segment_id in unit.source_segment_ids}
        if speaker_values != unit.speaker_ids or len(distinct_speakers) > 1:
            speaker_boundary_violations.append(unit.translation_unit_id)
    if actual != expected or duplicated or missing or unexpected or text_mismatches or speaker_boundary_violations:
        raise ValueError(
            "Translation-unit source coverage failed: "
            f"missing={missing}, duplicated={duplicated}, unexpected={unexpected}, "
            f"text_mismatches={sorted(set(text_mismatches))}, "
            f"speaker_boundary_violations={speaker_boundary_violations}"
        )
    return {
        "source_segment_count": len(expected),
        "assigned_segment_count": len(actual),
        "each_segment_assigned_exactly_once": True,
        "source_text_characters_preserved": True,
        "speaker_boundaries_respected": True,
        "segment_order_deterministic": actual == expected,
        "missing_segment_ids": [],
        "duplicate_segment_ids": [],
        "text_mismatch_unit_ids": [],
    }


def group_translation_units(
    segments: list[dict[str, Any]],
    *,
    fallback_language: str,
    max_gap_seconds: float = 1.0,
    max_segments: int = 3,
) -> list[TranslationUnit]:
    """Group clear continuations without crossing speaker/language boundaries."""
    if max_gap_seconds < 0 or max_segments < 1:
        raise ValueError("max_gap_seconds must be non-negative and max_segments must be positive")
    ordered = sorted(segments, key=lambda item: (float(item["start"]), int(item["id"])))
    groups: list[list[dict[str, Any]]] = []
    for segment in ordered:
        language = infer_segment_language(str(segment.get("text", "")), fallback_language)
        if not groups:
            groups.append([segment])
            continue
        current = groups[-1]
        previous = current[-1]
        previous_language = infer_segment_language(str(previous.get("text", "")), fallback_language)
        gap = float(segment["start"]) - float(previous["end"])
        can_join = (
            len(current) < max_segments
            and segment.get("speaker") == previous.get("speaker")
            and language == previous_language
            and -0.05 <= gap <= max_gap_seconds
            and not _SENTENCE_END.search(str(previous.get("text", "")).strip())
            and (is_incomplete(str(previous.get("text", "")), language) or _starts_continuation(str(segment.get("text", "")), language))
        )
        if can_join:
            current.append(segment)
        else:
            groups.append([segment])

    units: list[TranslationUnit] = []
    for index, group in enumerate(groups):
        first_text = str(group[0].get("text", ""))
        language = infer_segment_language(first_text, fallback_language)
        incomplete_ids = tuple(
            int(segment["id"])
            for segment in group
            if is_incomplete(str(segment.get("text", "")), language)
        )
        review_reasons: list[str] = []
        last = group[-1]
        last_text = str(last.get("text", ""))
        if is_incomplete(last_text, language):
            current_index = ordered.index(last)
            if current_index + 1 >= len(ordered):
                review_reasons.append(f"Segment {last['id']} ends incomplete with no following source context.")
            else:
                following = ordered[current_index + 1]
                following_language = infer_segment_language(str(following.get("text", "")), fallback_language)
                if following.get("speaker") != last.get("speaker"):
                    review_reasons.append(
                        f"Segment {last['id']} ends incomplete before speaker boundary at segment {following['id']}."
                    )
                elif following_language != language:
                    review_reasons.append(
                        f"Segment {last['id']} ends incomplete before language boundary at segment {following['id']}."
                    )
                elif float(following["start"]) - float(last["end"]) > max_gap_seconds:
                    review_reasons.append(
                        f"Segment {last['id']} ends incomplete; segment {following['id']} is outside the context gap."
                    )
                elif len(group) >= max_segments:
                    review_reasons.append(
                        f"Segment {last['id']} ends incomplete at the translation-unit size limit."
                    )
                else:
                    review_reasons.append(
                        f"Segment {last['id']} has a likely continuation at segment {following['id']} that was not grouped."
                    )
        speaker_ids = tuple(dict.fromkeys(str(segment["speaker"]) for segment in group if segment.get("speaker") is not None))
        source_text = " ".join(str(segment.get("text", "")).strip() for segment in group)
        units.append(
            TranslationUnit(
                translation_unit_id=f"tu-{index:04d}",
                source_segment_ids=tuple(int(segment["id"]) for segment in group),
                source_language=language,
                speaker_ids=speaker_ids,
                source_text=source_text,
                translation_input_text=_translation_input(group, language),
                incomplete_segment_ids=incomplete_ids,
                review_reasons=tuple(review_reasons),
            )
        )
    validate_source_coverage(segments, units)
    return units
