from __future__ import annotations

import json

import pytest

from dubflow_worker.benchmark.translation_models import (
    BenchmarkUnit,
    build_review_matrix,
    load_units,
    run_candidate,
    skipped_candidate,
)


def test_load_units_preserves_exact_saved_sources_and_order(tmp_path):
    rows = []
    translated_units = []
    for index in range(7):
        unit_id = f"tu-{index:04d}"
        source = f"exact source {index} 日本語"
        rows.append({"translation_unit_id": unit_id, "source_segment_ids": [index], "complete_source_text": source})
        translated_units.append({"translation_unit_id": unit_id, "source_language": "ja" if index == 1 else "en"})
    comparison = tmp_path / "comparison.json"
    translated = tmp_path / "translated.json"
    comparison.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    translated.write_text(json.dumps({"translation_units": translated_units}), encoding="utf-8")

    units = load_units(comparison, translated)

    assert [unit.unit_id for unit in units] == [f"tu-{index:04d}" for index in range(7)]
    assert [unit.source_text for unit in units] == [row["complete_source_text"] for row in rows]
    assert units[1].source_language == "ja"


def test_load_units_rejects_missing_or_reordered_set(tmp_path):
    comparison = tmp_path / "comparison.json"
    translated = tmp_path / "translated.json"
    rows = [{"translation_unit_id": "tu-0001", "complete_source_text": "source"}]
    comparison.write_text(json.dumps(rows), encoding="utf-8")
    translated.write_text(json.dumps({"translation_units": [{"translation_unit_id": "tu-0001", "source_language": "en"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected the seven ordered IDs"):
        load_units(comparison, translated)


class StubEngine:
    name = "stub"

    def validate_language_pair(self, source_language, target_language):
        if source_language == "ja":
            raise ValueError("unsupported test route")

    def translate_for_dubbing(self, text, source_language, target_language):
        return f"RAW<{text}>"


class StubNaturalizer:
    def naturalize(self, text, *, language, source_context):
        return f"NAT<{text}>"


def test_candidate_continues_after_item_error_and_keeps_raw_separate():
    units = [
        BenchmarkUnit("tu-0000", [0], "exact English", "en", 13),
        BenchmarkUnit("tu-0001", [1], "正確な日本語", "ja", 6),
    ]
    result = run_candidate(
        name="stub", engine_factory=StubEngine, units=units, naturalizer=StubNaturalizer(),
        model="stub-model", device="cpu",
    )

    assert result["status"] == "completed_with_failures"
    assert result["units"][0]["source_text"] == "exact English"
    assert result["units"][0]["raw_translation"] == "RAW<exact English>"
    assert result["units"][0]["naturalized_translation"] == "NAT<RAW<exact English>>"
    assert result["units"][1]["status"] == "failed"
    assert result["units"][1]["raw_translation"] is None


def test_skipped_candidate_and_review_matrix_are_serializable():
    units = [BenchmarkUnit(f"tu-{index:04d}", [index], "source", "en", 6) for index in range(7)]
    skipped = skipped_candidate("LLM", "not configured", units)
    matrix = build_review_matrix([skipped])

    assert len(skipped["units"]) == len(matrix) == 7
    assert all(row["status"] == "skipped" and not row["attempted"] for row in skipped["units"])
    assert all(row["review_required"] for row in matrix)
    assert len(json.dumps({"candidate": skipped, "review_matrix": matrix}, ensure_ascii=False)) > 0
