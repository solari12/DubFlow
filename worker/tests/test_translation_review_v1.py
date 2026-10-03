from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dubflow_worker.review.translation_review import (
    APPROVED_TU_0005,
    EXPECTED_UNIT_IDS,
    assert_review_valid,
    build_translation_review,
    render_translation_review_markdown,
    validate_translation_review,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_PATH = REPO_ROOT / "output/real-validation-v6/translation-comparison.json"
TRANSCRIPT_PATH = REPO_ROOT / "output/real-validation-v6/transcript.json"


@pytest.fixture
def artifacts():
    comparison_bytes = COMPARISON_PATH.read_bytes()
    transcript_bytes = TRANSCRIPT_PATH.read_bytes()
    comparison = json.loads(comparison_bytes.decode("utf-8"))
    transcript = json.loads(transcript_bytes.decode("utf-8"))
    review = build_translation_review(comparison)
    report = validate_translation_review(
        comparison,
        review,
        transcript,
        hashlib.sha256(transcript_bytes).hexdigest(),
    )
    return comparison_bytes, transcript_bytes, comparison, transcript, review, report


def test_review_contains_exact_seven_units_and_expected_classifications(artifacts):
    _, _, _, _, review, report = artifacts
    assert [unit["unit_id"] for unit in review["units"]] == EXPECTED_UNIT_IDS
    assert [unit["review_status"] for unit in review["units"]] == ["D", "C", "C", "C", "C", "B", "C"]
    assert report["status_counts"] == {"A": 0, "B": 1, "C": 5, "D": 1}


def test_review_copies_source_and_translation_fields_exactly(artifacts):
    _, _, comparison, _, review, report = artifacts
    for source, reviewed in zip(comparison, review["units"], strict=True):
        for field in ("unit_id", "source_segment_ids", "speaker_ids", "start", "end", "source_text", "translation_raw", "translation_final", "source_uncertain"):
            assert reviewed[field] == source[field]
    assert report["source_integrity"]["passed"]
    assert report["immutable_source_check"]["passed"]


def test_only_clear_b_unit_has_approved_text_and_is_tts_ready(artifacts):
    _, _, _, _, review, report = artifacts
    approved = [unit for unit in review["units"] if unit["approved_text"] is not None]
    assert len(approved) == 1
    assert approved[0]["unit_id"] == "tu-0005"
    assert approved[0]["approved_text"] == APPROVED_TU_0005
    assert approved[0]["tts_ready"] is True
    assert report["approved_text_check"]["passed"]


def test_c_and_d_units_are_blocked_with_null_approved_text(artifacts):
    _, _, _, _, review, report = artifacts
    blocked = [unit for unit in review["units"] if unit["review_status"] in {"C", "D"}]
    assert len(blocked) == 6
    assert all(unit["approved_text"] is None and not unit["tts_ready"] for unit in blocked)
    assert report["blocked_count"] == 6


def test_segment_coverage_has_no_lost_or_duplicated_ids(artifacts):
    _, _, _, transcript, _, report = artifacts
    assert report["source_coverage"]["transcript_segment_ids"] == [segment["id"] for segment in transcript["segments"]]
    assert report["source_coverage"]["each_segment_assigned_exactly_once"]


def test_timestamps_and_speaker_metadata_are_preserved(artifacts):
    _, _, comparison, _, review, _ = artifacts
    for source, reviewed in zip(comparison, review["units"], strict=True):
        assert reviewed["start"] == source["start"]
        assert reviewed["end"] == source["end"]
        assert reviewed["speaker_ids"] == source["speaker_ids"]


def test_uncertain_source_tokens_and_url_are_preserved(artifacts):
    _, _, _, _, review, report = artifacts
    by_id = {unit["unit_id"]: unit for unit in review["units"]}
    tokens = {
        "tu-0001": "Juissancei",
        "tu-0002": "ダロノード",
        "tu-0003": "Cuban Japanese",
        "tu-0004": "good blogs and errors",
        "tu-0006": "Bumpal",
    }
    assert all(token in by_id[unit_id]["source_text"] for unit_id, token in tokens.items())
    assert "tofugood.com" in by_id["tu-0003"]["source_text"]
    assert "tofugood.com" in by_id["tu-0003"]["translation_final"]
    assert "Takem's Guide to Learning Japanese" in by_id["tu-0006"]["source_text"]
    assert "Takem's Guide to Learning Japanese" in by_id["tu-0006"]["translation_final"]
    assert report["uncertain_source_check"]["passed"]


def test_review_validation_rejects_modified_source_text(artifacts):
    _, transcript_bytes, comparison, transcript, review, _ = artifacts
    review["units"][0]["source_text"] += " invented"
    report = validate_translation_review(comparison, review, transcript, hashlib.sha256(transcript_bytes).hexdigest())
    assert not report["source_integrity"]["passed"]
    assert not report["immutable_source_check"]["passed"]
    with pytest.raises(ValueError, match="source_integrity"):
        assert_review_valid(report)


def test_input_artifacts_are_not_modified_by_review_build(artifacts):
    comparison_before, transcript_before, comparison, _, _, _ = artifacts
    build_translation_review(comparison)
    assert COMPARISON_PATH.read_bytes() == comparison_before
    assert TRANSCRIPT_PATH.read_bytes() == transcript_before


def test_markdown_includes_summary_unit_review_and_explicit_tts_gate(artifacts):
    _, _, _, _, review, report = artifacts
    markdown = render_translation_review_markdown(review, report)
    assert "# Phase 1K.3 Translation Review" in markdown
    assert "## Summary" in markdown and "## Unit Review" in markdown and "## TTS Gate" in markdown
    assert "Allowed into TTS:" in markdown and "- tu-0005" in markdown
    assert "tu-0000:" in markdown and "tu-0006:" in markdown


def test_builder_rejects_unexpected_unit_ids(artifacts):
    _, _, comparison, _, _, _ = artifacts
    with pytest.raises(ValueError, match="exactly tu-0000"):
        build_translation_review(comparison[:-1])

