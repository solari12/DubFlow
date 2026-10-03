from __future__ import annotations

from dataclasses import replace

import pytest

from dubflow_worker.pipeline.translate_quality import translate_contextual_transcript
from dubflow_worker.translation.glossary import Glossary
from dubflow_worker.translation.naturalization import DeterministicVietnameseNaturalizer
from dubflow_worker.translation.units import group_translation_units, validate_source_coverage


class RecordingEngine:
    name = "recording-engine"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def validate_language_pair(self, source_language: str, target_language: str) -> None:
        assert source_language in {"en", "ja"}
        assert target_language == "vi"

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        self.calls.append((text, source_language, target_language))
        return f"VI<{text}>"

    def translate_for_dubbing(self, text: str, source_language: str, target_language: str) -> str:
        return self.translate(text, source_language, target_language)


def _segments(*items: tuple[int, str, float, float, str | None]) -> list[dict]:
    return [
        {"id": segment_id, "text": text, "start": start, "end": end, "speaker": speaker}
        for segment_id, text, start, end, speaker in items
    ]


def test_units_record_complete_source_once_in_deterministic_segment_order() -> None:
    segments = _segments(
        (8, "continuation.", 1.0, 2.0, "S0"),
        (7, "I started learning in", 0.0, 1.0, "S0"),
    )
    units = group_translation_units(segments, fallback_language="en")
    assert len(units) == 1
    assert units[0].translation_unit_id == "tu-0000"
    assert units[0].source_segment_ids == (7, 8)
    assert units[0].source_text == "I started learning in continuation."
    assert units[0].source_language == "en"
    assert units[0].speaker_ids == ("S0",)
    assert validate_source_coverage(segments, units)["each_segment_assigned_exactly_once"]


def test_known_incomplete_english_and_japanese_fragments_are_not_standalone_when_safe() -> None:
    english = _segments(
        (0, "and let me teach you guys how to do it exactly in one", 0.0, 1.0, "S0"),
        (1, "more lesson.", 1.0, 2.0, "S0"),
    )
    ja = _segments(
        (2, "\u3068\u308a\u3042\u3048\u305a\u30a2\u30d7\u30ea\u3092\u5168\u90e8", 0.0, 1.0, "S0"),
        (3, "\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9\u3057\u3066\u3001\u52c9\u5f37\u3092\u59cb\u3081\u305f\u3002", 1.0, 2.0, "S0"),
    )
    en_units = group_translation_units(english, fallback_language="en")
    ja_units = group_translation_units(ja, fallback_language="ja")
    assert en_units[0].source_segment_ids == (0, 1)
    assert "exactly in one more lesson" in en_units[0].translation_input_text
    assert ja_units[0].source_segment_ids == (2, 3)
    assert "\u5168\u90e8 \u30c0\u30a6\u30f3" in ja_units[0].translation_input_text


def test_do_not_join_adjacent_complete_or_unrelated_speech() -> None:
    segments = _segments(
        (0, "I finished the lesson.", 0.0, 1.0, "S0"),
        (1, "So pick a different book.", 1.0, 2.0, "S0"),
    )
    assert [u.source_segment_ids for u in group_translation_units(segments, fallback_language="en")] == [(0,), (1,)]


def test_speaker_and_language_boundaries_are_hard_grouping_boundaries() -> None:
    speaker = _segments(
        (0, "I wanted to", 0.0, 1.0, "S0"),
        (1, "continue.", 1.0, 2.0, "S1"),
    )
    language = _segments(
        (0, "I studied Japanese in", 0.0, 1.0, "S0"),
        (1, "\u65e5\u672c\u8a9e\u3092\u52c9\u5f37\u3057\u305f\u3002", 1.0, 2.0, "S0"),
    )
    assert [u.source_segment_ids for u in group_translation_units(speaker, fallback_language="en")] == [(0,), (1,)]
    assert [u.source_language for u in group_translation_units(language, fallback_language="ja")] == ["en", "ja"]


def test_source_coverage_rejects_dropped_or_duplicate_segments() -> None:
    segments = _segments(
        (0, "First.", 0.0, 1.0, "S0"),
        (1, "Second.", 1.0, 2.0, "S0"),
    )
    units = group_translation_units(segments, fallback_language="en")
    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        validate_source_coverage(segments, [units[0]])
    duplicated = replace(units[0], source_segment_ids=(0, 0))
    with pytest.raises(ValueError, match=r"duplicated=\[0\]"):
        validate_source_coverage(segments, [duplicated, units[1]])


def test_complete_unit_is_sent_to_engine_and_segment_fallback_retains_every_clause() -> None:
    transcript = {"detected_language": "en", "segments": _segments(
        (5, "The first thing is learn Hiragana", 0.0, 1.0, "S0"),
        (6, "So these are the three writing systems", 1.0, 2.0, "S0"),
        (7, "And keep making example sentences", 2.0, 3.0, "S0"),
    )}
    engine = RecordingEngine()
    result = translate_contextual_transcript(
        transcript, engine=engine, naturalizer=DeterministicVietnameseNaturalizer(),
        target_language="vi", glossary=Glossary(()),
    )
    unit = result["translation_units"][0]
    assert engine.calls[0][0] == unit["source_text"]
    assert unit["source_segment_ids"] == [5, 6, 7]
    assert len(unit["segment_literal_translations"]) == 3
    assert unit["literal_translation"] == ". ".join(piece.rstrip(" .!?\u3002\uff01\uff1f\u2026") for piece in unit["segment_literal_translations"])
    assert unit["translation_method"] == "source_segment_composition"
    assert unit["review_required"] is True


def test_segment_rows_are_references_and_never_repeat_canonical_translation() -> None:
    transcript = {"detected_language": "en", "segments": _segments(
        (4, "I started learning Japanese in", 0.0, 1.0, "S0"),
        (5, "college.", 1.0, 2.0, "S0"),
    )}
    result = translate_contextual_transcript(
        transcript, engine=RecordingEngine(), naturalizer=DeterministicVietnameseNaturalizer(),
        target_language="vi", glossary=Glossary(()),
    )
    assert len(result["translation_units"]) == 1
    assert [row["translation_unit_id"] for row in result["segments"]] == ["tu-0000", "tu-0000"]
    assert all("final_translation" not in row and "literal_translation" not in row for row in result["segments"])


def test_naturalizer_rewrites_known_malformed_vietnamese_and_preserves_all_clauses() -> None:
    naturalizer = DeterministicVietnameseNaturalizer()
    source = (
        "v\u00e0 s\u1edbm th\u00f4i b\u1ea1n s\u1ebd th\u1ea5y Bumpal, \u0111\u00f3 l\u00e0 ng\u1eef ph\u00e1p, l\u00e0 m\u1ed9t ch\u00fat nh\u1ea7m l\u1eabn. "
        "\u0110\u00f3 l\u00e0 n\u01a1i t\u00f4i khuy\u00ean b\u1ea1n n\u00ean tri\u1ec3n khai Takem's Guide, \u0111\u00f3 l\u00e0 ngu\u1ed3n mi\u1ec5n ph\u00ed n\u00e0y "
        "v\u00e0 s\u1eed d\u1ee5ng n\u00f3, h\u1ecdc ng\u1eef ph\u00e1p v\u00e0 c\u00e1c nguy\u00ean t\u1eafc."
    )
    final = naturalizer.naturalize(source, language="vi", source_context="")
    assert "R\u1ed3i b\u1ea1n s\u1ebd th\u1ea5y Bumpal" in final
    assert "h\u01a1i kh\u00f3 hi\u1ec3u" in final
    assert "tham kh\u1ea3o Takem's Guide" in final
    assert "\u0111\u00e2y l\u00e0 t\u00e0i li\u1ec7u mi\u1ec5n ph\u00ed" in final
    assert "H\u00e3y d\u00f9ng t\u00e0i li\u1ec7u \u0111\u00f3 \u0111\u1ec3 h\u1ecdc" in final
    assert "c\u00e1c nguy\u00ean t\u1eafc" in final
    assert ".;" not in final
    app_clause = naturalizer.naturalize(
        "n\u00ean h\u00e3y h\u1ecdc r\u1eb1ng khi b\u1ea1n s\u1eed d\u1ee5ng \u1ee9ng d\u1ee5ng",
        language="vi", source_context="learn that as you use an app",
    )
    assert app_clause == "H\u00e3y h\u1ecdc d\u1ea7n qua \u1ee9ng d\u1ee5ng"


def test_unsafe_or_unchanged_naturalization_preserves_literal_and_requires_review() -> None:
    naturalizer = DeterministicVietnameseNaturalizer()
    bad = "H\u1ecd c\u00f3 nhi\u1ec1u blog v\u00e0 l\u1ed7i t\u1ed1t."
    transcript = {"detected_language": "en", "segments": _segments((0, "They have blogs.", 0, 1, "S0"))}
    class BadEngine(RecordingEngine):
        def translate(self, text: str, source_language: str, target_language: str) -> str:
            self.calls.append((text, source_language, target_language))
            return bad
    result = translate_contextual_transcript(
        transcript, engine=BadEngine(), naturalizer=naturalizer,
        target_language="vi", glossary=Glossary(()),
    )
    unit = result["translation_units"][0]
    assert unit["final_translation"] == bad
    assert unit["naturalization_applied"] is False
    assert unit["review_required"] is True
    assert "Ambiguous phrase" in unit["review_reason"]


def test_unresolved_incomplete_source_keeps_literal_and_is_flagged() -> None:
    transcript = {"detected_language": "en", "segments": _segments(
        (0, "I have been studying Japanese for one", 0.0, 2.0, "S0"),
        (1, "\u65e5\u672c\u8a9e\u3092\u52c9\u5f37\u3057\u305f\u3002", 2.0, 3.0, "S0"),
    )}
    result = translate_contextual_transcript(
        transcript, engine=RecordingEngine(), naturalizer=DeterministicVietnameseNaturalizer(),
        target_language="vi", glossary=Glossary(()),
    )
    unit = result["translation_units"][0]
    assert unit["source_segment_ids"] == [0]
    assert unit["final_translation"] == unit["literal_translation"]
    assert unit["review_required"] is True
    assert unit["review_reason"]
    assert result["source_coverage"]["each_segment_assigned_exactly_once"] is True


def test_naturalization_can_punctuate_technical_terms_without_changing_their_spelling() -> None:
    transcript = {"detected_language": "en", "segments": _segments(
        (0, "Learn Hiragana Katakana.", 0.0, 1.0, "S0"),
    )}
    result = translate_contextual_transcript(
        transcript, engine=RecordingEngine(), naturalizer=DeterministicVietnameseNaturalizer(),
        target_language="vi", glossary=Glossary(("Hiragana", "Katakana")),
    )
    unit = result["translation_units"][0]
    assert "Hiragana, Katakana" in unit["final_translation"]
    assert "Hiragana" in unit["final_translation"] and "Katakana" in unit["final_translation"]


def test_incomplete_english_unit_can_be_naturalized_without_inventing_its_ending() -> None:
    naturalizer = DeterministicVietnameseNaturalizer()
    source = (
        "T\u00f4i \u0111\u00e3 h\u1ecdc ti\u1ebfng Nh\u1eadt trong b\u1ea3y n\u0103m, "
        "v\u00e0 h\u00e3y \u0111\u1ec3 t\u00f4i d\u1ea1y c\u00e1c b\u1ea1n l\u00e0m th\u1ebf n\u00e0o \u0111\u1ec3 l\u00e0m n\u00f3 ch\u00ednh x\u00e1c trong m\u1ed9t"
    )
    final = naturalizer.naturalize(source, language="vi", source_context="")
    assert "ch\u1ec9 cho c\u00e1c b\u1ea1n c\u00e1ch l\u00e0m ch\u00ednh x\u00e1c trong m\u1ed9t" in final
    assert final.endswith("trong m\u1ed9t")
