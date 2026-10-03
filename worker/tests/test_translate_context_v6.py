from __future__ import annotations

from dubflow_worker.pipeline.translate_v6 import translate_context_units_v6
from dubflow_worker.translation.glossary import Glossary
from dubflow_worker.translation.naturalization_v3 import ContextAwareVietnameseNaturalizer
from dubflow_worker.translation.units import group_translation_units, validate_source_coverage


class RecordingEngine:
    name = "test-nllb"
    device = "cpu"

    def __init__(self, output: str | None = None):
        self.calls = []
        self.output = output
        self.loaded = False

    def load(self):
        self.loaded = True

    def validate_language_pair(self, source_language, target_language):
        assert source_language in {"en", "ja"}
        assert target_language == "vi"

    def translate_for_dubbing(self, text, source_language, target_language):
        self.calls.append((text, source_language, target_language))
        return self.output if self.output is not None else f"Bản dịch {text}"


def _segments(*rows):
    return [
        {"id": item[0], "text": item[1], "start": item[2], "end": item[3], "speaker": item[4]}
        for item in rows
    ]


def _audit(*items):
    return {"items": list(items)}


def _run(transcript, engine=None, audit=None, glossary=None):
    return translate_context_units_v6(
        transcript,
        engine=engine or RecordingEngine(),
        naturalizer=ContextAwareVietnameseNaturalizer(),
        glossary=glossary or Glossary(()),
        source_audit=audit or _audit(),
    )


def test_context_unit_construction_groups_consecutive_same_speaker_thoughts():
    units = group_translation_units(_segments(
        (5, "The first thing is learn Hiragana", 0.0, 1.0, "S0"),
        (6, "So these are writing systems", 1.0, 2.0, "S0"),
        (7, "They have blogs.", 2.0, 3.0, "S0"),
    ), fallback_language="en")
    assert [unit.source_segment_ids for unit in units] == [(5, 6), (7,)]


def test_source_coverage_is_complete_and_segment_mapping_has_no_duplicates():
    source = _segments(
        (0, "I studied Japanese for years", 0.0, 1.0, "S0"),
        (1, "They recommend a site", 1.0, 2.0, "S1"),
    )
    result = _run({"detected_language": "en", "segments": source})
    unit_ids = [sid for unit in result["translation_units"] for sid in unit["source_segment_ids"]]
    assert result["source_coverage"]["each_segment_assigned_exactly_once"]
    assert unit_ids == [0, 1]
    assert len(unit_ids) == len(set(unit_ids)) == len(source)


def test_coverage_validator_rejects_omitted_and_duplicated_segments():
    source = _segments(
        (0, "First complete sentence.", 0.0, 1.0, "S0"),
        (1, "Second complete sentence.", 1.0, 2.0, "S0"),
    )
    units = group_translation_units(source, fallback_language="en")
    try:
        validate_source_coverage(source, units[:-1])
    except ValueError as exc:
        assert "missing=[1]" in str(exc)
    else:
        raise AssertionError("coverage validator accepted an omitted source segment")
    try:
        validate_source_coverage(source, [units[0], units[0]])
    except ValueError as exc:
        assert "duplicated=[0]" in str(exc)
    else:
        raise AssertionError("coverage validator accepted a duplicated source segment")


def test_speaker_boundary_is_never_merged():
    units = group_translation_units(_segments(
        (0, "I started learning in", 0.0, 1.0, "S0"),
        (1, "college.", 1.0, 2.0, "S1"),
    ), fallback_language="en")
    assert [unit.source_segment_ids for unit in units] == [(0,), (1,)]
    assert [unit.speaker_ids for unit in units] == [("S0",), ("S1",)]


def test_uncertain_source_terms_are_preserved_and_flag_review():
    source = "They recommend Cuban Japanese at tofugood.com"
    transcript = {"detected_language": "en", "segments": _segments((3, source, 0, 4, "S1"))}
    audit = _audit(
        {"item_id": "cuban", "segment_ids": [3], "suspicious_phrase": "Cuban Japanese", "reason_for_suspicion": "unclear"},
        {"item_id": "site", "segment_ids": [3], "suspicious_phrase": "tofugood.com", "reason_for_suspicion": "verify spelling"},
    )
    glossary = Glossary(("Cuban Japanese", "tofugood.com"))
    engine = RecordingEngine()
    result = _run(transcript, engine=engine, audit=audit, glossary=glossary)
    unit = result["translation_units"][0]
    assert unit["source_text"] == source
    assert unit["source_uncertain"] and unit["needs_review"]
    assert "Cuban Japanese" not in engine.calls[0][0]
    assert "Cuban Japanese" in unit["translation_raw"]
    assert "tofugood.com" in unit["translation_raw"]


def test_url_survives_context_translation_and_naturalization():
    transcript = {"detected_language": "en", "segments": _segments(
        (0, "I recommend tofugood.com.", 0, 2, "S0"),
    )}
    result = _run(
        transcript,
        engine=RecordingEngine("Tôi rất khuyên bạn nên ZXQTERM0000ZXQ."),
        glossary=Glossary(("tofugood.com",)),
    )
    unit = result["translation_units"][0]
    assert "tofugood.com" in unit["translation_raw"]
    assert "tofugood.com" in unit["translation_final"]


def test_raw_provider_output_is_retained_and_naturalized_output_is_separate():
    transcript = {"detected_language": "en", "segments": _segments(
        (0, "Learn Hiragana Katakana.", 0, 3, "S0"),
    )}
    raw = "Điều đầu tiên bạn muốn làm là học ZXQTERM0000ZXQ ZXQTERM0001ZXQ."
    result = _run(transcript, engine=RecordingEngine(raw), glossary=Glossary(("Hiragana", "Katakana")))
    unit = result["translation_units"][0]
    assert unit["provider_output_raw"] == raw
    assert unit["translation_raw"] == "Điều đầu tiên bạn muốn làm là học Hiragana Katakana."
    assert unit["translation_final"] == "Điều đầu tiên bạn muốn làm là học Hiragana, Katakana."
    assert unit["naturalization_rules_applied"]


def test_unit_to_segment_mapping_is_reversible_and_keeps_source_ids():
    transcript = {"detected_language": "en", "segments": _segments(
        (5, "Learn Hiragana", 0, 1, "S0"),
        (6, "and Katakana", 1, 2, "S0"),
    )}
    result = _run(transcript)
    unit = result["translation_units"][0]
    by_segment = {row["id"]: row for row in result["segments"]}
    assert unit["source_segment_ids"] == [5, 6]
    assert all(by_segment[sid]["translation_unit_id"] == unit["unit_id"] for sid in unit["source_segment_ids"])
    assert " ".join(by_segment[sid]["source_text"] for sid in unit["source_segment_ids"]) == unit["source_text"]


def test_incomplete_source_can_join_adjacent_same_speaker_context():
    units = group_translation_units(_segments(
        (0, "I started learning Japanese in", 0.0, 1.0, "S0"),
        (1, "college.", 1.0, 2.0, "S0"),
    ), fallback_language="en")
    assert len(units) == 1
    assert units[0].source_segment_ids == (0, 1)
    assert units[0].source_text == "I started learning Japanese in college."


def test_no_fabricated_correction_enters_canonical_source_text():
    original = "Soon you will find Bumpal and read about Takem's Guide to Learning Japanese"
    transcript = {"detected_language": "en", "segments": _segments((8, original, 0, 5, "S0"))}
    result = _run(
        transcript,
        audit=_audit({"item_id": "bumpal", "segment_ids": [8], "suspicious_phrase": "Bumpal", "reason_for_suspicion": "uncertain"}),
        glossary=Glossary(("Bumpal", "Takem's Guide to Learning Japanese")),
    )
    unit = result["translation_units"][0]
    assert unit["source_text"] == original
    assert "Bunpro" not in unit["source_text"]
    assert "Bumpal" in unit["translation_raw"]
    assert "Takem's Guide to Learning Japanese" in unit["translation_raw"]


def test_naturalizer_rewrites_general_spoken_phrases_without_special_casing_full_sentence():
    naturalizer = ContextAwareVietnameseNaturalizer()
    raw = (
        "Đây là ba hệ thống viết chính, nên hãy học rằng khi bạn sử dụng ứng dụng. "
        "Và sau đó khi bạn tiếp tục học, tạo ra các câu ví dụ, như sử dụng những gì bạn đang học"
    )
    result = naturalizer.naturalize_with_report(
        raw,
        language="vi",
        source_context=(
            "The first thing you want to do is learn Hiragana Katakana. These are the three main "
            "writing systems, so learn that as you use an app and then as you keep learning, "
            "create example sentences, like use what you're learning."
        ),
    )
    assert "ba hệ chữ viết chính" in result.text
    assert "hãy học dần qua ứng dụng" in result.text
    assert "hãy đặt câu ví dụ" in result.text
    assert "áp dụng những gì bạn đang học" in result.text
    assert "Khi tiếp tục học" in result.text
    assert len(result.rules_applied) >= 4

    duration = naturalizer.naturalize(
        "Tôi đã học tiếng Nhật trong bảy năm, và hãy để tôi dạy các bạn làm thế nào để làm nó chính xác trong một",
        language="vi",
        source_context="I've been studying Japanese for seven years, and let me teach you how to do it exactly in one",
    )
    assert duration.startswith("Tôi đã học tiếng Nhật được bảy năm, và để tôi chỉ các bạn cách làm chính xác")
    assert duration.endswith("trong một")


def test_timing_review_flags_long_output_without_truncating_it():
    transcript = {"detected_language": "en", "segments": _segments(
        (0, "Say one thing", 0, 1, "S0"),
    )}
    raw = "Đây là một bản dịch rất dài có nhiều âm tiết và vẫn được giữ nguyên đầy đủ trong phần kết quả đầu ra"
    result = _run(transcript, engine=RecordingEngine(raw))
    unit = result["translation_units"][0]
    assert unit["needs_timing_review"] is True
    assert unit["translation_final"]
    assert unit["translation_final"].endswith("đầu ra")


def test_short_context_decode_is_retained_and_clause_fallback_prevents_silent_omission():
    class TruncatingEngine(RecordingEngine):
        def translate_for_dubbing(self, text, source_language, target_language):
            self.calls.append((text, source_language, target_language))
            if "first thing that you want" in text:
                return "Học Hiragana, Katakana và Kanji."
            if text.startswith("The first thing"):
                return "Học Hiragana, Katakana và Kanji."
            return "Đây là ba hệ chữ viết chính. Hãy đặt câu ví dụ để luyện tập."

    transcript = {"detected_language": "en", "segments": _segments(
        (5, "The first thing that you want to do is learn Hiragana Katakana and a little bit of kanji", 0, 2, "S0"),
        (6, "So these are the three writing systems and create example sentences for practice", 2, 4, "S0"),
    )}
    engine = TruncatingEngine()
    result = _run(transcript, engine=engine, glossary=Glossary(()))
    unit = result["translation_units"][0]
    assert unit["coverage_repair_applied"] is True
    assert unit["translation_raw"] == "Học Hiragana, Katakana và Kanji."
    assert [piece["source_segment_id"] for piece in unit["translation_fallback_segment_translations"]] == [5, 6]
    assert "ba hệ chữ viết chính" in unit["translation_final"]
    assert "đặt câu ví dụ" in unit["translation_final"]
    assert "source coverage" in unit["coverage_note"]


def test_mixed_language_segment_uses_inferred_source_route():
    source = "Juissanceiの時に日本語勉強した"
    transcript = {"detected_language": "ja", "segments": _segments((1, source, 0, 3, "S0"))}
    engine = RecordingEngine()
    _run(transcript, engine=engine, audit=_audit(), glossary=Glossary(()))
    assert engine.calls[0][1] == "ja"


def test_review_metadata_marks_uncertainty_and_translation_as_unreviewed():
    transcript = {"detected_language": "en", "segments": _segments(
        (0, "I recommend Cuban Japanese.", 0, 3, "S0"),
    )}
    result = _run(
        transcript,
        audit=_audit({"item_id": "cuban", "segment_ids": [0], "suspicious_phrase": "Cuban Japanese", "reason_for_suspicion": "uncertain"}),
    )
    unit = result["translation_units"][0]
    assert unit["source_uncertain"] is True
    assert "Cuban Japanese" in unit["uncertainty_reason"]
    assert any("human bilingual review" in reason for reason in unit["review_reasons"])
