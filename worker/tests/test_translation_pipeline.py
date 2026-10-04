from __future__ import annotations

from dubflow_worker.pipeline.translate import translate_speaker_transcript
from dubflow_worker.translation.base import TranslationEngine, UnsupportedLanguageError
from dubflow_worker.translation.languages import nllb_language_code
from dubflow_worker.translation.units import group_translation_units, infer_segment_language


class FakeTranslationEngine(TranslationEngine):
    name = "fake"
    device = "cpu"

    def __init__(self, failure_text: str | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.validated_pairs: list[tuple[str, str]] = []
        self.failure_text = failure_text

    def validate_language_pair(self, source_language: str, target_language: str) -> None:
        self.validated_pairs.append((source_language, target_language))
        if source_language == "xx" or target_language == "xx":
            raise UnsupportedLanguageError(f"Unsupported pair: {source_language}->{target_language}")

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        self.calls.append((text, source_language, target_language))
        if self.failure_text and text == self.failure_text:
            raise RuntimeError("fake provider error")
        if source_language == "xx" or target_language == "xx":
            raise UnsupportedLanguageError(f"Unsupported pair: {source_language}->{target_language}")
        return f"{target_language}:{text}"


def make_transcript(*texts: str, language: str | None = None) -> dict:
    return {
        "asr": {"language": language} if language else {},
        "segments": [
            {
                "id": index + 4,
                "start": index * 2.5 + 0.25,
                "end": index * 2.5 + 1.75,
                "speaker": "SPEAKER_00" if index % 2 == 0 else None,
                "text": text,
            }
            for index, text in enumerate(texts)
        ]
    }


def test_basic_translation_produces_translated_transcript() -> None:
    result = translate_speaker_transcript(
        make_transcript("Hello."), source_language="en", target_language="vi",
        engine=FakeTranslationEngine(),
    ).to_dict()
    assert result["segments"][0]["target_text"] == "vi:Hello."
    assert result["segments"][0]["error"] is None


def test_source_and_target_languages_are_propagated() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("Hello."), source_language="en", target_language="vi", engine=engine,
    ).to_dict()
    assert engine.calls == [("Hello.", "en", "vi")]
    assert result["translation"] == {
        "engine": "fake",
        "detected_language": None,
        "translation_source_language": "en",
        "translation_target_language": "vi",
        "source_language_overridden": True,
        "segment_source_languages": {"4": "en"},
    }


def test_speaker_is_preserved() -> None:
    result = translate_speaker_transcript(
        make_transcript("Hello."), source_language="en", target_language="vi",
        engine=FakeTranslationEngine(),
    )
    assert result.segments[0].speaker == "SPEAKER_00"


def test_timestamps_are_preserved() -> None:
    result = translate_speaker_transcript(
        make_transcript("Hello."), source_language="en", target_language="vi",
        engine=FakeTranslationEngine(),
    )
    assert (result.segments[0].start, result.segments[0].end) == (0.25, 1.75)


def test_segment_id_is_preserved() -> None:
    result = translate_speaker_transcript(
        make_transcript("Hello."), source_language="en", target_language="vi",
        engine=FakeTranslationEngine(),
    )
    assert result.segments[0].id == 4


def test_empty_text_is_preserved_without_calling_engine() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("  "), source_language="en", target_language="vi", engine=engine,
    )
    assert result.segments[0].source_text == "  "
    assert result.segments[0].target_text == ""
    assert result.segments[0].error is None
    assert engine.calls == []


def test_one_failed_translation_is_explicit_and_does_not_stop_later_segments() -> None:
    engine = FakeTranslationEngine(failure_text="Fail me.")
    result = translate_speaker_transcript(
        make_transcript("Good.", "Fail me.", "Also good."),
        source_language="en", target_language="vi", engine=engine,
    )
    assert result.segments[0].target_text == "vi:Good."
    assert result.segments[1].target_text is None
    assert result.segments[1].error == "RuntimeError: fake provider error"
    assert result.segments[2].target_text == "vi:Also good."


def test_multiple_segments_translate_independently_and_in_order() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("One.", "Two.", "Three."),
        source_language="en", target_language="vi", engine=engine,
    )
    assert [segment.target_text for segment in result.segments] == [
        "vi:One.", "vi:Two.", "vi:Three.",
    ]
    assert [segment.id for segment in result.segments] == [4, 5, 6]


def test_unsupported_language_pair_fails_before_translating_segments() -> None:
    engine = FakeTranslationEngine()
    try:
        translate_speaker_transcript(
            make_transcript("Hello.", "Again."),
            source_language="xx", target_language="vi", engine=engine,
        )
    except UnsupportedLanguageError as exc:
        assert "xx->vi" in str(exc)
    else:
        raise AssertionError("unsupported language pair must fail before translating segments")
    assert engine.calls == []


def test_detected_japanese_routes_to_japanese_vietnamese() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("こんにちは", language="ja"), target_language="vi", engine=engine,
    ).to_dict()
    assert engine.calls == [("こんにちは", "ja", "vi")]
    assert result["translation"]["detected_language"] == "ja"
    assert result["translation"]["translation_source_language"] == "ja"


def test_detected_english_routes_to_english_vietnamese() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("Hello", language="en"), target_language="vi", engine=engine,
    ).to_dict()
    assert engine.calls == [("Hello", "en", "vi")]
    assert result["translation"]["translation_source_language"] == "en"


def test_explicit_source_language_overrides_asr_detection() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("Hello", language="ja"),
        source_language="en", target_language="vi", engine=engine,
    ).to_dict()
    assert engine.calls == [("Hello", "en", "vi")]
    assert result["translation"] == {
        "engine": "fake",
        "detected_language": "ja",
        "translation_source_language": "en",
        "translation_target_language": "vi",
        "source_language_overridden": True,
        "segment_source_languages": {"4": "en"},
    }


def test_detected_and_resolved_translation_metadata_are_recorded() -> None:
    result = translate_speaker_transcript(
        make_transcript("Hello", language="en"), target_language="vi", engine=FakeTranslationEngine(),
    ).to_dict()
    metadata = result["translation"]
    assert metadata["detected_language"] == "en"
    assert metadata["translation_source_language"] == "en"
    assert metadata["translation_target_language"] == "vi"
    assert metadata["source_language_overridden"] is False


def test_english_segments_in_japanese_clip_use_english_nllb_route() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("So pick one or a couple", language="ja"),
        target_language="vi", engine=engine,
    ).to_dict()

    assert engine.calls == [("So pick one or a couple", "en", "vi")]
    assert nllb_language_code(result["translation"]["segment_source_languages"]["4"]) == "eng_Latn"
    assert result["translation"]["detected_language"] == "ja"
    assert result["translation"]["translation_source_language"] == "en"


def test_japanese_segments_use_japanese_nllb_route() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("日本語を勉強しています", language="ja"),
        target_language="vi", engine=engine,
    ).to_dict()

    assert engine.calls == [("日本語を勉強しています", "ja", "vi")]
    assert nllb_language_code(result["translation"]["segment_source_languages"]["4"]) == "jpn_Jpan"


def test_mixed_japanese_english_uses_existing_dominant_script_policy() -> None:
    mixed = "ひらがなかたかな and a little bit of kanji"
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript(mixed, language="ja"), target_language="vi", engine=engine,
    ).to_dict()

    # The existing unit policy routes a substantial Japanese-script share as Japanese.
    assert engine.calls == [(mixed, "ja", "vi")]
    assert nllb_language_code(result["translation"]["segment_source_languages"]["4"]) == "jpn_Jpan"
    # One Japanese character in a Latin segment does not flip the source route.
    assert infer_segment_language("This is English text ん.", "ja") == "en"


def test_vietnamese_segments_keep_vietnamese_nllb_route() -> None:
    engine = FakeTranslationEngine()
    result = translate_speaker_transcript(
        make_transcript("Tôi đang học tiếng Việt.", language="vi"),
        target_language="vi", engine=engine,
    ).to_dict()

    assert engine.calls == [("Tôi đang học tiếng Việt.", "vi", "vi")]
    assert nllb_language_code(result["translation"]["segment_source_languages"]["4"]) == "vie_Latn"


def test_per_segment_routing_preserves_ids_speakers_times_order_and_metadata() -> None:
    engine = FakeTranslationEngine()
    source = make_transcript(
        "Hello there.", "こんにちは。", language="ja"
    )
    result = translate_speaker_transcript(
        source, target_language="vi", engine=engine,
    ).to_dict()

    assert [(row["id"], row["speaker"], row["start"], row["end"]) for row in result["segments"]] == [
        (4, "SPEAKER_00", 0.25, 1.75),
        (5, None, 2.75, 4.25),
    ]
    assert [row["target_text"] for row in result["segments"]] == ["vi:Hello there.", "vi:こんにちは。"]
    assert result["translation"]["detected_language"] == "ja"
    assert result["translation"]["translation_source_language"] == "mixed"
    assert result["translation"]["segment_source_languages"] == {"4": "en", "5": "ja"}


def test_vietnamese_translation_is_naturalized_after_segment_language_routing() -> None:
    incomplete = "I've been studying Japanese for 7 years, and let me teach you guys how to do it exactly in one"
    japanese = "13\u6b73\u306e\u6642\u306b\u65e5\u672c\u8a9e\u52c9\u5f37\u306f\u3058\u3081\u305f\u3093\u3060\u3051\u3069"
    first_step = "The first thing that you want to do is learn Hiragana Katakana and a little bit of kanji"
    resource = "That's where I recommend implementing Takem's Guide to Learning Japanese, it's this free resource"

    class NaturalizationEngine(FakeTranslationEngine):
        translations = {
            incomplete: "Tôi đã học tiếng Nhật trong 7 năm, và hãy để tôi dạy các bạn làm thế nào để làm nó chính xác trong một",
            japanese: "Nhưng tôi bắt đầu học tiếng Nhật khi tôi 13 tuổi, và điều đầu tiên tôi làm là ứng dụng.",
            first_step: "Điều đầu tiên bạn muốn làm là học Hiragana Katakana và một chút kanji",
            resource: "Đó là nơi tôi khuyên bạn nên thực hiện hướng dẫn học tiếng Nhật của Takem, đó là nguồn tài nguyên miễn phí này",
        }

        def translate(self, text, source_language, target_language):
            self.calls.append((text, source_language, target_language))
            return self.translations[text]

    transcript = {
        "asr": {"language": "ja"},
        "segments": [
            {"id": 0, "start": 0.0, "end": 3.38, "speaker": "S0", "text": incomplete},
            {"id": 1, "start": 3.38, "end": 5.0, "speaker": "S0", "text": japanese},
            {"id": 5, "start": 30.34, "end": 34.14, "speaker": "S0", "text": first_step},
            {"id": 9, "start": 50.6, "end": 55.6, "speaker": "S0", "text": resource},
        ],
    }
    engine = NaturalizationEngine()
    result = translate_speaker_transcript(transcript, target_language="vi", engine=engine)
    rows = {row.id: row for row in result.segments}
    units = group_translation_units(transcript["segments"], fallback_language="ja")

    assert [call[1] for call in engine.calls] == ["en", "ja", "en", "en"]
    assert units[0].source_segment_ids == (0,)
    assert units[1].source_segment_ids == (1,)
    assert rows[0].source_text == incomplete
    assert "được 7 năm" in rows[0].target_text
    assert rows[0].target_text.endswith("trong một")
    assert "\\1" not in rows[0].target_text
    assert "dạy các bạn làm thế nào để làm nó" not in rows[0].target_text
    assert "năm 13 tuổi" in rows[1].target_text
    assert "Việc đầu tiên là dùng ứng dụng" in rows[1].target_text
    assert "Hiragana, Katakana" in rows[5].target_text
    assert "một chút kanji" in rows[5].target_text
    assert "tham khảo" in rows[9].target_text
    assert "Takem" in rows[9].target_text
    assert "tài liệu miễn phí" in rows[9].target_text
    assert [row.id for row in result.segments] == [0, 1, 5, 9]
