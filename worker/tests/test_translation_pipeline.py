from __future__ import annotations

from dubflow_worker.pipeline.translate import translate_speaker_transcript
from dubflow_worker.translation.base import TranslationEngine, UnsupportedLanguageError


class FakeTranslationEngine(TranslationEngine):
    name = "fake"
    device = "cpu"

    def __init__(self, failure_text: str | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.failure_text = failure_text

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        self.calls.append((text, source_language, target_language))
        if self.failure_text and text == self.failure_text:
            raise RuntimeError("fake provider error")
        if source_language == "xx" or target_language == "xx":
            raise UnsupportedLanguageError(f"Unsupported pair: {source_language}->{target_language}")
        return f"{target_language}:{text}"


def make_transcript(*texts: str) -> dict:
    return {
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
        "engine": "fake", "source_language": "en", "target_language": "vi",
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


def test_unsupported_language_pair_is_recorded_per_segment() -> None:
    result = translate_speaker_transcript(
        make_transcript("Hello.", "Again."),
        source_language="xx", target_language="vi", engine=FakeTranslationEngine(),
    )
    assert all(segment.target_text is None for segment in result.segments)
    assert all(segment.error and segment.error.startswith("UnsupportedLanguageError:") for segment in result.segments)
