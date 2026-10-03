from __future__ import annotations

import wave

import pytest

from dubflow_worker.audit.source_audit import (
    build_source_audit,
    crop_wav,
    segment_by_id,
    validate_audit,
)


TEXTS = {
    0: "I've been studying Japanese for seven years, and let me teach you guys how to do it exactly in one",
    1: "Juissanceiの時に日本語勉強はじめたんだけど、一番最初にやったことが、アプリなんだよね",
    2: "とりあえずアップストアのアプリを全部ダロノードして、まあ全部じゃないんだけど、なんか自分にお勉強方法からスタートした方がいいと思ってて",
    3: "So pick one or a couple, I did Cuban Japanese, and I also highly recommend tofugood.com",
    4: "They have a lot of good blogs and errors, you can really learn how to learn if that makes sense",
    5: "The first thing that you want to do is learn Hiragana Katakana and a little bit of kanji",
    6: "So these are the three main writing systems, so learn that as you use an app",
    7: "and then as you keep learning, create example sentences, like use what you're learning",
    8: "and soon enough you'll find Bumpal, which is grammar, to be a little confusing",
    9: "That's where I recommend implementing Takem's Guide to Learning Japanese, it's this free resource",
    10: "and using that, learn the grammar and the principles",
}

WORDS = {
    1: ["J", "u", "issance", "i", "の"],
    3: ["Cuban", "Japanese,", "tofu", "good.", "com"],
    4: ["good", "blogs", "and", "errors,"],
    8: ["B", "ump", "al,"],
    9: ["Tak", "em", "'s", "Guide", "to", "Learning", "Japanese,"],
}


def fixture_artifacts():
    transcript_segments = [
        {"id": index, "start": float(index), "end": float(index + 1), "text": text}
        for index, text in TEXTS.items()
    ]
    asr_segments = []
    for segment in transcript_segments:
        words = WORDS.get(segment["id"], [segment["text"]])
        base = float(segment["id"])
        asr_segments.append({
            **segment,
            "words": [
                {"word": word, "start": base + index * 0.05, "end": base + index * 0.05 + 0.04}
                for index, word in enumerate(words)
            ],
        })
    comparison = []
    translation_units = []
    benchmark_units = []
    for index, text in TEXTS.items():
        unit_id = f"tu-{index:04d}"
        comparison.append({
            "translation_unit_id": unit_id, "source_segment_ids": [index],
            "complete_source_text": text, "final_translation": f"final {index}",
        })
        translation_units.append({
            "translation_unit_id": unit_id, "source_text": text,
            "source_language": "ja" if index in {1, 2} else "en",
        })
        benchmark_units.append({
            "translation_unit_id": unit_id, "source_text": text,
            "raw_translation": f"raw {index}", "naturalized_translation": f"natural {index}",
        })
    return (
        {"source": {"duration": 60.0}, "detected_language": "ja", "segments": transcript_segments},
        {"segments": asr_segments}, comparison,
        {"translation_units": translation_units},
        {"candidates": [{"name": "NLLB baseline", "units": benchmark_units}]},
    )


def test_segment_lookup_is_exact_and_reports_missing_id():
    transcript, *_ = fixture_artifacts()
    assert segment_by_id(transcript, 3)["text"] == TEXTS[3]
    with pytest.raises(ValueError, match="segment 99"):
        segment_by_id(transcript, 99)


def test_build_audit_schema_traces_original_source_and_translation():
    artifacts = fixture_artifacts()
    audit = build_source_audit(
        *artifacts, transcript_sha256="transcript-hash", asr_sha256="asr-hash",
    )

    assert len(audit["items"]) == 8
    assert audit["source_of_truth"]["transcript_artifacts_modified"] is False
    by_id = {item["item_id"]: item for item in audit["items"]}
    assert by_id["cuban-japanese"]["original_asr_text"] == TEXTS[3]
    assert by_id["cuban-japanese"]["translation_reference"]["nllb_raw_translation"] == "raw 3"
    assert by_id["incomplete-japanese-boundary"]["proposed_interpretation"].startswith("‘ダウンロード’")
    validate_audit(audit)


def test_schema_rejects_unknown_confidence_status():
    audit = {
        "audit_version": "1.0",
        "items": [{
            "item_id": "x", "segment_ids": [0], "original_asr_text": "source",
            "segment_language": "en", "timestamps_seconds": [], "suspicious_phrase": "source",
            "reason_for_suspicion": "test", "status": "certainly wrong",
            "proposed_interpretation": None, "manual_audio_listening_required": True,
            "downstream_translation_impact": "test",
            "audio_clip": {"start_seconds": 0.0, "end_seconds": 1.0},
        }],
    }
    with pytest.raises(ValueError, match="Invalid audit status"):
        validate_audit(audit)


def test_crop_wav_writes_requested_audio_interval(tmp_path):
    source = tmp_path / "source.wav"
    cropped = tmp_path / "audio" / "crop.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(b"\x00\x00" * 5000)

    crop_wav(source, cropped, 1.25, 2.75)

    with wave.open(str(cropped), "rb") as wav:
        assert wav.getnframes() == 1500
        assert wav.getframerate() == 1000
