from __future__ import annotations

import hashlib
import json
import re
import wave
from pathlib import Path
from typing import Any

ALLOWED_STATUSES = {
    "confirmed", "likely", "uncertain", "probably ASR error", "probably proper noun",
}

# Timings are resolved against sample-base.json word records at build time. The two
# incomplete-boundary items intentionally use their whole source segment as context.
ITEM_SPECS = (
    {
        "item_id": "juissancei", "phrase": "Juissancei", "segment_ids": [1],
        "word_tokens": ["J", "u", "issance", "i"], "status": "uncertain",
        "reason": "The Latin spelling is embedded in a Japanese sentence, and the ASR word record splits it into four fragments. The text does not establish whether it is a name or a recognition error.",
        "interpretation": None,
        "impact": "NLLB preserves ‘Juissancei’ in Vietnamese; its referent and spelling remain unverified.",
    },
    {
        "item_id": "cuban-japanese", "phrase": "Cuban Japanese", "segment_ids": [3],
        "word_tokens": ["Cuban", "Japanese,"], "status": "uncertain",
        "reason": "‘I did Cuban Japanese’ is semantically unclear without knowing whether this is a named course/resource or an ASR error.",
        "interpretation": None,
        "impact": "The phrase is central to the recommendation and NLLB translates it literally as ‘tiếng Nhật Cuba’; a source error would propagate directly.",
    },
    {
        "item_id": "blogs-and-errors", "phrase": "good blogs and errors", "segment_ids": [4],
        "word_tokens": ["good", "blogs", "and", "errors,"], "status": "uncertain",
        "reason": "‘Good blogs and errors’ is an unusual collocation, but the transcript alone cannot distinguish an intended phrase from misrecognition.",
        "interpretation": None,
        "impact": "NLLB renders the phrase as ‘blog và lỗi tốt’; mistranscription would make the resulting clause misleading.",
    },
    {
        "item_id": "bumpal", "phrase": "Bumpal", "segment_ids": [8],
        "word_tokens": ["B", "ump", "al,"], "status": "uncertain",
        "reason": "The unfamiliar name-like term occurs in a sentence about grammar and is split into ‘B’, ‘ump’, and ‘al’ in the ASR word stream. This is a review cue, but subword timing alone does not establish an ASR error or the intended term.",
        "interpretation": None,
        "impact": "NLLB keeps ‘Bumpal’ while translating the surrounding grammar clause; the unknown term may refer to a resource or concept.",
    },
    {
        "item_id": "takems-guide-title", "phrase": "Takem's Guide to Learning Japanese", "segment_ids": [9],
        "word_tokens": ["Tak", "em", "'s", "Guide", "to", "Learning", "Japanese,"], "status": "probably proper noun",
        "reason": "The possessive name and title-like phrase are split across several ASR word records. The transcript supports a title/name reading, but not its spelling or bibliographic identity.",
        "interpretation": "A guide title or resource name is being recommended; exact title and author spelling need listening or external confirmation.",
        "impact": "NLLB translates part of the title (‘hướng dẫn của Takem để học tiếng Nhật’), so a name/title error would affect attribution and retrieval.",
    },
    {
        "item_id": "tofugood-domain", "phrase": "tofugood.com", "segment_ids": [3],
        "word_tokens": ["tofu", "good.", "com"], "status": "uncertain",
        "reason": "The ASR word stream records three pieces (‘tofu’, ‘good.’, ‘com’) with a gap before ‘com’. The domain spelling and word boundaries need confirmation.",
        "interpretation": "A website/domain is being recommended; the exact URL is not confirmed by the transcript.",
        "impact": "NLLB preserves ‘tofugood.com’; a spelling error could make the recommended resource unreachable.",
    },
    {
        "item_id": "incomplete-english-boundary", "phrase": "...exactly in one", "segment_ids": [0],
        "word_tokens": None, "status": "likely",
        "reason": "The English segment ends on ‘one’ at 3.40 seconds, immediately before the next Japanese segment begins. The saved transcript has no continuation for the phrase.",
        "interpretation": None,
        "impact": "NLLB and Argos both translate the fragment as if ‘one’ were complete; missing context may change the intended duration or object.",
    },
    {
        "item_id": "incomplete-japanese-boundary", "phrase": "とりあえずアップストアのアプリを全部ダロノードして...と思ってて", "segment_ids": [2],
        "word_tokens": None, "status": "likely",
        "reason": "The Japanese segment ends at 17.16 seconds with ‘と思ってて’, a continuing colloquial form, just before a speaker change. Within the same ASR text, ‘ダロノード’ is also a nonstandard-looking rendering in the App Store/apps context; the transcript cannot confirm the spoken word.",
        "interpretation": "‘ダウンロード’ is a possible reading of ‘ダロノード’ from the surrounding ‘App Store apps’ context and phonetic resemblance; treat only as a listening hypothesis, not a correction.",
        "impact": "NLLB translates the apps clause but cannot restore the unfinished continuation; an incorrect verb reading changes the action being described.",
    },
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def segment_by_id(transcript: dict[str, Any], segment_id: int) -> dict[str, Any]:
    matches = [segment for segment in transcript.get("segments", []) if segment.get("id") == segment_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one transcript segment {segment_id}, found {len(matches)}")
    return matches[0]


def word_span(asr_transcript: dict[str, Any], segment_id: int, tokens: list[str]) -> tuple[float, float]:
    segment = segment_by_id(asr_transcript, segment_id)
    words = segment.get("words", [])
    for start in range(0, len(words) - len(tokens) + 1):
        if [word.get("word") for word in words[start:start + len(tokens)]] == tokens:
            return float(words[start]["start"]), float(words[start + len(tokens) - 1]["end"])
    raise ValueError(f"Could not resolve word tokens {tokens!r} in ASR segment {segment_id}")


def _segment_language(text: str) -> str:
    japanese = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))
    latin = bool(re.search(r"[A-Za-z]", text))
    if japanese and latin:
        return "mixed (ja + en/name text)"
    if japanese:
        return "ja"
    if latin:
        return "en"
    return "unknown"


def _check_artifact_agreement(
    transcript: dict[str, Any], comparison: list[dict[str, Any]],
    translated: dict[str, Any], benchmark: dict[str, Any],
) -> None:
    transcript_segments = {row["id"]: row for row in transcript["segments"]}
    translated_units = {row["translation_unit_id"]: row for row in translated.get("translation_units", [])}
    comparison_units = {row["translation_unit_id"]: row for row in comparison}
    baseline = next((row for row in benchmark.get("candidates", []) if row.get("name") == "NLLB baseline"), None)
    if baseline is None:
        raise ValueError("benchmark.json has no NLLB baseline candidate")
    for unit_id, comparison_row in comparison_units.items():
        segment_ids = comparison_row.get("source_segment_ids", [])
        expected = " ".join(transcript_segments[item]["text"] for item in segment_ids)
        # Phase 1I.1 concatenation uses spaces; unit source text is separately preserved in translated.json.
        translated_source = translated_units.get(unit_id, {}).get("source_text")
        if translated_source != expected:
            raise ValueError(f"Transcript/comparison source mismatch for {unit_id}")
        if comparison_row.get("complete_source_text") != expected:
            raise ValueError(f"Comparison source differs from transcript for {unit_id}")
    for unit_id, unit in translated_units.items():
        if unit_id not in comparison_units:
            continue
        if unit.get("source_text") != comparison_units[unit_id].get("complete_source_text"):
            raise ValueError(f"Translated artifact source differs from comparison for {unit_id}")


def _translation_reference(segment_ids: list[int], comparison: list[dict[str, Any]], benchmark: dict[str, Any]) -> dict[str, Any]:
    comparison_row = next((
        row for row in comparison
        if set(segment_ids).issubset(set(row.get("source_segment_ids", [])))
    ), None)
    if comparison_row is None:
        raise ValueError(f"No translation-comparison row maps source segments {segment_ids}")
    unit_id = comparison_row["translation_unit_id"]
    candidate = next(row for row in benchmark["candidates"] if row.get("name") == "NLLB baseline")
    result = next(row for row in candidate["units"] if row["translation_unit_id"] == unit_id)
    return {
        "translation_unit_id": unit_id,
        "comparison_final_translation": comparison_row.get("final_translation"),
        "nllb_raw_translation": result.get("raw_translation"),
        "nllb_naturalized_translation": result.get("naturalized_translation"),
    }


def build_source_audit(
    transcript: dict[str, Any], asr_transcript: dict[str, Any],
    comparison: list[dict[str, Any]], translated: dict[str, Any], benchmark: dict[str, Any],
    *, transcript_sha256: str, asr_sha256: str,
) -> dict[str, Any]:
    _check_artifact_agreement(transcript, comparison, translated, benchmark)
    asr_segments = {row["id"]: row for row in asr_transcript.get("segments", [])}
    if any(asr_segments.get(row["id"], {}).get("text") != row["text"] for row in transcript["segments"]):
        raise ValueError("Merged transcript text differs from the saved base ASR transcript")

    items: list[dict[str, Any]] = []
    for spec in ITEM_SPECS:
        source_segments = [segment_by_id(transcript, segment_id) for segment_id in spec["segment_ids"]]
        asr_source_segments = [segment_by_id(asr_transcript, segment_id) for segment_id in spec["segment_ids"]]
        original_text = " ".join(segment["text"] for segment in source_segments)
        if spec["word_tokens"]:
            clip_start, clip_end = word_span(asr_transcript, spec["segment_ids"][0], spec["word_tokens"])
        else:
            clip_start = min(float(segment["start"]) for segment in source_segments)
            clip_end = max(float(segment["end"]) for segment in source_segments)
        # Small context pads preserve coarticulation; clips remain bounded by the source duration.
        clip_start = max(0.0, clip_start - 0.25)
        clip_end = min(float(transcript["source"]["duration"]), clip_end + 0.35)
        item = {
            "item_id": spec["item_id"],
            "segment_ids": spec["segment_ids"],
            "original_asr_text": original_text,
            "asr_word_text": " ".join(segment["text"] for segment in asr_source_segments),
            "segment_language": _segment_language(original_text),
            "transcript_detected_language": transcript.get("detected_language", transcript.get("asr", {}).get("language")),
            "timestamps_seconds": [
                {"segment_id": segment["id"], "start": float(segment["start"]), "end": float(segment["end"])}
                for segment in source_segments
            ],
            "suspicious_phrase": spec["phrase"],
            "reason_for_suspicion": spec["reason"],
            "status": spec["status"],
            "proposed_interpretation": spec["interpretation"],
            "manual_audio_listening_required": True,
            "downstream_translation_impact": spec["impact"],
            "translation_reference": _translation_reference(spec["segment_ids"], comparison, benchmark),
            "audio_clip": {
                "path": f"audio/{spec['item_id']}.wav",
                "start_seconds": round(clip_start, 3),
                "end_seconds": round(clip_end, 3),
                "context_padding_seconds": {"before": 0.25, "after": 0.35},
            },
        }
        items.append(item)

    audit = {
        "audit_version": "1.0",
        "phase": "1K.1",
        "purpose": "Source/ASR audit only; no source text was corrected or replaced.",
        "source_of_truth": {
            "transcript_file": "output/real-validation-v5-1/transcript.json",
            "transcript_sha256": transcript_sha256,
            "base_asr_file": "output/real-validation-v5-1/asr/transcripts/sample-base.json",
            "base_asr_sha256": asr_sha256,
            "comparison_file": "output/real-validation-v5-1/translation-comparison.json",
            "benchmark_file": "output/translation-benchmark-v1/benchmark.json",
            "media_file": "input/real_test/sample.mp4",
            "source_media_duration_seconds": transcript["source"]["duration"],
            "transcript_artifacts_modified": False,
        },
        "items": items,
    }
    validate_audit(audit)
    return audit


def validate_audit(audit: dict[str, Any]) -> None:
    if audit.get("audit_version") != "1.0" or not isinstance(audit.get("items"), list):
        raise ValueError("Invalid source audit root schema")
    seen: set[str] = set()
    required = {
        "item_id", "segment_ids", "original_asr_text", "segment_language", "timestamps_seconds",
        "suspicious_phrase", "reason_for_suspicion", "status", "proposed_interpretation",
        "manual_audio_listening_required", "downstream_translation_impact", "audio_clip",
    }
    for item in audit["items"]:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Audit item missing fields: {sorted(missing)}")
        if item["item_id"] in seen:
            raise ValueError(f"Duplicate audit item {item['item_id']}")
        seen.add(item["item_id"])
        if item["status"] not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid audit status {item['status']!r}")
        if not item["segment_ids"] or not item["original_asr_text"]:
            raise ValueError(f"Audit item {item['item_id']} has no source segment or text")
        clip = item["audio_clip"]
        if clip["start_seconds"] < 0 or clip["end_seconds"] <= clip["start_seconds"]:
            raise ValueError(f"Audit item {item['item_id']} has invalid clip interval")


def crop_wav(source: Path, destination: Path, start_seconds: float, end_seconds: float) -> None:
    """Write a WAV interval while preserving the extracted source's PCM format."""
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("WAV crop interval must satisfy 0 <= start < end")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        first = min(reader.getnframes(), round(start_seconds * rate))
        last = min(reader.getnframes(), round(end_seconds * rate))
        if last <= first:
            raise ValueError("Requested WAV crop contains no frames")
        reader.setpos(first)
        frames = reader.readframes(last - first)
        params = reader.getparams()
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
