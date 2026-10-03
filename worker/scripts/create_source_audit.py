from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKER_ROOT.parent
sys.path.insert(0, str(WORKER_ROOT / "src"))

from dubflow_worker.audit.source_audit import (  # noqa: E402
    build_source_audit,
    crop_wav,
    load_json,
    sha256_file,
)


def render_markdown(audit: dict) -> str:
    source = audit["source_of_truth"]
    lines = [
        "# DubFlow source / ASR audit v1", "",
        "Phase 1K.1 is an audit only. The saved transcript and base ASR artifacts remain unchanged; proposed interpretations below are listening hypotheses, not applied corrections.", "",
        "## Source provenance", "",
        f"- Transcript: `{source['transcript_file']}` (SHA-256 `{source['transcript_sha256']}`)",
        f"- Base ASR: `{source['base_asr_file']}` (SHA-256 `{source['base_asr_sha256']}`)",
        f"- Media: `{source['media_file']}` ({source['source_media_duration_seconds']:.3f}s)",
        "- Transcript artifacts modified: **No**", "",
        "## Findings", "",
    ]
    for item in audit["items"]:
        ranges = ", ".join(
            f"segment {stamp['segment_id']} ({stamp['start']:.2f}–{stamp['end']:.2f}s)"
            for stamp in item["timestamps_seconds"]
        )
        lines += [
            f"### {item['item_id']} — {item['suspicious_phrase']}", "",
            f"- **Source segment(s):** {ranges}",
            f"- **Detected language:** {item['segment_language']} (whole-file ASR language: {item['transcript_detected_language']})",
            f"- **Original ASR text:** `{item['original_asr_text']}`",
            f"- **Status:** {item['status']}",
            f"- **Why flagged:** {item['reason_for_suspicion']}",
            f"- **Proposed interpretation:** {item['proposed_interpretation'] or 'None; evidence is insufficient.'}",
            f"- **Manual audio listening:** Required",
            f"- **Downstream translation impact:** {item['downstream_translation_impact']}",
            f"- **NLLB naturalized output:** {item['translation_reference']['nllb_naturalized_translation']}",
            f"- **Audio snippet:** [`{item['audio_clip']['path']}`]({item['audio_clip']['path']}) ({item['audio_clip']['end_seconds'] - item['audio_clip']['start_seconds']:.2f}s)", "",
        ]
    lines += [
        "## Listening order", "",
        "Listen to each clip against its original segment before editing any transcript. Check spelling and word boundaries for names and web domains. For the two incomplete boundaries, listen through the boundary and adjacent speech to determine whether the ASR omitted a continuation or the speaker intentionally left a colloquial fragment.", "",
        "No corrections have been applied. Do not treat any proposed interpretation as canonical until manually verified.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a read-only source/ASR audit with focused WAV clips.")
    parser.add_argument("--transcript", type=Path, default=REPO_ROOT / "output/real-validation-v5-1/transcript.json")
    parser.add_argument("--asr-transcript", type=Path, default=REPO_ROOT / "output/real-validation-v5-1/asr/transcripts/sample-base.json")
    parser.add_argument("--comparison", type=Path, default=REPO_ROOT / "output/real-validation-v5-1/translation-comparison.json")
    parser.add_argument("--translated", type=Path, default=REPO_ROOT / "output/real-validation-v5-1/translated.json")
    parser.add_argument("--benchmark", type=Path, default=REPO_ROOT / "output/translation-benchmark-v1/benchmark.json")
    parser.add_argument("--media", type=Path, default=REPO_ROOT / "input/real_test/sample.mp4")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/source-audit-v1")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    transcript = load_json(args.transcript)
    asr_transcript = load_json(args.asr_transcript)
    audit = build_source_audit(
        transcript,
        asr_transcript,
        load_json(args.comparison),
        load_json(args.translated),
        load_json(args.benchmark),
        transcript_sha256=sha256_file(args.transcript),
        asr_sha256=sha256_file(args.asr_transcript),
    )

    from dubflow_worker.audio.extractor import AudioExtractor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    extractor = AudioExtractor(ffmpeg_path=args.ffmpeg)
    with extractor.extract(args.media) as full_audio:
        for item in audit["items"]:
            clip = item["audio_clip"]
            destination = args.output_dir / clip["path"]
            crop_wav(full_audio.path, destination, clip["start_seconds"], clip["end_seconds"])
            with wave.open(str(destination), "rb") as wav:
                clip["duration_seconds"] = round(wav.getnframes() / wav.getframerate(), 3)
                clip["sample_rate_hz"] = wav.getframerate()
                clip["channels"] = wav.getnchannels()

    json_path = args.output_dir / "source-audit.json"
    md_path = args.output_dir / "source-audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Audit items: {len(audit['items'])}")
    print(f"Audio snippets generated: {len(audit['items'])} in {audio_dir}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
