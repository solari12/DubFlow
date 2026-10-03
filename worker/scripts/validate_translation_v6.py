from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKER_ROOT.parent
sys.path.insert(0, str(WORKER_ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

# This validation is local-only and must never download weights implicitly.
os.environ["HF_HOME"] = str(WORKER_ROOT / ".model-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
user_hub = Path.home() / ".cache" / "huggingface" / "hub"
user_nllb = user_hub / "models--facebook--nllb-200-distilled-600M" / "snapshots"
if any(user_nllb.glob("*/config.json")):
    os.environ["HF_HUB_CACHE"] = str(user_hub)

from dubflow_worker.pipeline.translate_v6 import translate_context_units_v6  # noqa: E402
from dubflow_worker.translation.glossary import Glossary  # noqa: E402
from dubflow_worker.translation.naturalization_v3 import ContextAwareVietnameseNaturalizer  # noqa: E402
from dubflow_worker.translation.nllb import NllbTranslationEngine  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Translation-only Phase 1K.2 validation using the saved transcript.")
    parser.add_argument("--transcript", type=Path, default=REPO_ROOT / "output/real-validation-v5-1/transcript.json")
    parser.add_argument("--source-audit", type=Path, default=REPO_ROOT / "output/source-audit-v1/source-audit.json")
    parser.add_argument("--glossary", type=Path, default=WORKER_ROOT / "config/translation-glossary.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/real-validation-v6")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    transcript_sha = sha256_file(args.transcript)
    source_audit = read_json(args.source_audit)
    expected_sha = source_audit.get("source_of_truth", {}).get("transcript_sha256")
    if expected_sha and transcript_sha != expected_sha:
        raise ValueError("The source transcript differs from the immutable Phase 1K.1 audited transcript")
    transcript = read_json(args.transcript)
    glossary = Glossary.from_json(args.glossary)
    engine = NllbTranslationEngine(device=args.device)
    naturalizer = ContextAwareVietnameseNaturalizer()

    run_started = time.perf_counter()
    result = translate_context_units_v6(
        transcript,
        engine=engine,
        naturalizer=naturalizer,
        glossary=glossary,
        source_audit=source_audit,
        target_language="vi",
    )
    total_runtime = time.perf_counter() - run_started
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transcript_out = args.output_dir / "transcript.json"
    shutil.copyfile(args.transcript, transcript_out)
    copied_sha = sha256_file(transcript_out)
    if copied_sha != transcript_sha:
        raise RuntimeError("Output transcript copy failed byte-for-byte integrity check")

    unit_rows = result["translation_units"]
    source_segments = transcript.get("segments", [])
    translated = {
        "version": "4.0",
        "success": result["success"],
        "source_of_truth": {
            "transcript_sha256": transcript_sha,
            "immutable_copy_verified": copied_sha == transcript_sha,
            "source_audit": str(args.source_audit.relative_to(REPO_ROOT)),
        },
        "translation": {
            "engine": result["engine"],
            "detected_language": result["source_language"],
            "translation_target_language": result["target_language"],
            "naturalization_provider": result["naturalizer"],
            "glossary_terms": list(glossary.terms),
            "mapping": "Each canonical translation unit lists every source_segment_id exactly once; segment rows only reference their unit.",
        },
        "source_coverage": result["source_coverage"],
        "translation_units": unit_rows,
        "segments": result["segments"],
        "quality": {
            "source_segment_count": len(source_segments),
            "translation_unit_count": len(unit_rows),
            "failed_unit_count": result["counts"]["failed_units"],
            "needs_review_unit_count": result["counts"]["needs_review"],
            "source_uncertain_unit_count": result["counts"]["source_uncertain"],
            "needs_timing_review_unit_count": result["counts"]["needs_timing_review"],
            "human_review_required": True,
            "automatic_quality_claimed": False,
        },
        "runtime": {
            "model_load_seconds": result["model_load_seconds"],
            "nllb_translation_seconds": result["translation_seconds"],
            "naturalization_seconds": result["naturalization_seconds"],
            "total_validation_seconds": round(total_runtime, 6),
            "device": args.device,
        },
    }
    comparison = [{
        "unit_id": row["unit_id"],
        "source_segment_ids": row["source_segment_ids"],
        "speaker_ids": row["speaker_ids"],
        "start": row["start"],
        "end": row["end"],
        "source_text": row["source_text"],
        "source_language": row["source_language"],
        "translation_raw": row["translation_raw"],
        "translation_fallback_segment_translations": row["translation_fallback_segment_translations"],
        "translation_base_for_naturalization": row["translation_base_for_naturalization"],
        "coverage_repair_applied": row["coverage_repair_applied"],
        "translation_final": row["translation_final"],
        "needs_review": row["needs_review"],
        "needs_timing_review": row["needs_timing_review"],
        "source_uncertain": row["source_uncertain"],
        "uncertainty_reason": row["uncertainty_reason"],
        "review_reasons": row["review_reasons"],
    } for row in unit_rows]
    examples = []
    for unit_id in ("tu-0005", "tu-0006", "tu-0000"):
        row = next((item for item in unit_rows if item["unit_id"] == unit_id), None)
        if row is not None:
            examples.append({
                "unit_id": unit_id,
                "source_text": row["source_text"],
                "translation_raw": row["translation_raw"],
                "translation_fallback_segment_translations": row["translation_fallback_segment_translations"],
                "translation_final": row["translation_final"],
            })
    report = {
        "version": "1.0",
        "phase": "1K.2",
        "status": "COMPLETED_REVIEW_REQUIRED" if result["success"] else "COMPLETED_WITH_TRANSLATION_ERRORS",
        "source_transcript_sha256": transcript_sha,
        "source_transcript_unchanged": True,
        "source_segment_count": len(source_segments),
        "translation_unit_count": len(unit_rows),
        "source_coverage": result["source_coverage"],
        "provider": result["engine"],
        "model": engine.model_name,
        "device": args.device,
        "model_load_seconds": result["model_load_seconds"],
        "nllb_runtime_seconds": result["translation_seconds"],
        "naturalization_runtime_seconds": result["naturalization_seconds"],
        "total_validation_seconds": round(total_runtime, 6),
        "units_requiring_review": result["counts"]["needs_review"],
        "units_with_source_uncertainty": result["counts"]["source_uncertain"],
        "units_needing_timing_review": result["counts"]["needs_timing_review"],
        "translation_errors": result["counts"]["failed_units"],
        "timing_screening_note": "Whitespace-separated Vietnamese syllable count at 4.5 units/s is a conservative triage signal only; no timing fit, truncation, TTS, or alignment was performed.",
        "representative_before_after": examples,
        "automatic_quality_claimed": False,
    }

    _write_json(args.output_dir / "translated.json", translated)
    _write_json(args.output_dir / "translation-units.json", unit_rows)
    _write_json(args.output_dir / "translation-comparison.json", comparison)
    _write_json(args.output_dir / "validation-report.json", report)

    print("Phase 1K.2 translation-only validation")
    print(f"Source segments: {len(source_segments)}; translation units: {len(unit_rows)}")
    print(f"Source coverage: each segment exactly once = {result['source_coverage']['each_segment_assigned_exactly_once']}")
    print(f"NLLB: {engine.model_name}; load={result['model_load_seconds']:.2f}s; translate={result['translation_seconds']:.2f}s")
    print(f"Naturalization: {naturalizer.name}; {result['naturalization_seconds']:.3f}s")
    print("Units requiring review: " + str(result["counts"]["needs_review"]))
    print("Units with source uncertainty: " + str(result["counts"]["source_uncertain"]))
    print("Units needing timing review: " + str(result["counts"]["needs_timing_review"]))
    for example in examples:
        print(f"\n{example['unit_id']} BEFORE: {example['translation_raw']}\n{example['unit_id']} AFTER:  {example['translation_final']}")
    print(f"\nWrote v6 translation artifacts to {args.output_dir}")
    print("No TTS, alignment, rendering, or full-video validation was run.")
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
