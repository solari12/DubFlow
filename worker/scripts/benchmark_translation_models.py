from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKER_ROOT.parent
sys.path.insert(0, str(WORKER_ROOT / "src"))

# Keep model resolution local and prevent the harness from downloading weights.
os.environ["HF_HOME"] = str(WORKER_ROOT / ".model-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
workspace_hub = WORKER_ROOT / ".model-cache" / "hub"
user_hub = Path.home() / ".cache" / "huggingface" / "hub"
local_nllb_cache = workspace_hub / "models--facebook--nllb-200-distilled-600M" / "snapshots"
if not any(local_nllb_cache.glob("*/config.json")):
    cached_user_nllb = user_hub / "models--facebook--nllb-200-distilled-600M" / "snapshots"
    if any(cached_user_nllb.glob("*/config.json")):
        os.environ["HF_HUB_CACHE"] = str(user_hub)

from dubflow_worker.benchmark.translation_models import (  # noqa: E402
    build_review_matrix,
    environment_info,
    load_units,
    run_candidate,
    skipped_candidate,
)
from dubflow_worker.translation.naturalization import (  # noqa: E402
    DeterministicVietnameseNaturalizer,
)


def argos_package_version() -> str:
    metadata_path = WORKER_ROOT / ".model-cache" / "argos-packages" / "translate-en_vi-1_9" / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return f"Argos {metadata.get('argos_version', 'unknown')} package {metadata.get('package_version', 'unknown')} (English to Vietnamese)"
    except (OSError, json.JSONDecodeError):
        return "Argos installed package metadata unavailable"


def markdown_report(data: dict) -> str:
    lines = [
        "# Translation model benchmark v1", "",
        "## Scope", "",
        "Seven saved translation units from `real-validation-v5-1/translation-comparison.json`; source text is preserved exactly. Translation outputs are raw model text followed by the existing `deterministic-vi-rules-v2` naturalizer. No production pipeline was run or changed.", "",
        "## Environment and models", "",
        f"- Python: {data['environment']['python']}",
        f"- Platform: {data['environment']['platform']}",
        f"- GPU detected by runtime: {data['environment']['gpu']}",
        f"- NLLB baseline: `{data['candidates'][0]['model']}`; device: {data['candidates'][0]['device']}",
        f"- Local candidate: {data['candidates'][1]['model']}; device: {data['candidates'][1]['device']}",
        f"- LLM provider: skipped — {data['candidates'][2]['reason']}", "",
        "## Candidate runtimes", "",
        "Times are measured in this run and are environment dependent. Total is model initialization plus the sum of attempted unit calls.", "",
        "| Candidate | Status | Load seconds | Translation seconds | Total seconds |", "|---|---:|---:|---:|---:|",
    ]
    for c in data["candidates"]:
        lines.append(f"| {c['name']} | {c['status']} | {c['load_time_seconds']:.3f} | {c['translation_time_seconds']:.3f} | {c['total_time_seconds']:.3f} |")
    lines += ["", "## Unit-by-unit outputs", ""]
    for unit_id in [f"tu-{i:04d}" for i in range(7)]:
        source = next(row["source_text"] for row in data["candidates"][0]["units"] if row["translation_unit_id"] == unit_id)
        lines += [f"### {unit_id}", "", f"**Source ({next(row['source_language'] for row in data['candidates'][0]['units'] if row['translation_unit_id'] == unit_id)}):** {source}", ""]
        for c in data["candidates"]:
            row = next(row for row in c["units"] if row["translation_unit_id"] == unit_id)
            lines.append(f"**{c['name']} — {row['status']}**")
            if row["raw_translation"] is not None:
                lines += [f"- Raw: {row['raw_translation']}", f"- Naturalized: {row['naturalized_translation']}"]
            else:
                lines.append(f"- No output: {row.get('error', c.get('reason', 'unavailable'))}")
        lines.append("")
    lines += [
        "## ASR and proper-noun risks", "",
        "These are source-text review flags only; the harness did not correct or reinterpret them:",
        "- `Juissancei` (tu-0001): possible proper noun or ASR rendering; uncertain.",
        "- `Cuban Japanese` (tu-0003): phrase is unclear in context and may be an ASR or proper-name issue.",
        "- `Bumpal` (tu-0006): uncertain name/term in the source.",
        "- `Takem's Guide to Learning Japanese` (tu-0006): title-like source phrase; preserve and verify against audio if needed.",
        "- `tofugood.com` (tu-0003): spoken domain-like string; verify spelling against audio.",
        "- tu-0000 ends with `in one`; tu-0002 ends mid-thought in Japanese. Both source units are incomplete.", "",
        "## Human review", "",
        "Every candidate/unit row in `benchmark.json` is marked `review_required: true`. Review all successful translations bilingually against the source/audio, with particular attention to the flags above. Argos has no Japanese-to-Vietnamese route in the installed package; those two units are recorded as unsupported, not translated.", "",
        "## Observations", "",
        "- The saved inputs cover English and Japanese source units; NLLB was exercised on all seven. The installed Argos package supports English to Vietnamese only.",
        "- Naturalized text is included for side-by-side review using the existing naturalizer. It is not a quality score and may preserve model errors.",
        "- There is no reference translation set here, and this benchmark does not select a winner.", "",
        "Production recommendation: HUMAN REVIEW REQUIRED", "",
        "Before changing a production provider, review each output against the source and audio, resolve the source-name/ASR risks, confirm quality for both language pairs, and assess latency and memory on the target deployment hardware.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline translation candidate benchmark for the saved seven-unit set.")
    parser.add_argument("--comparison", type=Path, default=REPO_ROOT / "output/real-validation-v5-1/translation-comparison.json")
    parser.add_argument("--translated", type=Path, default=REPO_ROOT / "output/real-validation-v5-1/translated.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/translation-benchmark-v1")
    args = parser.parse_args()

    units = load_units(args.comparison, args.translated)
    naturalizer = DeterministicVietnameseNaturalizer()
    candidates = [
        run_candidate(
            name="NLLB baseline", model="facebook/nllb-200-distilled-600M", device="cpu",
            engine_factory=lambda: __import__("dubflow_worker.translation.nllb", fromlist=["NllbTranslationEngine"]).NllbTranslationEngine(device="cpu"),
            units=units, naturalizer=naturalizer,
        ),
        run_candidate(
            name="Argos local candidate", model=argos_package_version(), device="cpu",
            engine_factory=lambda: __import__("dubflow_worker.translation.argos", fromlist=["ArgosTranslationEngine"]).ArgosTranslationEngine(),
            units=units, naturalizer=naturalizer,
            load_hook=False,
            no_load_status="route_loaded_on_first_use",
        ),
        skipped_candidate(
            "LLM provider", "No compatible LLM translation provider or translation API credentials are configured in this environment.", units,
        ),
    ]
    data = {
        "benchmark_version": "1.0", "input": {
            "comparison_file": str(args.comparison.relative_to(REPO_ROOT)),
            "translated_file": str(args.translated.relative_to(REPO_ROOT)),
            "translation_unit_ids": [unit.unit_id for unit in units],
            "source_text_policy": "Copied unchanged from complete_source_text in comparison file",
        },
        "environment": environment_info(), "candidates": candidates,
        "review_matrix": build_review_matrix(candidates),
        "source_risks": [
            {"text": "Juissancei", "unit_id": "tu-0001", "risk": "Possible ASR or proper-noun issue; uncertain; do not auto-correct."},
            {"text": "Cuban Japanese", "unit_id": "tu-0003", "risk": "Unclear phrase; possible ASR or proper-name issue; do not auto-correct."},
            {"text": "Bumpal", "unit_id": "tu-0006", "risk": "Uncertain name/term; preserve source wording."},
            {"text": "Takem's Guide to Learning Japanese", "unit_id": "tu-0006", "risk": "Title-like phrase; preserve and verify against audio if necessary."},
            {"text": "tofugood.com", "unit_id": "tu-0003", "risk": "Domain-like phrase; spelling can only be verified against audio."},
        ],
        "automatic_quality_scoring": False,
        "automatic_winner_selection": False,
        "production_recommendation": "HUMAN REVIEW REQUIRED",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "benchmark-report.md").write_text(markdown_report(data), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'benchmark.json'}")
    print(f"Wrote {args.output_dir / 'benchmark-report.md'}")
    for candidate in candidates:
        successes = sum(unit["status"] == "success" for unit in candidate["units"])
        failures = sum(unit["status"] == "failed" for unit in candidate["units"])
        print(f"{candidate['name']}: {candidate['status']}; success={successes}; failed={failures}; total={candidate['total_time_seconds']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
