from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dubflow_worker.pipeline.speaker_transcription import merge_speaker_transcript
from dubflow_worker.pipeline.translate_quality import translate_contextual_transcript
from dubflow_worker.translation.factory import create_translation_engine
from dubflow_worker.translation.glossary import Glossary
from dubflow_worker.translation.naturalization import DeterministicVietnameseNaturalizer


ROOT = Path(__file__).resolve().parents[2]
ASR_SCRIPT = ROOT / "worker" / "scripts" / "benchmark_asr.py"
DIARIZATION_SCRIPT = ROOT / "worker" / "scripts" / "benchmark_diarization.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-source Phase 1I translation-quality validation")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asr-python", type=Path, default=ROOT / "worker" / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--diarization-python", type=Path, default=ROOT / "worker" / ".venv-diarization" / "Scripts" / "python.exe")
    parser.add_argument("--diarization-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--provider", choices=("nllb", "argos"), default="nllb")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-language", default="vi")
    parser.add_argument("--reuse-upstream", action="store_true", help="reuse saved v5 ASR and diarization outputs")
    parser.add_argument("--offline-models", action="store_true", help="require Hugging Face models to come from local cache")
    parser.add_argument("--glossary", type=Path, default=ROOT / "worker" / "config" / "translation-glossary.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "output" / "real-validation-v4" / "translated.json")
    return parser


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _subprocess(command: list[str], env: dict[str, str]) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    return process.returncode, process.stdout, process.stderr, round(time.perf_counter() - started, 3)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "asr" / "transcripts").mkdir(parents=True, exist_ok=True)
    (output / "diarization").mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "version": "1.0",
        "success": False,
        "overall_result": "FAIL",
        "source": str(args.input),
        "source_duration_seconds": None,
        "segment_count": 0,
        "translation_unit_count": 0,
        "review_required_unit_count": 0,
        "source_coverage": None,
        "regression_checks": {},
        "regression_checks_passed": False,
        "source_language": None,
        "target_language": args.target_language,
        "provider": args.provider,
        "model": args.model if args.provider == "nllb" else None,
        "segments_translated": 0,
        "segments_with_literal_translation": 0,
        "units_translated": 0,
        "units_naturalized": 0,
        "glossary_terms_configured": [],
        "glossary_terms_preserved": [],
        "incomplete_segments_contextualized": [],
        "incomplete_segments_unresolved": [],
        "contextual_segments_handled": [],
        "review_required_units": [],
        "runtime_seconds": None,
        "stages": {},
        "errors": {},
        "subjective_human_review_required": True,
        "translation_quality_claimed": False,
        "segment_translation_mapping_semantics": "",
    }
    started = time.perf_counter()
    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"Input media does not exist: {args.input}")
        asr_report_path = output / "asr-benchmark.json"
        asr_transcripts = output / "asr" / "transcripts"
        diar_report_path = output / "diarization" / "diarization-benchmark.json"
        if args.reuse_upstream:
            if not asr_report_path.is_file() or not diar_report_path.is_file():
                raise FileNotFoundError("--reuse-upstream requires saved asr-benchmark.json and diarization report")
            asr_transcript_path = asr_transcripts / f"{args.input.stem}-base.json"
            asr_report = _read(asr_report_path)
            diarization = _read(diar_report_path)
            asr_case = asr_report["cases"][0]
            asr_result = asr_case["results"][0]
            report["stages"]["asr"] = {
                "runtime_seconds": asr_result.get("processing_seconds"),
                "reused_saved_result": True,
                "success": asr_result.get("success"),
            }
            report["stages"]["diarization"] = {
                "runtime_seconds": diarization.get("total_runtime_seconds"),
                "device": diarization.get("environment", {}).get("device_requested"),
                "reused_saved_result": True,
                "success": diarization.get("success"),
            }
            if not asr_result.get("success") or not diarization.get("success"):
                raise RuntimeError("Saved ASR or diarization result was not successful")
        else:
            if not args.asr_python.is_file() or not args.diarization_python.is_file():
                raise FileNotFoundError("ASR and diarization Python environments must exist")
            asr_env = os.environ.copy()
            source_root = str(ROOT / "worker" / "src")
            asr_env["PYTHONPATH"] = source_root + os.pathsep + asr_env.get("PYTHONPATH", "")
            asr_cmd = [
                str(args.asr_python), str(ASR_SCRIPT), str(args.input.resolve()),
                "--models", "base", "--device", "cuda", "--compute-type", "float16",
                "--transcripts-dir", str(asr_transcripts), "--output", str(asr_report_path),
            ]
            code, stdout, stderr, elapsed = _subprocess(asr_cmd, asr_env)
            report["stages"]["asr"] = {"runtime_seconds": elapsed, "return_code": code, "stderr": stderr.strip() or None}
            if code != 0 or not asr_report_path.is_file():
                raise RuntimeError("ASR failed: " + (stderr.strip() or stdout.strip() or "report not produced"))
            asr_report = _read(asr_report_path)
            asr_case = asr_report["cases"][0]
            asr_result = asr_case["results"][0]
            if not asr_result.get("success"):
                raise RuntimeError("ASR failed: " + str(asr_result.get("error")))
            asr_transcript_path = Path(asr_result["transcript_file"])
            asr_env = os.environ.copy()
            diar_cmd = [
                str(args.diarization_python), str(DIARIZATION_SCRIPT), str(args.input.resolve()),
                "--model", "pyannote/speaker-diarization-community-1",
                "--device", args.diarization_device,
                "--min-speakers", "2", "--max-speakers", "2", "--output", str(diar_report_path),
            ]
            code, stdout, stderr, elapsed = _subprocess(diar_cmd, asr_env)
            report["stages"]["diarization"] = {"runtime_seconds": elapsed, "return_code": code, "stderr": stderr.strip() or None}
            if code != 0 or not diar_report_path.is_file():
                raise RuntimeError("Diarization failed: " + (stderr.strip() or stdout.strip() or "report not produced"))
            diarization = _read(diar_report_path)
            if not diarization.get("success"):
                raise RuntimeError("Diarization failed: " + str(diarization.get("error")))
        asr_report = _read(asr_report_path)
        asr_transcript = _read(asr_transcript_path)

        merge_started = time.perf_counter()
        merged = merge_speaker_transcript(
            asr_transcript.get("segments", []), diarization["result"].get("segments", []), threshold=0.20
        )
        merge_seconds = round(time.perf_counter() - merge_started, 6)
        detected = asr_transcript.get("asr", {}).get("language")
        speaker_transcript = {
            "version": "1.0",
            "source": asr_transcript.get("source", {}),
            "asr": asr_transcript.get("asr", {}),
            "detected_language": detected,
            "diarization": {
                "engine": "pyannote.audio",
                "model": "pyannote/speaker-diarization-community-1",
                "device": args.diarization_device,
            },
            "speakers": diarization["result"].get("speakers", []),
            "speaker_assignment_threshold": 0.20,
            "segments": merged,
        }
        transcript_path = output / "transcript.json"
        _write(transcript_path, speaker_transcript)

        glossary = Glossary.from_json(args.glossary)
        baseline: dict[int, dict[str, Any]] = {}
        if args.baseline.is_file():
            baseline = {int(row["id"]): row for row in _read(args.baseline).get("segments", [])}
        if args.offline_models:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        engine = create_translation_engine(args.provider, model=args.model, device=args.device)
        load_started = time.perf_counter()
        if hasattr(engine, "load"):
            engine.load()
        load_seconds = round(time.perf_counter() - load_started, 6)
        translation_started = time.perf_counter()
        translated = translate_contextual_transcript(
            speaker_transcript,
            engine=engine,
            naturalizer=DeterministicVietnameseNaturalizer(),
            target_language=args.target_language,
            glossary=glossary,
            baseline_segments=baseline,
        )
        translation_seconds = round(time.perf_counter() - translation_started, 6)
        translated_path = output / "translated.json"
        _write(translated_path, translated)
        comparison = [
            {
                "translation_unit_id": unit["translation_unit_id"],
                "source_segment_ids": unit["source_segment_ids"],
                "complete_source_text": unit["source_text"],
                "context_translation": unit["context_translation"],
                "source_character_count": unit["source_character_count"],
                "literal_translation": unit["literal_translation"],
                "final_translation": unit["final_translation"],
                "review_required": unit["review_required"],
                "review_reason": unit["review_reason"],
            }
            for unit in translated["translation_units"]
        ]
        _write(output / "translation-comparison.json", comparison)

        quality = translated["quality"]
        units_by_id = {unit["translation_unit_id"]: unit for unit in translated["translation_units"]}
        unit5 = units_by_id.get("tu-0005")
        unit6 = units_by_id.get("tu-0006")
        segment_to_unit = {
            segment_id: unit["translation_unit_id"]
            for unit in translated["translation_units"]
            for segment_id in unit["source_segment_ids"]
        }
        malformed_patterns = (
            "\u0111\u1ec3 l\u00e0 m\u1ed9t ch\u00fat nh\u1ea7m l\u1eabn",
            "th\u1ef1c hi\u1ec7n takem",
            "blog v\u00e0 l\u1ed7i t\u1ed1t",
            "l\u00e0 m\u1ed9t ch\u00fat nh\u1ea7m l\u1eabn",
            "tri\u1ec3n khai Takem",
            "\u0111\u00f3 l\u00e0 t\u00e0i li\u1ec7u mi\u1ec5n ph\u00ed n\u00e0y",
        )
        regression_checks = {
            "tu_0005_retains_segments_5_6_7": bool(
                unit5 and unit5["source_segment_ids"] == [5, 6, 7]
                and len(unit5["segment_literal_translations"]) == 3
                and unit5["translation_method"] == "source_segment_composition"
            ),
            "tu_0006_has_no_known_malformed_phrase": bool(
                unit6 and unit6["final_translation"]
                and not any(pattern in unit6["final_translation"].casefold() for pattern in malformed_patterns)
            ),
            "incomplete_segments_0_and_2_reviewed_at_source_boundaries": all(
                segment_id in segment_to_unit
                and len(units_by_id[segment_to_unit[segment_id]]["source_segment_ids"]) == 1
                and units_by_id[segment_to_unit[segment_id]]["review_required"]
                for segment_id in (0, 2)
            ),
            "no_source_segment_silently_dropped": translated["source_coverage"]["each_segment_assigned_exactly_once"],
            "canonical_translation_not_duplicated_in_segment_rows": all(
                "final_translation" not in row and "literal_translation" not in row
                for row in translated["segments"]
            ),
        }
        regression_checks_passed = all(regression_checks.values())
        total_pipeline_runtime = round(
            float(report["stages"]["asr"].get("runtime_seconds") or 0)
            + float(report["stages"]["diarization"].get("runtime_seconds") or 0)
            + merge_seconds
            + load_seconds
            + translation_seconds,
            3,
        )
        report.update(
            success=translated["success"] and regression_checks_passed,
            overall_result="COMPLETED_REVIEW_REQUIRED" if translated["success"] and regression_checks_passed else "FAIL",
            source_duration_seconds=asr_transcript.get("source", {}).get("duration"),
            segment_count=quality["input_segment_count"],
            translation_unit_count=quality["translation_unit_count"],
            source_language=detected,
            provider=engine.name,
            device=engine.device,
            segments_translated=quality["translated_segment_count"],
            segments_with_literal_translation=quality["translated_segment_count"],
            units_translated=sum(unit["literal_translation"] is not None for unit in translated["translation_units"]),
            units_naturalized=quality["naturalized_unit_count"],
            glossary_terms_configured=list(glossary.terms),
            glossary_terms_preserved=quality["glossary_terms_preserved"],
            incomplete_segments_contextualized=quality["contextual_segment_ids"],
            incomplete_segments_unresolved=quality["unresolved_incomplete_segments"],
            contextual_segments_handled=quality["contextual_segment_ids"],
            review_required_units=quality["review_required_units"],
            review_required_unit_count=quality["review_required_unit_count"],
            source_coverage=translated["source_coverage"],
            regression_checks=regression_checks,
            regression_checks_passed=regression_checks_passed,
            segment_translation_mapping_semantics=translated["translation"]["segment_mapping"],
            runtime_seconds=total_pipeline_runtime,
            benchmark_execution_wall_seconds=round(time.perf_counter() - started, 3),
            stages={
                **report["stages"],
                "speaker_merge": {"runtime_seconds": merge_seconds, "segments": len(merged)},
                "translation_model_load": {"runtime_seconds": load_seconds},
                "context_translation_and_naturalization": {"runtime_seconds": translation_seconds},
            },
            errors={
                "asr": None,
                "diarization": None,
                "translation_units": [
                    {"translation_unit_id": unit["translation_unit_id"], "error": unit["translation_error"]}
                    for unit in translated["translation_units"] if unit["translation_error"]
                ],
            },
        )
    except Exception as exc:
        report["errors"]["pipeline"] = f"{type(exc).__name__}: {exc}"
        report["runtime_seconds"] = round(time.perf_counter() - started, 3)
    report["subjective_human_review_required"] = True
    report["translation_quality_claimed"] = False
    _write(output / "validation-report.json", report)
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
