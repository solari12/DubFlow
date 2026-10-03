from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

EXPECTED_UNIT_IDS = [f"tu-{index:04d}" for index in range(7)]
NATURALIZER_NAME = "deterministic-vi-rules-v2"


@dataclass(frozen=True)
class BenchmarkUnit:
    unit_id: str
    source_segment_ids: list[int]
    source_text: str
    source_language: str
    source_character_count: int


def load_units(comparison_path: Path, translated_path: Path) -> list[BenchmarkUnit]:
    """Load exactly the seven saved comparison rows, joining language by stable unit ID."""
    comparison = json.loads(comparison_path.read_text(encoding="utf-8-sig"))
    translated = json.loads(translated_path.read_text(encoding="utf-8-sig"))
    rows = comparison if isinstance(comparison, list) else comparison.get("units", [])
    translated_units = translated.get("translation_units", [])
    languages = {row["translation_unit_id"]: row["source_language"] for row in translated_units}
    ids = [row.get("translation_unit_id") for row in rows]
    if ids != EXPECTED_UNIT_IDS:
        raise ValueError(f"Expected the seven ordered IDs {EXPECTED_UNIT_IDS}; got {ids}")
    result: list[BenchmarkUnit] = []
    for row in rows:
        unit_id = row["translation_unit_id"]
        source = row.get("complete_source_text")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Missing exact complete_source_text for {unit_id}")
        if unit_id not in languages:
            raise ValueError(f"Missing source_language metadata for {unit_id}")
        result.append(BenchmarkUnit(
            unit_id=unit_id,
            source_segment_ids=list(row.get("source_segment_ids", [])),
            source_text=source,
            source_language=languages[unit_id],
            source_character_count=len(source),
        ))
    return result


def run_candidate(
    *, name: str, engine_factory: Callable[[], Any], units: Iterable[BenchmarkUnit],
    naturalizer: Any, model: str, device: str, load_hook: bool = True,
    no_load_status: str = "initialized_eagerly",
) -> dict[str, Any]:
    units = list(units)
    load_start = time.perf_counter()
    try:
        engine = engine_factory()
        if load_hook and callable(getattr(engine, "load", None)):
            engine.load()
            load_status = "loaded"
        else:
            load_status = no_load_status
        load_time = time.perf_counter() - load_start
    except Exception as exc:  # keep candidate failures inspectable, and continue other candidates
        load_time = time.perf_counter() - load_start
        return {
            "name": name, "model": model, "device": device, "status": "failed",
            "model_load_status": "failed", "load_time_seconds": round(load_time, 3),
            "translation_time_seconds": 0.0, "total_time_seconds": round(load_time, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "units": [failed_unit(unit, f"Model load failed: {type(exc).__name__}: {exc}", "failed") for unit in units],
        }

    results = []
    translation_total = 0.0
    for unit in units:
        started = time.perf_counter()
        try:
            engine.validate_language_pair(unit.source_language, "vi")
            raw = engine.translate_for_dubbing(unit.source_text, unit.source_language, "vi")
            elapsed = time.perf_counter() - started
            translation_total += elapsed
            naturalized = naturalizer.naturalize(raw, language="vi", source_context=unit.source_text)
            results.append({
                **unit_dict(unit), "status": "success", "attempted": True,
                "model_load_status": load_status, "translation_time_seconds": round(elapsed, 3),
                "raw_translation": raw, "naturalized_translation": naturalized,
                "naturalizer": NATURALIZER_NAME, "error": None,
            })
        except Exception as exc:
            elapsed = time.perf_counter() - started
            translation_total += elapsed
            results.append(failed_unit(unit, f"{type(exc).__name__}: {exc}", load_status, elapsed))
    status = "completed_with_failures" if any(row["status"] == "failed" for row in results) else "completed"
    return {
        "name": name, "model": model, "device": device, "status": status,
        "model_load_status": load_status, "load_time_seconds": round(load_time, 3),
        "translation_time_seconds": round(translation_total, 3),
        "total_time_seconds": round(load_time + translation_total, 3), "units": results,
    }


def unit_dict(unit: BenchmarkUnit) -> dict[str, Any]:
    return {
        "translation_unit_id": unit.unit_id,
        "source_segment_ids": unit.source_segment_ids,
        "source_text": unit.source_text,
        "source_language": unit.source_language,
        "target_language": "vi",
        "source_character_count": unit.source_character_count,
    }


def failed_unit(unit: BenchmarkUnit, error: str, load_status: str, elapsed: float = 0.0) -> dict[str, Any]:
    return {
        **unit_dict(unit), "status": "failed", "attempted": True,
        "model_load_status": load_status, "translation_time_seconds": round(elapsed, 3),
        "raw_translation": None, "naturalized_translation": None,
        "naturalizer": NATURALIZER_NAME, "error": error,
    }


def skipped_candidate(name: str, reason: str, units: Iterable[BenchmarkUnit]) -> dict[str, Any]:
    return {
        "name": name, "model": None, "device": None, "status": "skipped",
        "model_load_status": "skipped", "load_time_seconds": 0.0,
        "translation_time_seconds": 0.0, "total_time_seconds": 0.0, "reason": reason,
        "units": [{
            **unit_dict(unit), "status": "skipped", "attempted": False,
            "model_load_status": "skipped", "translation_time_seconds": 0.0,
            "raw_translation": None, "naturalized_translation": None,
            "naturalizer": NATURALIZER_NAME, "error": reason,
        } for unit in units],
    }


def build_review_matrix(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes = {
        "tu-0000": "Source ends with the incomplete phrase ‘in one’.",
        "tu-0001": "Review source name ‘Juissancei’; it may be a proper noun or an ASR rendering.",
        "tu-0002": "Japanese source ends mid-thought; preserve the source as given.",
        "tu-0003": "Review ‘did Cuban Japanese’ and tofugood.com as possible ASR or proper-name risks.",
        "tu-0004": "Review source phrase ‘good blogs and errors’; its intended meaning is unclear.",
        "tu-0005": "Long source unit with multiple clauses and Japanese writing-system names.",
        "tu-0006": "Review ‘Bumpal’ and the title ‘Takem’s Guide to Learning Japanese’; do not auto-correct.",
    }
    matrix = []
    for candidate in candidates:
        for output in candidate["units"]:
            matrix.append({
                "candidate": candidate["name"],
                "translation_unit_id": output["translation_unit_id"],
                "status": output["status"],
                "raw_translation": output["raw_translation"],
                "naturalized_translation": output["naturalized_translation"],
                "review_required": True,
                "reviewer_notes": notes[output["translation_unit_id"]],
            })
    return matrix


def environment_info() -> dict[str, str]:
    gpu = "Not queried"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        else:
            gpu = "CUDA unavailable to benchmark runtime"
    except Exception as exc:
        gpu = f"Unavailable ({type(exc).__name__})"
    return {"python": platform.python_version(), "platform": platform.platform(), "gpu": gpu}
