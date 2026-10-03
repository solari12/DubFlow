from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from dubflow_worker.tts.benchmark_v1 import TTSBenchmarkFixture  # noqa: E402
from dubflow_worker.tts.kokoro_benchmark_v2 import (  # noqa: E402
    MODEL_REPO_ID,
    MODEL_REVISION,
    SAMPLE_RATE,
    SELECTED_VOICES,
    SOURCE_COMMIT,
    SOURCE_REPO,
    KokoroBenchmarkProvider,
    benchmark_voice_device,
    build_quality_review,
    environment_metadata,
    fixture_sha256,
    prepare_model_assets,
    render_benchmark_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark-only Kokoro Vietnamese CPU/CUDA TTS")
    parser.add_argument("--fixture", type=Path, default=REPO / "output/tts-benchmark-v1/fixture.json")
    parser.add_argument("--output", type=Path, default=REPO / "output/tts-benchmark-v2/kokoro")
    parser.add_argument("--voices", nargs="+", choices=tuple(SELECTED_VOICES), default=list(SELECTED_VOICES))
    args = parser.parse_args(argv)

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    hf_home = ROOT / ".model-cache/kokoro"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TORCH_HOME"] = str(hf_home / "torch")

    before_path = output_root / "environment-before.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.is_file() else None
    environment = environment_metadata(before=before)
    (output_root / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fixture = TTSBenchmarkFixture.load(args.fixture)
    fixture_data = json.loads(args.fixture.read_text(encoding="utf-8"))
    asset_error = None
    assets_dir = hf_home / "missing-model-assets"
    resolved_revision = None
    try:
        assets_dir, resolved_revision = prepare_model_assets(hf_home / "hub", args.voices)
    except Exception as exc:
        asset_error = f"{type(exc).__name__}: {exc}"

    def provider_factory(**kwargs):
        if asset_error:
            raise RuntimeError(f"Pinned Kokoro model assets could not be loaded: {asset_error}")
        return KokoroBenchmarkProvider(**kwargs)

    results = []
    started = time.perf_counter()
    for device in ("cpu", "cuda"):
        for voice in args.voices:
            result = benchmark_voice_device(
                fixture=fixture,
                device=device,
                voice=voice,
                output_dir=output_root / device / voice,
                assets_dir=assets_dir,
                provider_factory=provider_factory,
            )
            results.append(result)
            try:
                import torch

                if device == "cuda":
                    del result
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass

    env_after = environment_metadata(before=before)
    (output_root / "environment.json").write_text(json.dumps(env_after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    existing = _existing_candidate_results(REPO / "output/tts-benchmark-v1")
    expected_combinations = len(args.voices) * 2
    successful_combinations = sum(
        result["initialization_status"] == "LOADED" and all(case.get("status") == "SUCCESS" for case in result["cases"])
        for result in results
    )
    successful_wav_count = sum(
        len(case.get("measured_runs", [])) if case.get("case") == "original" else int(bool(case.get("file_size_bytes")))
        for result in results
        for case in result.get("cases", [])
    )
    report = {
        "benchmark_version": "2.0",
        "phase": "1K.4.2",
        "status": "COMPLETED" if successful_combinations == expected_combinations else "COMPLETED_WITH_FAILURES",
        "provider": "Kokoro Vietnamese (benchmark-only)",
        "model": {
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "resolved_revision": resolved_revision,
            "pytorch_checkpoint": "kokoro_vi.pth",
            "sample_rate_hz": SAMPLE_RATE,
            "package_name": "kokoro-vietnamese",
            "package_version": env_after["packages"].get("kokoro-vietnamese") or "unavailable",
            "source_repository": SOURCE_REPO,
            "source_commit": SOURCE_COMMIT,
            "asset_download_error": asset_error,
            "inference_backend": "PyTorch (CUDA and CPU device selection is explicit; CUDA requests are rejected if unavailable).",
        },
        "environment": env_after,
        "fixture": {
            "path": str(args.fixture.resolve()),
            "sha256": fixture_sha256(args.fixture),
            "version": fixture_data["version"],
            "original_text": fixture.original_text,
            "normalized_text": fixture.normalized_text,
            "case_names": list(fixture.cases()),
            "modified": False,
        },
        "voices": [
            {"voice": voice, **SELECTED_VOICES[voice]}
            for voice in args.voices
        ],
        "devices": ["cpu", "cuda"],
        "results": results,
        "successful_wav_count": successful_wav_count,
        "benchmark_runtime_seconds": time.perf_counter() - started,
        "existing_candidates": existing,
        "automatic_quality_score": False,
        "automatic_winner_selection": False,
        "human_quality_review": "NOT_REVIEWED",
        "production_changed": False,
        "limitations": [
            "Only three selected voice identities and five short fixture cases were evaluated.",
            "Whole-device VRAM samples are coarse and are not process-specific peak memory.",
            "Windows process peak working set is cumulative across this benchmark process.",
            "Quality and voice suitability require human listening; no automated score or winner is generated.",
        ],
    }
    (output_root / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "benchmark-report.md").write_text(render_benchmark_report(report), encoding="utf-8")
    (output_root / "quality-review.md").write_text(build_quality_review(results), encoding="utf-8")
    print(json.dumps({"status": report["status"], "results": [{"device": r["device"], "voice": r["voice"], "status": r["status"], "cases": len(r["cases"])} for r in results], "successful_wavs": successful_wav_count}, ensure_ascii=True, indent=2))
    return 0


def _existing_candidate_results(root: Path) -> dict:
    def read(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    piper = read(root / "piper/report.json")
    korva = read(root / "korvatts/report.json")
    korva_cuda = read(root / "korvatts-cuda/report.json")
    piper_original = next((row for row in piper.get("cases", []) if row.get("case") == "original"), {})
    korva_original = next((row for row in korva.get("cases", []) if row.get("case") == "original"), {})
    return {
        "piper_cpu_rtf": piper_original.get("rtf"),
        "piper_report": str(root / "piper/report.json"),
        "korva_cpu_rtf": korva_original.get("rtf"),
        "korva_original_duration_seconds": korva_original.get("duration_seconds"),
        "korva_report": str(root / "korvatts/report.json"),
        "korva_cuda_status": korva_cuda.get("status", "not found"),
        "korva_cuda_error": korva_cuda.get("initialization_error"),
        "korva_cuda_was_oom": False,
        "korva_cuda_report": str(root / "korvatts-cuda/report.json"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
