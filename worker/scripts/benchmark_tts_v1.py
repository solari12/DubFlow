from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from dubflow_worker.tts.benchmark_providers_v1 import KorvaBenchmarkProvider, PiperBenchmarkProvider  # noqa: E402
from dubflow_worker.tts.benchmark_v1 import (  # noqa: E402
    TTSBenchmarkFixture,
    benchmark_provider,
    write_provider_report,
)
from dubflow_worker.tts.cuda_benchmark_v1 import (  # noqa: E402
    render_cuda_benchmark_markdown,
    run_korva_cuda_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline, benchmark-only Vietnamese TTS comparison")
    parser.add_argument("--provider", choices=("all", "piper", "korva"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=REPO / "output/tts-benchmark-v1")
    parser.add_argument("--fixture", type=Path, default=REPO / "output/tts-benchmark-v1/fixture.json")
    parser.add_argument("--piper-model-dir", type=Path, default=ROOT / ".model-cache/tts/vits-piper-vi_VN-vivos-x_low")
    parser.add_argument("--korva-voice", default="khanh_vy")
    parser.add_argument("--korva-steps", type=int, default=32)
    args = parser.parse_args(argv)
    if args.device == "cuda":
        if args.provider != "korva":
            parser.error("CUDA benchmark mode supports only --provider korva; Piper's existing CPU benchmark is unchanged")
        fixture = TTSBenchmarkFixture.load(args.fixture)
        cuda_dir = args.output / "korvatts-cuda"
        cpu_report_path = args.output / "korvatts/report.json"
        report = run_korva_cuda_benchmark(fixture, cuda_dir, cpu_report_path)
        (cuda_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (cuda_dir / "benchmark-report.md").write_text(render_cuda_benchmark_markdown(report), encoding="utf-8")
        print(json.dumps({"provider": "korvatts-cuda", "status": report["status"], "initialization_status": report["initialization_status"], "cases": len(report["cases"])}, ensure_ascii=True, indent=2))
        return 0 if report["status"] == "CUDA_SUPPORTED" else 1
    fixture = TTSBenchmarkFixture.load(args.fixture)
    cases = fixture.cases()
    providers = []
    if args.provider in {"all", "piper"}:
        providers.append(("piper", "sherpa-onnx-piper-vivos-x_low", lambda: PiperBenchmarkProvider(args.piper_model_dir)))
    if args.provider in {"all", "korva"}:
        providers.append(("korvatts", "dogenthq/KorvaTTS", lambda: KorvaBenchmarkProvider(args.korva_voice, args.korva_steps)))

    aggregate_path = args.output / "benchmark-results.json"
    previous = {}
    if aggregate_path.is_file():
        try:
            previous = {item["provider"]: item for item in json.loads(aggregate_path.read_text(encoding="utf-8"))}
        except (ValueError, KeyError, TypeError):
            previous = {}
    reports_by_provider = dict(previous)
    for folder, model_name, factory in providers:
        report = benchmark_provider(
            factory,
            provider_name=folder,
            model_name=model_name,
            cases=cases,
            output_dir=args.output / folder,
        )
        if folder == "korvatts" and report.get("model_size_bytes") is not None:
            report["model_size_note"] = "Installed Python package tree only; Hugging Face model assets may be cached separately."
        write_provider_report(args.output / folder, report)
        reports_by_provider[folder] = report
    reports = list(reports_by_provider.values())
    aggregate_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_human_summary(args.output, reports)
    print(json.dumps([{"provider": r["provider"], "status": r["status"], "initialization_status": r["initialization_status"], "cases": len(r["cases"])} for r in reports], ensure_ascii=True, indent=2))
    return 0 if all(r["status"] == "COMPLETED" for r in reports) else 1


def _write_human_summary(output_dir: Path, reports: list[dict]) -> None:
    piper = next((row for row in reports if row["provider"] == "piper"), None)
    korva = next((row for row in reports if row["provider"] == "korvatts"), None)
    lines = [
        "# Phase 1K.4 TTS Benchmark Report", "",
        "Status: `NEEDS_HUMAN_REVIEW`. This report records controlled local synthesis only; it does not select a winner.", "",
        "## Models", "",
        "| Model | Loading status | Compatibility status | License note |", "|---|---|---|---|",
        f"| Piper `vi_VN-vivos-x_low` (Sherpa-ONNX, CPU) | {_init(piper)} | {_compat(piper)} | VIVOS data license: CC BY-NC-SA 4.0; verify intended-use terms. |",
        f"| KorvaTTS `dogenthq/KorvaTTS` (CPU ONNX) | {_init(korva)} | {_compat(korva)} | Model repository declares Apache-2.0 for weights, voice styles, and audio. |",
        "", "## Installation", "",
        "Piper uses the existing `worker/.venv-tts`; KorvaTTS uses the isolated `worker/.venv-tts-korva`. Candidate package and local model loading succeeded.",
        "", "## Results", "",
        "| Model | Init seconds | Model size | Original duration / RTF | Normalized duration / RTF | Sample rate | Peak process working set | VRAM observation | Pronunciation tests | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for name, row in (("Piper VIVOS x_low", piper), ("KorvaTTS", korva)):
        if row is None:
            lines.append(f"| {name} | Pending | Pending | Pending | Pending | Pending | Pending | NEEDS_HUMAN_REVIEW |")
            continue
        original = _case(row, "original")
        normalized = _case(row, "normalized")
        memory = row.get("peak_process_working_set_mib")
        vram = row.get("vram_measurement", {})
        vram_note = (
            f"CPU; system-wide samples {vram.get('before_mib')}→{vram.get('peak_sampled_mib')}→{vram.get('after_mib')} MiB (not process-specific)"
            if vram.get("available") else "CPU; nvidia-smi unavailable"
        )
        sample_rates = sorted({case.get("sample_rate") for case in row.get("cases", []) if case.get("sample_rate")})
        lines.append(
            f"| {name} | {_fmt(row.get('model_initialization_seconds'))} | {_fmt(row.get('model_size_bytes') / 1_000_000 if row.get('model_size_bytes') else None)} MB | {_duration_rtf(original)} | {_duration_rtf(normalized)} | "
            f"{', '.join(map(str, sample_rates)) or '—'} Hz | {_fmt(memory)} MiB | {vram_note} | {_pron_status(row)} | {_compat(row)} |"
        )
    lines.extend(["", "## Files", "", "Per-provider `report.json` contains all five cases, input text, status/error, WAV path, sample rate, channels, PCM validation, duration, synthesis time, RTF, initialization time, model-size measurement, working-set peak, VRAM sampling method, and warnings.", ""])
    for row in reports:
        lines.append(f"- {row['provider']}: `{row['status']}` — `output/tts-benchmark-v1/{row['provider']}/`")
        for case_name in ("original", "normalized", "isolated-hiragana", "isolated-katakana", "isolated-kanji"):
            case = _case(row, case_name)
            if case and case.get("status") == "SUCCESS":
                lines.append(f"  - `{case['output_wav']}`")
    lines.extend(["", "## Human review", "", "All quality fields start as `NOT_REVIEWED`. Use `quality-review.md` after listening. No numerical score or automatic winner is generated.", ""])
    (output_dir / "benchmark-report.md").write_text("\n".join(lines), encoding="utf-8")


def _init(row: dict | None) -> str:
    return row.get("initialization_status", "PENDING") if row else "PENDING"


def _compat(row: dict | None) -> str:
    if row is None:
        return "NEEDS_HUMAN_REVIEW"
    return "TECHNICALLY_COMPATIBLE" if row.get("status") == "COMPLETED" else "FAILED_GENERATION"


def _case(row: dict, name: str) -> dict | None:
    return next((case for case in row.get("cases", []) if case.get("case") == name), None)


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def _duration_rtf(case: dict | None) -> str:
    if not case or case.get("status") != "SUCCESS":
        return "FAILED"
    return f"{_fmt(case.get('duration_seconds'))}s / {_fmt(case.get('rtf'))}"


def _pron_status(row: dict) -> str:
    names = ("isolated-hiragana", "isolated-katakana", "isolated-kanji")
    cases = [_case(row, name) for name in names]
    if not cases or any(case is None or case.get("status") != "SUCCESS" for case in cases):
        return "FAILED/partial"
    return "WAVs generated; listening required"


if __name__ == "__main__":
    raise SystemExit(main())
