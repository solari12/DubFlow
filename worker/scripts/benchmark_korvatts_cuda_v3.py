from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1]
REPO = WORKER.parent
sys.path.insert(0, str(WORKER / "src"))

from dubflow_worker.tts.korvatts_cuda_benchmark_v3 import (  # noqa: E402
    VOICE,
    STEPS,
    SPEED,
    build_quality_review,
    render_report,
    run_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark-only KorvaTTS CUDA evaluation using the existing Phase 1K.4 fixture")
    parser.add_argument("--fixture", type=Path, default=REPO / "output/tts-benchmark-v1/fixture.json")
    parser.add_argument("--output", type=Path, default=REPO / "output/tts-benchmark-v3/korvatts-cuda")
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--voice", default=VOICE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--speed", type=float, default=SPEED)
    args = parser.parse_args(argv)

    report = run_benchmark(
        fixture_path=args.fixture,
        output_dir=args.output,
        repo=REPO,
        assets_dir=args.assets_dir,
        voice=args.voice,
        steps=args.steps,
        speed=args.speed,
    )
    report_path = args.output / "benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "benchmark-report.md").write_text(render_report(report), encoding="utf-8")
    (args.output / "quality-review.md").write_text(build_quality_review(report["cases"]), encoding="utf-8")
    listening_sheet = {
        "phase": "1K.4.3",
        "status_policy": "Every generated listening case begins as NOT_REVIEWED; edit only after human listening.",
        "items": [
            {
                "case": case["case"],
                "voice": args.voice,
                "status": "NOT_REVIEWED",
                "output_wav": case.get("output_wav"),
                "wav_validation": case.get("wav_validation", "NOT_RUN"),
                "notes": "",
            }
            for case in report["cases"]
        ],
    }
    (args.output / "listening-sheet.json").write_text(json.dumps(listening_sheet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cuda_execution_verification": report["cuda_execution_verification"].get("status"), "cuda_provider": report.get("execution_providers"), "success_count": report["success_count"], "failure_count": report["failure_count"], "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "CUDA_SUPPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
