from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKER_ROOT.parent
sys.path.insert(0, str(WORKER_ROOT / "src"))

from dubflow_worker.review.translation_review import (  # noqa: E402
    assert_review_valid,
    build_translation_review,
    render_translation_review_markdown,
    validate_translation_review,
)


def main() -> int:
    source_path = REPO_ROOT / "output/real-validation-v6/translation-comparison.json"
    transcript_path = REPO_ROOT / "output/real-validation-v6/transcript.json"
    output_dir = REPO_ROOT / "output/real-validation-v6-review"
    comparison = json.loads(source_path.read_text(encoding="utf-8"))
    transcript_bytes = transcript_path.read_bytes()
    transcript = json.loads(transcript_bytes.decode("utf-8"))
    transcript_hash = hashlib.sha256(transcript_bytes).hexdigest()

    review = build_translation_review(comparison)
    report = validate_translation_review(comparison, review, transcript, transcript_hash)
    assert_review_valid(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "translation-review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "translation-review.md").write_text(
        render_translation_review_markdown(review, report), encoding="utf-8"
    )
    (output_dir / "validation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **report}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
