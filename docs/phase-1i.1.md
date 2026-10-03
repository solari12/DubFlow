# Phase 1I.1: Translation source integrity and canonical mapping

Phase 1I.1 is a translation-only correction pass. It does not change ASR, diarization, the Phase 1H timeline, rendering, TTS, Java/React, or product architecture.

## Translation units

Each unit records its stable ID, ordered source segment IDs, full joined source text, per-unit language and speaker IDs. The builder validates that every input segment appears exactly once, in deterministic order; that source text and model input preserve the complete ordered source; and that speaker boundaries are respected. Grouping requires a same-speaker and same-language continuation cue, an incomplete source ending, or an explicit discourse continuation. Adjacent timestamps alone do not group speech.

NLLB receives the complete unit source first. If a multi-segment unit consists of complete ASR segments, the result is composed from translations of those original source segments, with sentence separators, so a long-context decode cannot silently omit a later source segment. Both the full-context result and component translations are retained for review. Incomplete fragments continue to use the full-context result.

## Naturalization and review

The deterministic Vietnamese naturalizer has explicit fixture regressions for the previously malformed phrasing. It leaves ambiguous output literal and sets `review_required`; unchanged text is also flagged. A heuristic rewrite is never treated as a translation quality score, and the report always states that human review is required.

## Mapping

`translated.json.translation_units` contains the canonical source and target text exactly once. `translated.json.segments` contains only the source segment ID, timing, speaker, and translation-unit reference. Do not concatenate segment rows or treat them as per-segment TTS input.

`translation-comparison.json` is unit-level and includes the complete source, full-context literal result, composed literal result, final result, and review reason. `validation-report.json` includes exact source coverage and the Phase 1I.1 regression checks.

## Real-source validation

Run against the saved Phase 1I v5 ASR and diarization artifacts:

```powershell
worker/.venv-diarization/Scripts/python.exe worker/scripts/validate_translation_quality.py `
  input/real_test/sample.mp4 `
  --output-dir output/real-validation-v5-1 `
  --reuse-upstream --offline-models
```

The validation command does not invoke TTS. Passing structural checks means that source segments are accounted for and canonical text is not duplicated in segment rows. It does not establish translation accuracy or make the output ready for TTS benchmarking.
