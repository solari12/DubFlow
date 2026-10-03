# Phase 1I: Context-aware translation quality experiment

Phase 1I evaluates translation quality without running TTS or changing Phase 1H alignment/rendering. It retains the original transcript segment IDs, times, and speaker assignments.

## Why the previous translation needed review

The v4 transcript has Japanese and English speech, but ASR reports one global language (`ja`). The Phase 1C/1G path used that language for every segment and sent each ASR segment independently to NLLB. On the real fixture, this produced obvious literal or wrong-language output such as “và lỗi tốt” for “good blogs and errors” and “Tôi đã làm một số người ở Nhật Bản” for “I did Cuban Japanese”. ASR also ended some fragments mid-thought. A successful provider call did not indicate that the Vietnamese was natural or complete.

## Implementation

`translation/units.py` deterministically groups adjacent segments only when they share a speaker and inferred language, have a small timestamp gap, and show an incomplete ending or continuation lead-in. Obvious Japanese-script and Latin-script segments route independently to `ja` and `en`; otherwise the ASR language remains the fallback. Speaker or language changes are hard boundaries. Unit IDs and source segment IDs are stable.

Each unit is sent to NLLB once. A configurable glossary (`worker/config/translation-glossary.json`) masks terms before translation and restores their configured spelling afterward. A provider-neutral `NaturalizationProvider` interface follows translation; the current local implementation applies a small deterministic set of Vietnamese phrase rewrites and leaves unmatched text unchanged. It is intentionally conservative, not a fluent generative writer.

`translated.json` carries unit-level canonical results plus source-segment mappings. Incomplete source tails that have no same-speaker, same-language continuation keep their literal translation for review, but the final rewrite is withheld rather than guessing an ending. The canonical complete unit result is in `translation_units`; segment rows repeat that contextual value for review and must not be concatenated or sent directly to TTS. `translation-comparison.json` compares the old v4 per-segment literal baseline with each context-aware final result.

## Run

```powershell
worker/.venv-diarization/Scripts/python.exe worker/scripts/validate_translation_quality.py `
  input/real_test/sample.mp4 `
  --output-dir output/real-validation-v5 `
  --diarization-device cpu --provider nllb `
  --model facebook/nllb-200-distilled-600M --device cpu --offline-models
```

After successful ASR and diarization have been saved, rerun just the translation stage with `--reuse-upstream --offline-models` to avoid repeating inference and to require local model cache access.

The runner executes base ASR in its existing CUDA environment, then Community-1 diarization in its existing CPU configuration, speaker merge, and NLLB translation/naturalization. ASR and diarization are separate child processes. It writes `transcript.json`, `translated.json`, `translation-comparison.json`, and `validation-report.json`; it does not run TTS or rendering.

## Real-source results

The benchmark ran on `input/real_test/sample.mp4`. Base ASR and Community-1 diarization were executed and saved before the NLLB cache-only retry; the successful report reuses those exact upstream JSON outputs and reruns speaker merge. NLLB loaded from the local Hugging Face cache, with network access disabled.

| Measurement | Result |
| --- | ---: |
| Source duration / ASR segments | 60.075 s / 11 |
| Translation units | 7 |
| Overall ASR language / per-unit routing | ja / Japanese and English |
| Provider / target | facebook/nllb-200-distilled-600M / vi |
| Provider-translated units / segments with literal output | 7 / 11 |
| Segments with final mapped text | 9 |
| Naturalized units | 1 |
| Glossary terms preserved | 7 |
| Contextual segments mapped | 6 (IDs 5–10) |
| Incomplete fragments resolved / unresolved | 0 / 2 (IDs 0, 2) |
| ASR / diarization / speaker merge | 8.652 s / 187.236 s CPU / 0.000058 s |
| NLLB load / grouped translation and rewrite | 14.275 s / 28.648 s |
| Summed end-to-end stage runtime | 238.811 s |
| Translation provider errors | 0 |
| Human review required / quality claimed | yes / no |

The translation-only cache-reuse command took 42.937 s wall time. The report's 238.811 s sums the recorded ASR and diarization timings with merge and translation, even though those two upstream artifacts were reused on the final translation retry. All artifacts are under `output/real-validation-v5/`.

The output explicitly distinguishes translated units from segment rows: each member row repeats its complete unit translation for context; `translation_units` holds the canonical text. Do not concatenate mapped rows. Two incomplete fragments have literal MT output but no final rewrite because there is no safe same-speaker, same-language continuation. One grouped NLLB result (unit `tu-0005`, source IDs 5–7) omitted the latter source clauses; the report does not infer quality from successful inference, and this omission is a required human-review finding.

## Representative comparison

The final text below is the unit result and is repeated for every member segment. These are comparison samples, not endorsements of quality.

| Segment(s) | Source excerpt | Previous literal output | Phase 1I unit output |
| --- | --- | --- | --- |
| 0 | I've been studying Japanese… exactly in one | Tôi đã học tiếng Nhật trong bảy năm, và hãy để tôi dạy bạn làm thế nào để làm nó chính xác trong một | Withheld; incomplete source tail |
| 1 | Juissanceiの時に日本語勉強はじめた… | Tôi bắt đầu học tiếng Nhật khi tôi ở Juissancei, nhưng điều đầu tiên tôi làm là ứng dụng. | Tôi bắt đầu học tiếng Nhật khi tôi ở Juissancei, nhưng điều đầu tiên tôi làm là ứng dụng. |
| 2 | とりあえずアップストアのアプリを全部… | Và tôi đã download tất cả các ứng dụng trên App Store… | Withheld; incomplete source tail |
| 3 | So pick one or a couple, I did Cuban Japanese… | Tôi đã làm một số người ở Nhật Bản… | Vì vậy chọn một hoặc hai, tôi đã làm Cuban Japanese, và tôi cũng rất khuyên bạn nên tofugood.com |
| 4 | They have a lot of good blogs and errors… | Họ có rất nhiều blog và lỗi tốt… nếu điều đó có ý nghĩa | Họ có rất nhiều blog và lỗi tốt… nếu bạn hiểu ý tôi |
| 5–7 | The first thing… Hiragana…; So these are…; and then as you keep learning… | Three separate baseline translations | Điều đầu tiên bạn muốn làm là học Hiragana Katakana và một chút Kanji (later clauses omitted by NLLB) |
| 8–10 | …Bumpal…; That's where I recommend…; and using that… | Three separate baseline translations | và sớm bạn sẽ thấy Bumpal… để là một chút nhầm lẫn… |

## Known bad output still present

- `tu-0005` demonstrates that a larger context call can omit source clauses; grouping is not a guarantee of complete translation.
- “blog và lỗi tốt” is still an inaccurate/literal rendering of “good blogs and errors”.
- “để là một chút nhầm lẫn” and “thực hiện [a] guide” remain unnatural in `tu-0006`.
- “Cuban Japanese”, “Juissancei”, and “Bumpal” may be ASR errors. The glossary intentionally preserves configured spellings and cannot verify names.
- Two incomplete fragments (IDs 0 and 2) remain unresolved. Their final Vietnamese values are withheld instead of inventing endings.

This run proves the pipeline can generate deterministic context units, preserve configured terms, produce a separate rewrite stage, and emit a reviewable report. It does **not** validate Vietnamese translation quality. Manual bilingual review is the next required step.

## Quality limitations

The deterministic naturalizer is deliberately small and can only fix a few known surface constructions; it cannot correct ASR mistakes or reliably resolve idioms and ambiguous names. The source fixture is mixed-language while ASR exposes one global language, so script-based segment language detection is only a heuristic. The default glossary includes terms seen in the fixture; users should review and adjust those entries because it can preserve misspelled ASR names as though they were correct. Units crossing speaker or language boundaries are not merged. Any incomplete fragment without safe matching context is explicitly flagged and has no final rewrite.

The comparison is for human review. No subjective review, translation-accuracy score, or naturalness claim is produced by this benchmark.
