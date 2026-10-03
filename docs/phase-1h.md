# Phase 1H: Non-overlapping dubbing timeline

Phase 1H fixes dialogue spill and restores measurable pauses in the existing Python audio pipeline. It does not change translation models, diarization, voice selection, or the video architecture.

## Root cause

Phase 1E aligned every clip at its transcript start but kept overlong TTS audio under the default `preserve` policy. `_mix_timeline` extended the output to contain that full audio while later clips kept their original start times. Their samples were summed together, and global peak protection only lowered the mixed level. ASR segment boundaries also touched throughout the real fixture, even where the source waveform had quiet pauses, so timestamp-only placement played adjacent TTS clips back to back.

## Timeline plan and fitting

Before aligning audio, the worker now creates a plan for each segment with its ID, speaker, source interval, translated text, generated TTS duration/path, planned interval, available duration, stretch ratio, overflow state, rephrase need, and preserved silence before/after. Plans follow input order. The planner and final audio interval check reject ordinary dialogue overlap before mixing.

If the original media is supplied with `--source-media`, FFmpeg extracts mono 16 kHz PCM and a deterministic energy scan detects sustained quiet near transcript boundaries. Runs below RMS 0.005 that last at least 80 ms, within 600 ms after a segment start, are kept as leading silence for that segment. The final v4 output was checked sample by sample: all 10 measured pauses were silent.

Short TTS is padded within its planned window. Slight overrun uses the existing bounded time stretch. For remaining long clips, the worker marks that concise rephrasing is needed, applies only the maximum safe speed-up, then truncates the tail to the planned window. It records the pre-fit and post-fit overflow, shortening attempts, shortening result, truncated duration, and whether truncation was forced. The default overflow policy is now `trim`; `preserve` remains an explicit legacy option and schedules later dialogue after preserved overflow. The lower-level mixer keeps its sum and global peak protection for existing Phase 1E mixing use, while the ordinary dialogue pipeline verifies zero overlap before calling it.

The provider-neutral `shorten_for_duration(text, target_duration, language)` hook delegates to a provider's existing concise-rephrase method when available. Its safe default returns the original text. The current NLLB/Argos providers have no deterministic shortening strategy, so an attempt is recorded without changing text.

Example alignment command:

```powershell
worker/.venv-tts/Scripts/python.exe worker/scripts/benchmark_alignment.py `
  output/real-validation-v4/tts/translated.json `
  output/real-validation-v4/tts/audio `
  --output output/real-validation-v4 `
  --source-media input/real_test/sample.mp4
```

## Real-world v4 result

The full pipeline was rerun with `input/real_test/sample.mp4`. The report and required artifacts are under `output/real-validation-v4/`.

| Metric | Result |
| --- | ---: |
| Source duration / transcript segments / speakers | 60.075 s / 11 / 2 |
| Planned dialogue overlaps / actual overlaps | 0 / 0 |
| Source pauses measured / verified silent in output | 10 / 10 |
| Overflow before fitting / still over safe fit after stretch | 9 / 7 |
| Actual timeline overflows after truncation | 0 |
| Severe overflows (over 1.5× target) | 2 |
| TTS duration before fitting | 70.120 s |
| Shortened segments / attempts | 0 / 5 |
| Forcibly truncated segments | 7 |
| Final dubbed WAV duration / clipping samples | 59.860 s / 0 |
| ASR / diarization / translation | 4.645 s / 40.580 s (CPU) / 43.353 s total (CPU) |
| TTS / alignment / render | 4.956 s / 1.031 s / 3.251 s |
| Render / MP4 decode | Passed / passed |
| Output MP4 | 34,138,590 bytes; H.264 + AAC; 60.07 s |
| Summed pipeline runtime, including MP4 decode validation | 99.438 s |

Technical checks passed: all stages completed, the final MP4 decoded, no dialogue intervals overlap, each detected pause remains silent, and clipping count is zero. Overflow remains visible: seven segments were truncated after safe stretching and still need concise rephrasing. No segment was shortened, as the installed providers return original text when no shortening strategy exists. The validation report records all seven truncations explicitly; technical PASS does not establish that every spoken phrase is complete.

## Tests and limitations

Tests cover adjacent and paused segments, slight and severe overrun, multiple speakers and consecutive segments, source-waveform pause detection, zero final overlap, silent gaps, explicit legacy preserve behavior, and global clipping protection. Human listening quality, translation accuracy, pronunciation, and perceived naturalness were not evaluated. Truncated TTS tails may omit spoken content; each such segment is flagged for review/rephrasing. ASR still reports one global language for this mixed Japanese/English clip.
