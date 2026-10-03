# Phase 1E: Audio Alignment and Mixing

## Scope and architecture

Phase 1E turns the Phase 1D segment WAVs into one time-aligned dubbed WAV. It uses the translated transcript's original `start`, `end`, `speaker`, and segment ID as timeline metadata, while placing each generated Vietnamese clip at its original segment start. Phase 1H adds a deterministic planning pass that caps ordinary dialogue clips at their allocated source windows before mixing. It does not render video, mix source/background audio, or assign different voices.

Input is the Phase 1D benchmark JSON, whose `segments` entries include the original transcript data and an `audio.audio_path`. The pipeline reuses that path mapping; where it is absent, it resolves the conventional `segment-{id:04d}.wav` filename from the supplied TTS audio directory. Segment IDs must be unique. Missing or malformed audio fails only that segment, leaves its target time window silent, and processing continues without shifting later segments.

The core uses Python's `wave` and `array` modules. It reuses the worker's existing FFmpeg executable resolution (system FFmpeg or bundled `imageio-ffmpeg`) for format conversion and `atempo`. No new Python audio dependency or CUDA model was added. Output is mono 16 kHz signed 16-bit PCM WAV by default; output sample rate and one/two-channel layout are configurable.

## Alignment policy

For each segment, the target duration is `end - start`; audio metadata is read from the actual input and output WAVs. The reported stretch factor is `target_duration / original_tts_duration`.

- If TTS is shorter, the pipeline preserves the samples and appends silence to the target duration. Set `--no-pad-short` to leave the short clip as-is.
- If TTS is longer but the factor falls within `--min-stretch` and `--max-stretch` (defaults `0.90` and `1.10`), FFmpeg `atempo` speeds it up with pitch preservation. It adjusts the result by at most sample rounding to match the target frame count.
- If the required factor is outside those bounds, the current default `--overflow-policy trim` applies only the configured maximum safe speed-up, then explicitly truncates the tail to the allocated source window. The segment remains marked overflow, with `requires_concise_rephrasing`, `forcibly_truncated`, and truncated duration recorded. `preserve` remains an explicit legacy option; its excess duration shifts later planned dialogue while keeping it non-overlapping. `fail` excludes that clip from the timeline.
- Invalid/empty WAVs and missing files become `failed` segment records. The remaining segments are still aligned.

The default dialogue timeline covers silence from time zero through the latest source end. The planner preserves each source gap, schedules segments in their source order, and ensures that an overlong clip cannot run into the next ordinary dialogue segment. The final mix retains the existing global 0.99 peak protection; gaps remain zero-valued silence. The mixer can still sum intentionally overlapping audio supplied to its lower-level interface, while the ordinary transcript pipeline rejects any overlap introduced by its plan. See [Phase 1H](phase-1h.md) for the validation report and audio timing limitations.

## Run commands

Use the existing Phase 1D result:

```powershell
worker/.venv/Scripts/python.exe worker/scripts/align_audio.py `
  output/phase1d/benchmark/tts-benchmark.json `
  output/phase1d/benchmark/audio `
  --output output/phase1e `
  --min-stretch 0.90 --max-stretch 1.10 `
  --overflow-policy trim --sample-rate 16000 --channels 1
```

Run the timing and peak report command (it also writes `alignment.json` and WAV outputs):

```powershell
worker/.venv/Scripts/python.exe worker/scripts/benchmark_alignment.py `
  output/phase1d/benchmark/tts-benchmark.json `
  output/phase1d/benchmark/audio `
  --output output/phase1e
```

Generated files are `output/phase1e/dubbed_timeline.wav`, `output/phase1e/aligned_audio/segment-*.wav`, `output/phase1e/alignment.json`, and `output/phase1e/alignment-benchmark.json`.

## Real Phase 1D benchmark result

The benchmark used the four actual clips under `output/phase1d/benchmark/audio`. It completed without failed segments. Three clips fit their target windows after padding or bounded stretch; the fourth was retained and marked overflow.

| Metric | Result |
| --- | ---: |
| Segments / successful / failed / overflow | 4 / 3 / 0 / 1 |
| Source timeline duration | 33.180 s |
| Total source TTS duration | 24.70125 s |
| Alignment runtime | 0.465599 s |
| RTF (alignment runtime / source timeline duration) | 0.014033 |
| Maximum / average absolute duration error | 1.6845 s / 0.421125 s |
| Stretch factor | 0.977036 (one clip) |
| Final timeline duration | 34.8645 s |
| Output WAV | 16 kHz, mono, PCM s16le |
| Peak amplitude / clipped samples | 0.565674 / 0 |

This recorded Phase 1E benchmark explicitly used the legacy `preserve` policy. Its long clip began at 26.96 s with a target end of 33.18 s. Its original TTS duration was 7.9045 s, longer than the 6.22 s target window by 1.6845 s. The configured policy retained it, so the output timeline ended at 34.8645 s. Current default behavior is documented above and validated in Phase 1H.

Validation reopened the final timeline and every aligned segment WAV, confirmed nonzero audio metadata and the reported duration, checked that the gap from 7.72 s to 15.98 s remains silent, and confirmed segment IDs, speaker labels, and time windows are preserved. Clipping count is zero. This validates technical format and placement only; no listening evaluation was performed.

## Tests and regression

The focused alignment tests use short synthetic WAVs and cover exact match, padding, bounded time-stretch, overflow preserve/trim, missing/corrupt/zero-duration input, gaps, multiple clips, speaker and ID preservation, overlap mixing, peak limiting, empty and one-segment timelines, duration, and output format.

After Phase 1E changes, the complete worker suite passed in the ASR environment (**62 passed, 1 skipped**) and in the diarization/translation environment (**63 passed**). The existing Phase 1D Vietnamese smoke was rerun: it generated and reopened a 16 kHz WAV successfully. Earlier phase implementations and their environments were not modified. See Phase 1A–1D records for their model quality and GPU limitations.

## Limitations and next phase

Phase 1D's Piper VIVOS x-low voice logs skipped Vietnamese phonemes; the resulting TTS is not validated for pronunciation or naturalness. Phase 1E does not measure semantic alignment or speech quality, restore background music, separate source speech, or blend voice levels perceptually. The single global peak limit prevents digital clipping but is not loudness normalization.

Before proceeding to video rendering, review the preserved overflow and evaluate the Vietnamese voice with a listening test or a better licensed model. A later video phase can mux this WAV only after those audio choices are accepted.
