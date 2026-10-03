# Phase 1G: Dubbing Quality Safeguards

Phase 1G makes source-language routing explicit and adds bounded duration-fitting and translation-feedback controls to the existing worker pipeline. It does not replace the ASR, diarization, translation, TTS, alignment, or FFmpeg rendering architecture.

## Language routing

`translate_speaker_transcript` reads the ASR language from the transcript metadata. Without an explicit `--source-language`, that detected code is the translation source. A caller may provide an explicit override; the output records `detected_language`, `translation_source_language`, `translation_target_language`, and whether the source was overridden. The translation engine validates the requested pair before translating any segment. Unsupported pairs fail the translation stage with the provider's reason; no fallback to English occurs.

The translation interface also exposes provider hooks for spoken-dubbing phrasing and concise rephrasing. The default implementation delegates to ordinary translation. Argos has no contextual style controls or rephrasing implementation, so the interface does not imply that Argos output has been made more natural or concise.

## Bounded TTS feedback and fitting

`DubbingQualitySettings` defaults to a target/source duration factor of 0.85–1.10 (at most about 1.18x speed-up), a maximum retry count of one (configurable up to two), a translation expansion guard of 1.25, and severe overflow above 1.50 times the target duration. The TTS feedback helper synthesizes once, requests a shorter rephrase only when the clip exceeds the configured stretch capacity, and retries only when a provider implements the rephrase hook and returns a shorter candidate. It retains speaker IDs and records the original/final text, pre-fit/post-retry duration, retries, overflow duration, and severity. If no rephraser exists or the segment remains long, the audio is preserved and reported as overflow.

Alignment defaults to the corresponding 0.85 minimum fit factor. Its benchmark records severe overflow count, percentage of segments that fit without overflow, pre-fit/post-fit total TTS durations, clipping, and per-segment text/timing/retry fields. Longer speech is not silently discarded.

`benchmark_quality_tts.py` runs the bounded TTS feedback step when its input transcript has a supported translation route. For example:

```powershell
worker/.venv-tts/Scripts/python.exe worker/scripts/benchmark_quality_tts.py `
  output/translated.json `
  --output output/quality-tts `
  --min-time-stretch-ratio 0.85 `
  --max-time-stretch-ratio 1.10 `
  --max-translation-expansion-ratio 1.25 `
  --max-overflow-ratio 1.50 `
  --max-retries 1
```

## Real source validation

The real input `input/real_test/sample.mp4` was rerun after these changes. ASR detected `ja`; Community-1 diarization succeeded with the CPU fallback and found two speakers. The installed Argos environment contains only `en` and `vi`, with no `ja → vi` package or route. Automatic routing therefore correctly stopped at translation with an explicit unsupported-pair error. No v2 TTS, alignment, dubbed WAV, or final video was created from the wrong language pair. `output/real-validation-v2/validation-report.json` records the stop point and per-segment blocked fields.

The previous forced-English run had 7 overflow segments out of 11, 1 severe overflow using the 1.50 threshold, and 4/11 (36.36%) fitting segments. Its pre-fit summed TTS duration was 59.177938 s and post-fit segment duration sum was 71.389938 s. These are a baseline from the incorrectly forced `en → vi` run, not a Phase 1G after measurement. The new Japanese route could not reach TTS because no compatible translation model is installed.

## Tests and limitations

The ASR worker environment passes **80 tests with 1 skipped**; the diarization/translation environment passes **81 tests**. Added tests cover automatic `ja → vi` and `en → vi` routing, explicit override and metadata, unsupported-pair rejection, one/two bounded rephrase retries, preservation of severe overflows, and conservative settings.

The code does not provide a Japanese→Vietnamese model, a contextual spoken-language translator, or an Argos rephrasing hook. Translation correctness, pronunciation, naturalness, and speaker-specific voice quality were not evaluated. GPU diarization can still fail on this machine; the real validation used CPU fallback. Since Phase 1G stopped before TTS, there is no real-v2 overflow or speech-quality result.
