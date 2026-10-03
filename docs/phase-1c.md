# Phase 1C: Translation Layer

## Scope and design

Phase 1C consumes a Phase 1B.2 speaker-aware transcript and adds `source_text`, `target_text`, and `error` to each segment. Segment IDs, start/end times, and speaker labels are preserved. The provider-neutral `TranslationEngine` interface accepts text and source/target language codes; the current adapter is Argos Translate. A later engine can implement the same interface without changing the transcript merge pipeline.

The pipeline currently translates each ASR segment independently. It does not pass neighboring segments as context, so pronouns, names, and phrases split across turns may be translated less naturally. Empty source text becomes an empty target without invoking the engine. If a segment fails, its target is `null`, its error is recorded, and subsequent segments continue. Missing language routes produce a per-segment unsupported-language error.

## Initial engine and model

Argos Translate is a local, offline-capable translation library whose installed language packages provide translation routes. This implementation uses the English-to-Vietnamese `translate-en_vi` package and runs it on CPU. This keeps translation from competing with ASR or diarization for the RTX 3050's 4 GiB of VRAM. The downloaded model archive was 67,770,159 bytes; its unpacked `model.bin` was 76,828,998 bytes. The bundled English MiniSBD sentence splitter was 188,043 bytes. The sentence splitter is cached under `worker/.model-cache` on first use.

The optional `translation` extra pins `argostranslate==1.11.0`. Argos Translate is MIT licensed; the English-to-Vietnamese model package identifies its underlying OPUS model as CC BY 4.0. Review the model and library terms before redistributing either. Argos can use installed packages to form routes between languages, but only installed routes are supported. The adapter forces CPU execution and uses MiniSBD to avoid an optional Stanza resource download.

The translation extra has heavyweight transitive dependencies, including Stanza and PyTorch. For this Windows setup it was installed into the existing `.venv-diarization` environment; the ASR `.venv` was left unchanged. Cache paths default to the ignored `worker/.model-cache` directory. Install the package and run the benchmark as described in [worker/README.md](../worker/README.md#translation-benchmark).

References: [Argos Translate project](https://github.com/argosopentech/argos-translate), [official package index](https://github.com/argosopentech/argospm-index/blob/main/index.json), [Argos Translate on PyPI](https://pypi.org/project/argostranslate/).

## Benchmark

Command:

```powershell
worker/.venv-diarization/Scripts/python.exe worker/scripts/benchmark_translation.py `
  output/phase1b2/speaker-transcription-benchmark.json `
  --source-language en --target-language vi `
  --output output/phase1c/translation-benchmark.json
```

Recorded report: `output/phase1c/translation-benchmark.json` (generated output is ignored by Git).

| Metric | Result |
| --- | ---: |
| Engine / device | Argos Translate / CPU |
| Source → target | `en` → `vi` |
| Input segments | 4 |
| Translated segments | 4 |
| Failed segments | 0 |
| Runtime | 8.902616 s |
| Benchmark status | Success |

This is a functional smoke benchmark, not a translation-quality evaluation. The fixture contains Vietnamese speech in one segment despite the benchmark's declared English source language. The engine translated that text as English and produced poor output for it. Language codes are applied to every segment as provided; language detection and mixed-language handling are not implemented. The short fixture does not support broader quality or performance conclusions.

## Tests and regression check

Translation unit tests cover ordinary translation, language forwarding, metadata preservation, blank segments, per-segment failures, continuing after a failure, multiple segments, and unsupported routes. The combined test run passed all 39 tests; the ASR-only environment passed 38 tests with one expected Torch-dependent skip. Existing ASR and diarization tests remain in place. The Phase 1B.2 end-to-end benchmark was rerun after adding Phase 1C: ASR took 7.393 s, diarization 48.667 s, merge 0.000258 s, and total runtime was 80.444 s. It completed with 4 transcript segments, 3 assigned speakers, 1 unassigned segment, sampled whole-GPU memory of 3925 MiB, and no OOM. The ASR/diarization stages remain sequential; translation runs only after the input transcript has been produced and uses CPU.

No TTS, word alignment, translation API server, or production service is included in this phase.
