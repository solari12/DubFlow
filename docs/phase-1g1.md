# Phase 1G.1: Japanese-to-Vietnamese NLLB Provider

Phase 1G.1 adds an explicit, lazy Hugging Face NLLB provider for the Phase 1G translation interface. It leaves the ASR, diarization, TTS, alignment, and rendering architecture in place. Provider selection is explicit; an NLLB failure does not fall back to Argos.

## Provider and language routing

The default model is `facebook/nllb-200-distilled-600M`, and the default device is CPU. `DUBFLOW_NLLB_MODEL` and `DUBFLOW_NLLB_DEVICE` configure defaults; explicit `--model` and `--device` arguments take precedence. Transformers and the model are loaded only when the NLLB engine is explicitly selected. The shared language map currently covers `ja -> jpn_Jpan`, `vi -> vie_Latn`, and `en -> eng_Latn`. Unsupported codes fail during language-pair validation before any segments are translated.

Install the `nllb` optional dependency in an environment with the desired PyTorch build. The real run used the existing Python 3.11 diarization environment and its existing PyTorch 2.11.0+cu130; Transformers 4.57.6 and NLLB weights were added there without changing PyTorch. The downloaded checkpoint occupied about 2.48 GB in the user Hugging Face cache. Model weights are not in the repository, and the first run requires network access unless the cache is already populated.

Example:

```powershell
worker/.venv-diarization/Scripts/python.exe worker/scripts/benchmark_translation.py `
  output/real-validation-v3/transcript.json `
  --provider nllb --device cpu --target-language vi `
  --output output/real-validation-v3/translation-benchmark.json
```

The benchmark records provider/model, device, detected and resolved languages, model-load time, segment translation time, total time, and per-segment output/errors.

## Translation benchmark

The real source was `input/real_test/sample.mp4`. ASR detected Japanese and produced 11 transcript segments. The speaker-aware transcript was translated as one global `ja -> vi` route, as Phase 1G currently carries one detected source language for the clip.

| Measurement | Result |
| --- | ---: |
| Model | `facebook/nllb-200-distilled-600M` |
| Device | CPU |
| Model load (warm local cache) | 5.661 s |
| Translation of 11 segments | 18.998 s |
| Translation benchmark total | 24.660 s |
| Successful / failed segments | 11 / 0 |

These timings exclude downloading weights. A prior cold-cache run measured 62.142 s for model loading and 86.878 s total for 11 segments. Both runs used CPU. No human translation-quality review was performed.

## Real-world pipeline validation

The post-implementation run wrote artifacts under `output/real-validation-v3/`: `transcript.json`, `translated.json`, `alignment.json`, `dubbed_timeline.wav`, `dubbed_video.mp4`, and `validation-report.json`, plus per-stage benchmark reports.

| Stage | Result |
| --- | ---: |
| Source duration | 60.075 s |
| ASR, base / CUDA | 4.454 s processing; 11 segments; detected `ja` |
| Community-1 diarization / CPU | 28.592 s inference; 2 speakers; 11 turns |
| NLLB translation / CPU | 24.660 s total; 11 translated; 0 failed |
| Vietnamese TTS | 3.671 s; 11 successful; 0 failed |
| Alignment | 0.805 s; 5 overflow segments; 1 severe overflow |
| Segments fitting without severe stretch | 54.55% (6 of 11) |
| Measured clipping | 0 samples |
| FFmpeg rendering | 2.159 s; output duration 60.07 s |
| Final MP4 decode check | Passed; 1.338 s |
| Final file | 34,164,139 bytes; H.264 video + AAC audio |
| Summed pipeline runtime | 72.069 s |

The summed runtime uses each stage's measured processing time, includes final decode validation, and excludes model downloads. CPU diarization was used for this run because the RTX 3050 GPU path had failed with the existing Community-1 setup in the earlier validation. The NLLB model was run on CPU to keep it separate from the 4 GB GPU workload.

## Limits

ASR exposes a single clip-level language (`ja`) even though this video contains English as well as Japanese speech. Consequently, English segments were also passed to NLLB as Japanese; per-segment language detection or mixed-language routing is not part of this phase. Technical translation success means NLLB returned non-empty text for all segments, not that a human verified translation accuracy.

The current Vietnamese TTS adapter does not implement the optional concise-rephrasing hook. The bounded duration feedback stage therefore made zero translation retries and preserved overlong audio. Five segments exceeded their target duration, one by more than the configured severe-overflow threshold. Speech quality and intelligibility were not evaluated.

The full worker suites passed after the provider was added: **86 passed, 1 skipped** in `worker/.venv`, and **87 passed** in `worker/.venv-diarization`. The skip is an existing environment-dependent test.
