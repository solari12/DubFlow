# Phase 1K.4 — Offline TTS benchmark

## Objective

Compare the current Vietnamese Piper voice with one local Vietnamese candidate using identical approved translation text and a TTS-only pronunciation variant. This is a listening benchmark, not a provider-selection step. All additions are benchmark-only; no production defaults or ASR, diarization, translation, alignment, rendering, or validation behavior is changed.

## Hardware

Target: Windows 10/11, Python 3.11, Intel i5-12500H, NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM. The benchmark adapters use CPU inference so they do not contend with the existing diarization model's VRAM use. A model's observed CPU performance is not a claim of CUDA acceleration.

## Candidate models and pre-download assessment

| Candidate | Source / license | Approximate size | Expected device requirements | Hardware rationale |
|---|---|---:|---|---|
| Piper baseline `vi_VN-vivos-x_low` via Sherpa-ONNX | Existing local model; voice card reports VIVOS training data under CC BY-NC-SA 4.0. | 27.7 MB ONNX file (plus phonemizer data) | CPU; 16 kHz output | Already installed, small, fast baseline. |
| KorvaTTS `dogenthq/KorvaTTS` | [Model repository](https://huggingface.co/dogenthq/KorvaTTS); model card states Apache-2.0 for weights, voice styles, and audio. | 419 MB repository; model card lists a 256 MB fp32 vector estimator. | ONNX Runtime, CPU supported; no CUDA or API required. Package/model download needed in an isolated TTS benchmark environment. | 99M-parameter Vietnamese-first, code-switch-capable model with a sub-GB stated artifact. CPU inference avoids the 4 GB GPU constraint. Benchmark-only pending actual loading, speed, and listening review. |

Repository inspection found only the existing Sherpa-ONNX/Piper TTS runtime and its local Piper model. No other TTS library/model or VoiceStudio-derived component was installed. At most one new candidate was selected. No candidate weights are considered production-ready from this benchmark alone.

## Installation requirements

Piper uses the existing `worker/.venv-tts` environment. KorvaTTS must use a separate Python 3.11 environment (for example `worker/.venv-tts-korva`) to avoid changing the ASR/diarization environment. After reviewing the model size/license above, create and install the isolated environment with:

```powershell
cd D:\Project\DubFlow\worker
& '.venv-tts/Scripts/python.exe' -m venv .venv-tts-korva
& '.venv-tts-korva/Scripts/python.exe' -m pip install korvatts
```

The benchmark itself makes no cloud/API calls; KorvaTTS's first-run Hugging Face download is a one-time model asset download, after which synthesis is local.

## Benchmark methodology

- Same original and normalized Vietnamese full-sentence inputs for each provider.
- Separate original, normalized, isolated Hiragana, isolated Katakana, and isolated Kanji WAVs.
- No alignment, time stretching, mixing, or video rendering.
- Capture provider initialization time, synthesis time per WAV, measured WAV duration/sample rate/channels/PCM status, and RTF (synthesis seconds divided by audio duration).
- Capture process peak working set where measured by Windows; do not confuse system RAM with VRAM.
- Current harness labels VRAM as unmeasured unless process-specific sampling is available. It never reports a sampled GPU number as exact process peak. The selected Korva and Piper adapters are CPU-only and report that explicitly.
- Any failed initialization/generation is written into the provider report rather than silently omitted.
- Human listening is required for naturalness, robotic quality, foreign-word pronunciation, and pauses; there is no automatic score or winner.

## Test text and pronunciation variants

Original approved text (immutable):

> Trước tiên, bạn nên học Hiragana, Katakana và một chút Kanji. Đây là ba hệ chữ viết chính, nên hãy học chúng trong quá trình sử dụng ứng dụng. Khi tiếp tục học, hãy đặt câu ví dụ và áp dụng những gì mình đang học.

TTS-only normalized experiment: `Hiragana` → `hi-ra-ga-na`, `Katakana` → `ka-ta-ka-na`, and `Kanji` → `kan-ji`. Hyphens are used as hypothesized syllable separators based on the example supplied for this phase. These are not asserted Japanese pronunciations, and they do not replace subtitle/translation text. The benchmark also tests the untouched romanized terms in isolation.

## Benchmark results

Results from the controlled CPU run (the report JSON is authoritative for full precision):

| Model | Loaded | Total local assets | Init | Original duration / RTF | Normalized duration / RTF | Peak process working set | Output | Compatibility |
|---|---|---:|---:|---:|---:|---:|---|---|
| Piper VIVOS x_low | Yes | 45.7 MB local model directory (27.7 MB ONNX) | 1.03 s | 14.05 s / 0.034 | 14.78 s / 0.110 | 205.7 MiB | PCM mono, 16 kHz | TECHNICALLY_COMPATIBLE on CPU |
| KorvaTTS | Yes | 400.5 MB cached package + model assets | 2.91 s (cached assets) | 13.56 s / 2.681 | 14.05 s / 2.708 | 685.1 MiB | PCM mono, 44.1 kHz | TECHNICALLY_COMPATIBLE on CPU; no CUDA claim |

All five WAVs per model (original, normalized, and three isolated terms) were generated and reopened successfully. In the final 32-step KorvaTTS CPU pass, the two full sentences had RTF above 1, so synthesis was slower than audio duration on this run; the isolated-word runs were also slower than real time. Earlier warm runs differed, so treat the report's single final pass as machine-condition-specific. Pronunciation and prosody remain `NOT_REVIEWED` pending human listening. The `nvidia-smi` samples in the report were 0 MiB before/peak/after; they are system-wide, coarse observations and not per-process measurements. Both adapters use CPU, so GPU compatibility was not tested. Quality stays `NOT_REVIEWED` until a person listens and fills `quality-review.md`. Compatibility is technical only and does not select a winner.

## Limitations

- One short spoken passage is not a broad Vietnamese voice-quality evaluation.
- English/romanized Japanese terms are not evidence that a model can pronounce Japanese correctly.
- Sampled/system VRAM is not an exact per-process peak. The v1 adapters use CPU; CUDA fit is not tested.
- Human ratings are required. No numeric MOS or automatic winner is produced.
- The Piper voice card's CC BY-NC-SA 4.0 training-data license and the candidate's model license must be reviewed for the intended use; source-code licensing alone is insufficient.

## Reproduce

From PowerShell, run Piper in the existing isolated TTS environment:

```powershell
cd D:\Project\DubFlow\worker
& '.venv-tts/Scripts/python.exe' 'scripts/benchmark_tts_v1.py' --provider piper
```

After creating the separate candidate environment and installing KorvaTTS as documented above, run:

```powershell
cd D:\Project\DubFlow\worker
& '.venv-tts-korva/Scripts/python.exe' 'scripts/benchmark_tts_v1.py' --provider korva
```

The two commands write into the same benchmark directory. Then listen to the WAVs under `output/tts-benchmark-v1/piper/` and `output/tts-benchmark-v1/korvatts/`, and fill `output/tts-benchmark-v1/quality-review.md`. `benchmark-report.md` summarizes technical reports without ranking models.
