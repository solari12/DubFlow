# DubFlow Python Worker

Phase 1A is a local prototype for media-to-transcript processing. It extracts a mono 16 kHz WAV with FFmpeg, transcribes it through a replaceable ASR interface, and writes stable JSON. It does not include an API server, queue, database, or TTS.

## Phase 2A local development API

The optional FastAPI server sequences the existing ASR, diarization, translation, TTS, alignment, and rendering modules in a local in-process background task. It is a development prototype: jobs are held in memory, only one API process is expected, and uploaded media/output files are stored under `worker/output/dev-api/`.

Install the worker and API/model extras from the repository root:

```powershell
uv sync --project worker --extra dev --extra cuda --extra diarization --extra nllb --extra tts --extra dev-api
```

Set a Hugging Face token with access to `pyannote/speaker-diarization-community-1` if required by your account. The Dev API uses KorvaTTS for speech synthesis in the separate `.venv-tts-korva` environment; CUDA sessions are required when `DUBFLOW_TTS_DEVICE=gpu`. Set `DUBFLOW_TTS_VOICE` (default `gia_bao`), `DUBFLOW_TTS_DEVICE` (default `gpu`), and `DUBFLOW_TTS_STEPS` (default `32`) to configure synthesis. `DUBFLOW_TTS_PYTHON` can point to the Korva environment's Python executable. Piper benchmark scripts and history remain available but are not called by the Dev API. NLLB runs on CPU by default; set `DUBFLOW_TRANSLATION_PROVIDER=argos` to explicitly use Argos instead. ASR defaults to CUDA per worker settings; for CPU-only operation set `DUBFLOW_ASR_DEVICE=cpu` and `DUBFLOW_ASR_COMPUTE_TYPE=int8`.

For this checkout, the existing `.venv-diarization` environment contains ASR, diarization, translation, and the installed FastAPI server dependencies. Install cuBLAS into the same Python environment used by the API (the standard `uv sync` target is `worker/.venv`):

```powershell
python -m uv pip install --python worker\.venv-diarization\Scripts\python.exe "nvidia-cublas-cu12==12.6.4.1"
```

DubFlow discovers CUDA DLL folders from the active environment's NVIDIA wheels, CTranslate2 package, and PyTorch package at Python startup. The ASR engine repeats this setup before importing CTranslate2, including when launched through the Dev API's `--app-dir` path. No machine-wide CUDA path is required. The API calls `scripts/synthesize_korva.py` with `.venv-tts-korva` so Korva's ONNX Runtime and CUDA packages stay isolated. Start the API from the repository root with CUDA ASR, CUDA diarization, and GPU KorvaTTS:

```powershell
$env:DUBFLOW_ASR_DEVICE = "cuda"
$env:DUBFLOW_ASR_COMPUTE_TYPE = "float16"
$env:DUBFLOW_DIARIZATION_DEVICE = "cuda"
$env:DUBFLOW_TTS_DEVICE = "gpu"
worker\.venv-diarization\Scripts\python.exe -m uvicorn dubflow_worker.dev_api.main:app --app-dir worker/src --reload --port 8000
```

Set `HF_HUB_OFFLINE=1` only when the ASR, diarization, and NLLB models are already cached. The API logs the detected CTranslate2 CUDA device count and the loaded Whisper model's device and compute type.

In another terminal, start the frontend:

```powershell
cd D:\Project\DubFlow\frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The API allows CORS only from `http://localhost:5173`.

## Requirements

- Python 3.11 (the project intentionally rejects Python 3.12+ for this phase)
- FFmpeg on `PATH`, or the bundled `imageio-ffmpeg` binary installed with the worker
- `uv` is recommended for Python and dependency management
- The optional `cuda` extra for project-local NVIDIA cuBLAS runtime support; CPU mode is supported as a fallback

Install `uv` using its official Windows installation instructions if it is not available. Then, from the DubFlow repository root:

```powershell
uv python install 3.11
uv venv --python 3.11 worker/.venv
uv sync --project worker --extra dev --extra cuda
```

The `cuda` extra installs `nvidia-cublas-cu12==12.6.4.1` into the worker environment. The first install also downloads the Python environment and faster-whisper/CTranslate2 dependencies. ASR model weights are downloaded from Hugging Face on first use and are not bundled in this repository. For a CPU-only setup, omit `--extra cuda`.

## Run transcription

From the repository root:

```powershell
uv run --project worker python -m dubflow_worker.cli transcribe input/sample.mp4 `
  --language en --model base --device cuda --compute-type float16 `
  --output output/transcript.json
```

For CPU mode, use `--device cpu --compute-type int8`. The default model is `base`, selected as a conservative starting point rather than a guarantee of fit or performance on a 4 GB GPU. CTranslate2 4.8.2 requires CUDA 12.x and cuDNN 9 for GPU speech recognition. On Windows, the worker adds the optional NVIDIA package's `nvidia/cublas/bin` directory to this process's DLL search path before loading CTranslate2; it does not change the machine-wide PATH. The worker's CUDA extra supplies `cublas64_12.dll` and its cuBLAS companion DLLs. The installed CTranslate2 package supplies `cudnn64_9.dll`. If CUDA is requested but unavailable or incomplete, the command fails explicitly; it does not silently switch to CPU.

The output parent directory must already exist. For a different media input, use an absolute path or a path relative to the current directory. Supported extensions are `.mp4`, `.mkv`, `.mov`, `.webm`, `.mp3`, `.wav`, `.m4a`, and `.flac`.

## Configuration

CLI options override these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DUBFLOW_ASR_MODEL` | `base` | faster-whisper model name or local model path |
| `DUBFLOW_ASR_DEVICE` | `cuda` | `cuda` or `cpu` |
| `DUBFLOW_ASR_COMPUTE_TYPE` | device-dependent | `float16` for CUDA; `int8` for CPU |
| `DUBFLOW_FFMPEG` | `ffmpeg` | FFmpeg executable or path; the packaged binary is used if PATH lookup fails |
| `DUBFLOW_TTS_VOICE` | `gia_bao` | KorvaTTS voice style |
| `DUBFLOW_TTS_DEVICE` | `gpu` | KorvaTTS device: `gpu`, `cpu`, or `auto` |
| `DUBFLOW_TTS_STEPS` | `32` | KorvaTTS denoising steps, from 1 to 32 |
| `DUBFLOW_TTS_PYTHON` | `worker/.venv-tts-korva/Scripts/python.exe` | Python executable for the isolated KorvaTTS adapter |

The model cache follows Hugging Face's cache configuration, including `HF_HOME`. Model download size is not a measurement of peak VRAM use.

## Tests

Unit tests do not load a model, use a GPU, or require FFmpeg to be installed:

```powershell
uv run --project worker --extra dev pytest worker/tests
```

## Benchmark

Run named samples, with an explicit language per file, and save both a report and transcripts:

```powershell
uv run --project worker python worker/scripts/benchmark_asr.py `
  --case vietnamese=vi=input/phase1a2/vietnamese.wav `
  --case english-short=en=input/phase1a2/english-short.wav `
  --case english-long=en=input/phase1a2/english-long.wav `
  --models tiny base small --device cuda --compute-type float16 `
  --references-dir input/phase1a2/references `
  --transcripts-dir output/phase1a2/transcripts `
  --output output/asr-real-world-benchmark.json
```

The legacy single-input form is still supported. Repeating `--case NAME=LANGUAGE=PATH` permits multiple files and explicit language settings in one report. The report records model-load-plus-transcription time, duration, RTF, segment and transcript character counts, timestamp checks, optional WER/CER against `NAME.txt`, and sampled `nvidia-smi` memory. Transcripts are saved separately as JSON. CUDA models run sequentially in one process per case, so the first model pays context initialization. The memory figure is sampled **whole-GPU** usage, not process-attributed memory or an exact peak. Do not treat this short benchmark as proof that every model fits every input or machine state. See [Phase 1A.2](../docs/phase-1a2.md) for the current evaluation and local fixture sources.

## Diarization benchmark

Diarization is optional and isolated from the ASR environment in `worker/.venv-diarization/`. Its extra pins `pyannote.audio==4.0.7`, `torch==2.11.0`, `torchaudio==2.11.0`, and `torchcodec==0.15.0`; uv installs the matched CUDA 13.0 wheels from the official PyTorch index. Set the separate environment before syncing or running commands:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) 'worker/.venv-diarization'
uv sync --project worker --extra diarization --no-install-project
```

The model `pyannote/speaker-diarization-community-1` is gated. Sign in to Hugging Face, accept the [model conditions](https://huggingface.co/pyannote/speaker-diarization-community-1), then use `worker/.venv-diarization/Scripts/hf.exe auth login` to save auth in Hugging Face's local user cache. Never commit tokens. On Windows, TorchCodec's DLL loader cannot load its core DLL or a dependency in this environment. DubFlow's diarization loader uses the existing FFmpeg extractor to create mono 16 kHz audio and passes an in-memory tensor to pyannote, avoiding TorchCodec file decoding. CUDA model load and real fixture inference have been validated. The ASR environment remains separate and unchanged.

After model access is configured, run the standalone Phase 1B.1 benchmark from the repository root:

```powershell
uv run --project worker --extra diarization python worker/scripts/benchmark_diarization.py `
  input/phase1b1/two-voice-alternating.wav `
  --model pyannote/speaker-diarization-community-1 `
  --device cuda --min-speakers 2 --max-speakers 2 `
  --output output/phase1b1/diarization-benchmark.json
```

The runner records model loading separately from inference, computes inference RTF, and distinguishes sampled whole-device VRAM from PyTorch process peak counters. See [Phase 1B.1](../docs/phase-1b1.md) for sample provenance, current access/runtime status, and limitations.

## Speaker-aware transcription benchmark

To run faster-whisper `base` followed by Community-1 diarization and temporal-overlap speaker assignment:

```powershell
worker/.venv/Scripts/python.exe worker/scripts/benchmark_speaker_transcription.py `
  input/phase1b1/two-voice-alternating.wav `
  --output output/phase1b2/speaker-transcription-benchmark.json
```

The ASR child process exits before the diarization child starts, releasing the ASR model and CUDA context before pyannote loads. Speaker labels use the greatest single-turn overlap, with a default 20% overlap ratio threshold. The report includes actual ASR text, speaker labels, overlap measurements, stage timings, RTF, and sampled whole-GPU memory. The Phase 1B.2 result and test status are documented in [Phase 1B.2](../docs/phase-1b2.md).

## Translation benchmark

Phase 1C adds a provider-neutral `TranslationEngine` interface and an offline Argos Translate adapter. Argos is an optional dependency and should be installed into the diarization environment on this Windows GPU setup so its transitive dependencies do not alter the ASR environment. From the repository root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) 'worker/.venv-diarization'
uv sync --project worker --extra diarization --extra translation --no-install-project

$env:ARGOS_PACKAGES_DIR = Join-Path (Get-Location) 'worker/.model-cache/argos-packages'
$env:XDG_CONFIG_HOME = Join-Path (Get-Location) 'worker/.model-cache/argos-config'
$env:XDG_DATA_HOME = Join-Path (Get-Location) 'worker/.model-cache/argos-data'
$env:XDG_CACHE_HOME = Join-Path (Get-Location) 'worker/.model-cache/argos-cache'
worker/.venv-diarization/Scripts/argospm.exe install translate-en_vi
```

Argos downloads model weights on first setup; its bundled English sentence splitter may also be fetched on first use. Translation runs on CPU. The benchmark accepts a Phase 1B.2 speaker-aware transcript and preserves its IDs, times, and speakers:

```powershell
worker/.venv-diarization/Scripts/python.exe worker/scripts/benchmark_translation.py `
  output/phase1b2/speaker-transcription-benchmark.json `
  --source-language en --target-language vi `
  --output output/phase1c/translation-benchmark.json
```

See [Phase 1C](../docs/phase-1c.md) for engine details, the recorded run, and its language and quality limitations.

## Context-aware translation quality experiment

Phase 1I reruns ASR and diarization for a real source, merges speaker labels, groups only clear adjacent continuations, and translates each unit once. Language-script routing handles obvious English spans in a globally Japanese-detected transcript. Original segment IDs, timestamps, and speaker labels remain mapped into `translated.json`. The configured term list is in `worker/config/translation-glossary.json`; protected terms are masked across providers. NLLB output then passes through a separate deterministic Vietnamese naturalizer. Incomplete units without safe same-speaker, same-language context keep their literal output for review and have no final rewrite.

Run from the repository root; ASR and diarization run in their existing environments, followed by CPU NLLB in the diarization environment:

```powershell
worker/.venv-diarization/Scripts/python.exe worker/scripts/validate_translation_quality.py `
  input/real_test/sample.mp4 `
  --output-dir output/real-validation-v5 `
  --diarization-device cpu --provider nllb `
  --model facebook/nllb-200-distilled-600M --device cpu --offline-models
```

The command writes `transcript.json`, `translated.json`, `translation-comparison.json`, and `validation-report.json`. It does not run TTS or render video. `--reuse-upstream --offline-models` can reuse saved ASR/diarization outputs and force cached model use after those upstream stages have already passed. The report requires subjective human review and does not turn successful inference into a translation-quality claim. Each segment row repeats its complete context-unit translation for review convenience; `translation_units` is canonical, and segment-row `target_text` must not be concatenated or sent directly to TTS. See [Phase 1I](../docs/phase-1i.md) for the real benchmark and unresolved limitations.

## TTS benchmark

Phase 1D uses a provider-neutral `TTSEngine` interface with a CPU-only Sherpa-ONNX adapter for Piper's Vietnamese VIVOS voice. Keep TTS in its own environment so its ONNX Runtime and Sherpa DLLs cannot affect the ASR or diarization runtimes. From the repository root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) 'worker/.venv-tts'
uv sync --project worker --extra tts --no-install-project

$ttsCache = Join-Path (Get-Location) 'worker/.model-cache/tts'
$ttsArchive = Join-Path $ttsCache 'vits-piper-vi_VN-vivos-x_low.tar.bz2'
New-Item -ItemType Directory -Force -Path $ttsCache | Out-Null
Invoke-WebRequest `
  'https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-vi_VN-vivos-x_low.tar.bz2' `
  -OutFile $ttsArchive
tar -xjf $ttsArchive -C $ttsCache
```

After downloading the model once, synthesis runs offline and always uses CPU. A translated transcript segment becomes a separate WAV under the output directory; its ID, timestamps, and speaker label are copied into the report. The current voice has one fixed speaker ID, so labels are preserved as metadata and do not alter the generated voice:

```powershell
worker/.venv-tts/Scripts/python.exe worker/scripts/benchmark_tts.py `
  output/phase1c/translation-benchmark.json `
  --language vi --output output/phase1d/benchmark
```

The report is `output/phase1d/benchmark/tts-benchmark.json`. See [Phase 1D](../docs/phase-1d.md) for the selected voice's limitations, license notes, smoke check, and measured results.

## Audio alignment benchmark

Phase 1E takes the Phase 1D benchmark JSON and its audio directory. It reuses each segment's `audio.audio_path` mapping, aligns the clip to its transcript window, and writes one PCM WAV timeline plus segment-level aligned WAVs and metadata:

```powershell
worker/.venv/Scripts/python.exe worker/scripts/align_audio.py `
  output/phase1d/benchmark/tts-benchmark.json `
  output/phase1d/benchmark/audio `
  --output output/phase1e `
  --min-stretch 0.90 --max-stretch 1.10 `
  --overflow-policy trim --sample-rate 16000 --channels 1
```

Benchmark the same run with:

```powershell
worker/.venv/Scripts/python.exe worker/scripts/benchmark_alignment.py `
  output/phase1d/benchmark/tts-benchmark.json `
  output/phase1d/benchmark/audio `
  --output output/phase1e
```

The default `trim` behavior plans clips in source order, pads short clips, applies bounded FFmpeg time-stretch for slight overruns, and explicitly truncates remaining overflows at each segment's source window. The opt-in `preserve` policy remains available for legacy use and shifts later dialogue after preserved overflow. The report records shortening attempts, remaining rephrase needs, truncation, preserved pauses, and actual overlap counts. Ordinary dialogue clips cannot overlap in the final timeline. The existing global peak protection remains in place. See [Phase 1H](../docs/phase-1h.md) for the implementation and full real-source validation.

Pass `--source-media` to `benchmark_alignment.py` or `align_audio.py` to decode the original media and retain measurable silence around transcript boundaries. The deterministic `shorten_for_duration` translation hook returns unchanged text when the provider has no shortening method. Remaining long clips are explicitly truncated at their planned ends and marked for concise rephrasing; they are never mixed over later dialogue.

## Video rendering

Phase 1F replaces the original video's audio with the Phase 1E WAV, copies the source video stream by default, and encodes the dubbed audio as AAC. Longer dubbed audio is trimmed to the video duration; shorter audio is padded with silence to the video end. The full Phase 1E WAV is preserved.

```powershell
worker/.venv/Scripts/python.exe worker/scripts/render_video.py `
  input/sample.mp4 `
  output/phase1e/dubbed_timeline.wav `
  --output output/phase1f/dubbed_video.mp4 `
  --audio-codec aac --audio-bitrate 192k --video-mode copy

worker/.venv/Scripts/python.exe worker/scripts/benchmark_render.py `
  input/sample.mp4 `
  output/phase1e/dubbed_timeline.wav `
  --output-dir output/phase1f
```

Use `--video-mode h264` to explicitly re-encode when the source cannot be copied into MP4. See [Phase 1F](../docs/phase-1f.md) for result fields, duration policies, validation, and benchmark details.

## Dubbing quality safeguards

Translation now uses the ASR-detected language by default. Pass `--source-language` only as an explicit override. Unsupported provider pairs stop with an error instead of silently using English. The quality TTS benchmark provides a bounded rephrase hook and conservative duration settings:

```powershell
worker/.venv-tts/Scripts/python.exe worker/scripts/benchmark_quality_tts.py `
  output/translated.json `
  --output output/quality-tts `
  --min-time-stretch-ratio 0.85 --max-time-stretch-ratio 1.10 `
  --max-translation-expansion-ratio 1.25 --max-overflow-ratio 1.50 `
  --max-retries 1
```

Argos currently does not implement concise rephrasing, and installed translation routes are environment-specific. NLLB-200 is available as a separate lazy provider for `ja -> vi`; install the `nllb` extra in an environment that already has its desired PyTorch build. The default device is CPU. Weights are fetched by Hugging Face on first load and are not stored in the repository. Select it explicitly with `benchmark_translation.py ... --provider nllb --device cpu`; provider failures do not fall back to Argos. See [Phase 1G.1](../docs/phase-1g1.md) for the benchmark and full pipeline run.

## CUDA verification record

Verified on 2026-10-02 with Python 3.11.17, faster-whisper 1.2.1, CTranslate2 4.8.2, NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB), and driver 610.62. `nvidia-smi` reports CUDA UMD 13.3; this is the driver capability, not the installed toolkit version. `CUDA_PATH` and `PATH` referenced a CUDA 11.8 directory that was absent on disk. No CUDA 12 cuBLAS DLL was present in the worker environment, CTranslate2 package, or configured toolkit location. The missing component was `cublas64_12.dll`.

The project-local fix is `nvidia-cublas-cu12==12.6.4.1` (cuBLAS for CUDA 12). It supplies `cublas64_12.dll`, `cublasLt64_12.dll`, and `nvblas64_12.dll`. The DLL directory is made visible to the worker process before CTranslate2 loads. CTranslate2 4.8.2 and faster-whisper 1.2.1 were not changed. Tiny, base, and small all completed transcription using CUDA / float16 on the 11.0-second JFK fixture. Times include cold model load plus transcription; audio extraction and model downloads are excluded.

| Model | Device | Compute | Time | RTF | Sampled whole-GPU memory | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| tiny | cuda | float16 | 2.639 s | 0.2399 | 209 MiB / 4096 MiB | Passed; 1 segment |
| base | cuda | float16 | 0.852 s | 0.0774 | 313 MiB / 4096 MiB | Passed; 1 segment |
| small | cuda | float16 | 1.982 s | 0.1802 | 769 MiB / 4096 MiB | Passed; 1 segment |

`nvidia-smi` memory is a sampled whole-device reading, not process-attributed allocation or an exact peak. PyTorch allocated, reserved, and peak memory are unavailable because PyTorch is not installed. The benchmark processes models in one process in tiny/base/small order: tiny includes CUDA context startup, while later models reuse the initialized context. These are observed run timings, not equivalent cold-start measurements. The tiny CLI run also wrote a transcript successfully. The existing CPU benchmark remains separately recorded in `docs/phase-1a.md`; its measurements use the same short fixture and are not directly comparable as quality claims.

## Manual integration check

Place a short `sample.mp4` in `input/`, make sure FFmpeg (system or packaged) and the desired CUDA runtime are available, then run the CLI command above. It should extract temporary audio, run faster-whisper, and produce `output/transcript.json`. The extracted WAV is deleted after the command finishes. No sample media or model weights should be committed.
