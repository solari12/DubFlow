# Phase 1A — Python AI Worker + ASR Vertical Slice

## Scope

Phase 1A is a local, independently runnable worker prototype:

```text
Local audio/video → FFmpeg extraction → ASR → timestamped transcript → JSON
```

There is no Spring Boot API, React application, PostgreSQL, Redis, authentication, URL downloading, translation, TTS, speaker diarization, subtitle editor, or production deployment in this phase.

## Architecture

```text
CLI
 ↓
TranscriptionPipeline
 ├── AudioExtractor → FFmpeg → temporary 16 kHz mono PCM WAV
 └── ASREngine interface → FasterWhisperASR → faster-whisper / CTranslate2
 ↓
Transcript model → stable JSON
```

The pipeline depends on `ASREngine`, not on faster-whisper directly. The audio artifact is held in a temporary directory and cleaned when transcription ends, including failure paths. Unit tests use fake extractor/ASR implementations and do not download models.

## ASR engine choice and model selection

faster-whisper is the first engine because its CTranslate2 backend supports efficient Whisper inference and timestamped segments. The adapter returns a common transcript model and optionally includes word timestamps. `base` is the conservative initial default for experimentation; this is not a claim that it will fit or perform well under every 4 GB GPU workload. `tiny`, `base`, and `small` are selectable by the benchmark.

Model file size and peak runtime VRAM are different measurements. The benchmark measures elapsed time and samples whole-device GPU memory using `nvidia-smi` when present. That sample is not attributable solely to DubFlow and can miss short-lived peaks. PyTorch allocator metrics are best-effort and can be unavailable because faster-whisper uses CTranslate2 rather than requiring PyTorch for inference.

## RTX 3050 constraints

The development target is an Intel Core i5-12500H, NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM, on Windows. CUDA is requested explicitly by default; if CTranslate2 cannot find a CUDA device, the worker returns an actionable error instead of silently falling back. CPU execution is available with `--device cpu --compute-type int8`.

Benchmark run on 2026-10-02 against the 11-second `jfk.flac` speech fixture from [OpenAI Whisper's test data](https://github.com/openai/whisper/blob/main/tests/test_transcribe.py), on CPU with `int8` and model weights already cached. Timings include loading each model into a fresh engine plus transcription, but exclude the one-time audio extraction and network download. This is one short English sample and is not a general quality or throughput claim.

| Model | Device / compute | Input duration | Processing time | RTF | Result |
| --- | --- | ---: | ---: | ---: | --- |
| tiny | CPU / int8 | 11.0 s | 1.826 s | 0.1660 | Passed; 1 segment |
| base | CPU / int8 | 11.0 s | 2.341 s | 0.2128 | Passed; 1 segment |
| small | CPU / int8 | 11.0 s | 6.730 s | 0.6118 | Passed; 1 segment |

CUDA device enumeration reports one device, and `nvidia-smi` identifies the RTX 3050 with 4096 MiB total memory. The machine has driver 610.62 (`nvidia-smi` CUDA UMD 13.3). `CUDA_PATH` and `PATH` referenced a CUDA 11.8 directory that was absent on disk. CTranslate2 4.8.2 needs CUDA 12.x and cuDNN 9 for GPU speech recognition. Its first inference failed because the required `cublas64_12.dll` was absent from the worker environment, CTranslate2 package, and configured toolkit location.

The fix adds the optional worker dependency `nvidia-cublas-cu12==12.6.4.1`. It provides `cublas64_12.dll`, `cublasLt64_12.dll`, and `nvblas64_12.dll` from the project virtual environment. The worker adds this package's `nvidia/cublas/bin` directory to its process-local DLL search path before importing CTranslate2. The CTranslate2 installation already contains `cudnn64_9.dll`. No driver, system CUDA installation, faster-whisper version, or CTranslate2 version was changed. `uv sync --project worker --extra cuda` installs this optional runtime; include `--extra dev` as well for the test dependencies.

CUDA inference was then verified end to end. The 11-second JFK fixture produced one transcript segment for each model. Timings include loading each model into a fresh engine plus transcription; they exclude audio extraction and model download.

| Model | Device | Compute | Time | RTF | Sampled whole-GPU memory | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| tiny | cuda | float16 | 2.639 s | 0.2399 | 209 MiB / 4096 MiB | Passed; 1 segment |
| base | cuda | float16 | 0.852 s | 0.0774 | 313 MiB / 4096 MiB | Passed; 1 segment |
| small | cuda | float16 | 1.982 s | 0.1802 | 769 MiB / 4096 MiB | Passed; 1 segment |

These `nvidia-smi` figures are sampled whole-GPU readings, not per-process allocation or exact peak VRAM. PyTorch allocated, reserved, and peak counters are unavailable because PyTorch is not a worker dependency. The benchmark processes models in one process in tiny/base/small order: tiny includes CUDA context startup, while later models reuse the initialized context. Treat these as observed timings rather than equivalent cold-start measurements. The benchmark output is in `output/cuda-benchmark.json` for this local run. The earlier CPU tiny/base/small timings above remain a separate CPU record; they are measurements on the same short fixture, not general performance claims.

## Benchmark methodology

The script extracts the input once, then for each requested model measures cold model load plus transcription. It reports model, device, compute type, audio duration, processing seconds, real-time factor (processing time divided by audio duration), success/error, and observed GPU memory if available. Extraction time is excluded so the timing compares ASR configurations. Run on a short representative sample first and record the machine state and chosen options alongside results. A failed model run is a result; it is not replaced with an invented estimate.

## Output schema

The CLI writes UTF-8 JSON with version, source filename and extracted-audio duration, ASR engine/model/language, and an array of segments with integer IDs, start/end seconds, and text. Word timestamps are included only when the engine returns them. Speaker identifiers are intentionally out of scope.

## Manual integration test

1. Install Python 3.11, worker dependencies, and a compatible CUDA runtime (or choose CPU mode). The worker uses system FFmpeg if found, otherwise its `imageio-ffmpeg` binary.
2. Place a short `sample.mp4` in `input/`.
3. From the repository root run `uv run --project worker python -m dubflow_worker.cli transcribe input/sample.mp4 --model base --device cuda --output output/transcript.json`.
4. Confirm extraction completes, faster-whisper returns timestamped segments, the output JSON is parseable, and temporary extracted audio is removed.
5. Repeat with `--device cpu --compute-type int8` to check CPU fallback.

Do not commit large media samples or downloaded model weights.

## Known limitations

- URL downloading, diarization, translation, TTS, editing, and web APIs are out of scope.
- FFmpeg must be available on PATH, configured explicitly, or provided by the installed `imageio-ffmpeg` package.
- Model downloads require network access on first use; cached local model paths can be selected with the model setting.
- CUDA inference is verified for tiny, base, and small with float16 on the development RTX 3050 using the optional project-local cuBLAS runtime. Performance and fit may differ on other inputs or when other processes use GPU memory.
- `nvidia-smi` sampling reports whole-device usage, not reliable per-process peak VRAM.
- The 11-second Phase 1A.1 results alone did not establish fit for longer audio. Phase 1A.2 subsequently verified all three models on a 179-second English sample; concurrent workloads, other inputs, and other GPU memory states remain untested.

## Future Java / Redis integration

The worker remains independent from Spring Boot and Redis. A later phase can wrap its request/result boundary in a job consumer. The current JSON is a prototype contract; the Java API should version and validate its own contract before production use. Keep media references and outputs outside PostgreSQL, persisting metadata there instead.

## Benchmark record

CPU and CUDA benchmarks are recorded separately above for `tiny`, `base`, and `small` on the short sample described above. CUDA transcription succeeds with the documented worker-local CUDA runtime.

Phase 1A.2 adds Vietnamese and longer English evaluation, reference-based WER/CER, saved transcripts, and timestamp diagnostics. See [Phase 1A.2 benchmark report](phase-1a2.md).
