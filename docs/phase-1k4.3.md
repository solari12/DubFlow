# Phase 1K.4.3 - KorvaTTS CUDA benchmark

## Scope

This was a benchmark-only run using the unchanged Phase 1K.4 fixture and the isolated `worker/.venv-tts-korva` environment. It did not modify production behavior or prior benchmark artifacts. Voice `gia_bao` was used because the shared fixture fixes text/normalization but no voice; the earlier Korva CPU artifact used `khanh_vy`.

## Result

The official run is **`CUDA_FAILED` under the phase's no-CPU-provider-fallback criterion**. All four ONNX sessions listed `CUDAExecutionProvider` first, and an independent ORT profile probe confirmed CUDA node execution. It also recorded **6,071 `CPUExecutionProvider` node events** (46,472 CUDA node events), so the provider did fall back for some graph nodes. This is not a full-device CPU fallback: all benchmark WAVs were generated with the upstream `device="gpu"` API, but the run cannot be reported as CUDA-only.

The full-text case still produced descriptive timing data: 1 warm-up, 3 measured runs, mean synthesis 1.697 s, mean total wall 1.712 s, mean audio duration 12.318 s, and mean RTF 0.138. The measured run range was 1.663-1.718 s synthesis and RTF 0.135-0.139. The 5 main fixture WAV cases validated as readable nonempty PCM WAV files at 44.1 kHz mono.

GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 610.62. Coarse system-wide `nvidia-smi` sampled memory peaked at 1345 MiB. Process-level VRAM is unavailable in this Windows WDDM mode; it is not inferred from system-wide memory.

## Descriptive comparison

The detailed table is in [`benchmark-report.md`](../output/tts-benchmark-v3/korvatts-cuda/benchmark-report.md), sourced from the existing Piper, Korva CPU, and Kokoro artifacts plus this run. Korva CPU used `khanh_vy`; this CUDA run used `gia_bao`, so their comparison is descriptive rather than a same-voice controlled comparison. No ranking or winner is selected.

All 5 human quality review items remain `NOT_REVIEWED` in [`quality-review.md`](../output/tts-benchmark-v3/korvatts-cuda/quality-review.md) and [`listening-sheet.json`](../output/tts-benchmark-v3/korvatts-cuda/listening-sheet.json).

## Verification

Focused benchmark tests cover fixture loading, CUDA-provider verification, CPU fallback detection, RTF, WAV metadata, report generation, and listening-sheet status. The benchmark records CUDA initialization failures, full-device CPU fallback separately from CPU EP node fallback, failed cases, and WAV validation failures. ORT profile traces are summarized by node-provider counts and discarded after parsing; they can otherwise be tens of megabytes.

The user-provided manual GPU command result is prior smoke-test context. The official benchmark's profile found CPU EP node events, so the phase's requested strict CUDA compliance is not met even though CUDA was active.
