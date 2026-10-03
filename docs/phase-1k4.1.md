# Phase 1K.4.1 — KorvaTTS CUDA benchmark

## Scope

This is an isolated, benchmark-only CUDA attempt using the Phase 1K.4 fixture and the installed KorvaTTS candidate. Piper remains the production default. Translation selection and production translation, TTS, alignment, rendering, and video validation behavior were not changed. No full-video, alignment, or rendering workload was run.

## Environment

- Windows 10, Python 3.11.17, NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB), driver 610.62, nvidia-smi CUDA UMD 13.3.
- Isolated `worker/.venv-tts-korva`: KorvaTTS 0.1.3 and ONNX Runtime GPU 1.30.0. The CPU ONNX Runtime package was replaced only inside this candidate environment; the Piper environment was not changed.
- `CUDAExecutionProvider` is reported by ONNX Runtime and its DLL preload succeeds. This confirms that the provider package is discoverable; it does not guarantee that every Korva model graph can be assigned without CPU execution.
- PyTorch is not installed. Korva uses ONNX Runtime, so PyTorch allocated/reserved memory metrics are null and would not measure its CUDA allocations.

## CUDA-only outcome

Korva CUDA initialization failed. To prevent hidden CPU execution, the benchmark requests only `CUDAExecutionProvider` and sets `session.disable_cpu_ep_fallback=1`. ONNX Runtime rejects a model session because at least one graph node is assigned to the default CPU execution provider. The error is recorded with the model file name in `output/tts-benchmark-v1/korvatts-cuda/report.json`.

No warm-up or measured CUDA synthesis completed; all five fixture cases and the three repeated original runs are recorded as failed/skipped. No CUDA WAVs were created. CUDA RTF and process GPU memory are unavailable. The RTX 3050's VRAM sufficiency is undetermined because failure happened during graph initialization, before a successful synthesis or useful model allocation measurement. This is not classified as OOM.

The CPU baseline is preserved in `output/tts-benchmark-v1/korvatts/report.json`: original audio duration 13.5605 s, synthesis 36.3497 s, RTF 2.6806, and process peak working set 685.12 MiB. During the failed CUDA attempt, sampled system-wide VRAM was 0 MiB before and 63 MiB peak/after; these coarse samples are not process-level model usage. This Windows driver mode returned `[N/A]` for per-process VRAM, so that metric is unavailable. The failed process peak working set is reported separately and is not GPU memory.

## Artifacts

- `output/tts-benchmark-v1/korvatts-cuda/report.json` — environment, strict provider policy, initialization error, skipped cases/runs, CPU baseline, and separated system/process GPU telemetry.
- `output/tts-benchmark-v1/korvatts-cuda/benchmark-report.md` — human-readable summary.
- `output/tts-benchmark-v1/korvatts-cuda/dependency-audit-before.json` — candidate environment package/provider state before installing the CUDA runtime.

## Reproduce

From PowerShell in `D:\Project\DubFlow\worker`:

```powershell
& '.venv-tts-korva/Scripts/python.exe' 'scripts/benchmark_tts_v1.py' --provider korva --device cuda
```

The command exits nonzero when strict CUDA initialization or any CUDA synthesis fails. It never retries the candidate on CPU. To reproduce the original CPU baseline, use the same script with `--provider korva --device cpu`; this keeps the CPU request explicit even though the candidate environment now contains ONNX Runtime GPU.

## Compatibility result

`CUDA_FAILED` for this KorvaTTS model/runtime configuration. ONNX Runtime's CUDA provider is available, but the complete model graphs cannot initialize with CPU fallback disabled. Since fallback is intentionally disabled, no performance or pronunciation comparison was possible. Piper remains the production TTS provider; this phase does not promote or configure Korva for production use.
