# Phase 1K.4.2 — Kokoro Vietnamese TTS benchmark

## Scope and reproducibility

This phase benchmarks the PyTorch implementation of `iamdinhthuan/Kokoro-Vietnamese` against the Phase 1K.4 fixture on CPU and explicit CUDA. It does not add Kokoro to production or change TTS, translation, alignment, or rendering behavior. No automatic audio quality score or winner was selected.

- Fixture: `output/tts-benchmark-v1/fixture.json`, used unchanged (SHA-256 `1231fcaa1c6aaab76db8ff2ac6eeec8507057f253490c5c759eac7d4712ca56a`).
- Pinned upstream package commit: `a249afe5555aec6c435165c2f61ec0f71284812f`; package `kokoro-vietnamese==0.1.0`.
- Model: `contextboxai/Kokoro-Vietnamese`, pinned revision `9f210d622209fcc216fe2ac6159fed2ff381cb8a`.
- Environment: isolated `worker/.venv-tts-kokoro`; Python 3.11.17, `vig2p==0.1.2`, `transformers==4.57.6`, `torch==2.11.0+cu130`, CUDA 13.0.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM.
- Voices: `diem_trinh`, `thanh_dat`, and `mai_linh`. The upstream voice registry does not specify voice gender or timbre; all voices remain subject to listening review.
- Run: `worker/.venv-tts-kokoro/Scripts/python.exe worker/scripts/benchmark_kokoro_v2.py` from the repository root. CPU is run before CUDA. CUDA initialization is checked explicitly and cannot silently fall back to CPU.

For each voice/device pair, one warm-up synthesis was discarded and the original full-text case was measured three times. Four additional fixture cases (normalized text and isolated Hiragana, Katakana, and Kanji) were synthesized once each. All generated WAVs are 24 kHz, mono PCM. Detailed run records, measurements, and paths are in [`benchmark.json`](../output/tts-benchmark-v2/kokoro/benchmark.json), with an at-a-glance report in [`benchmark-report.md`](../output/tts-benchmark-v2/kokoro/benchmark-report.md). Listening results are intentionally blank in [`quality-review.md`](../output/tts-benchmark-v2/kokoro/quality-review.md).

## Original full-text results

RTF is synthesis time divided by generated audio duration; lower is faster. Times are arithmetic means from three measured runs (the report includes min and max).

| Voice | CPU average seconds / RTF | CUDA average seconds / RTF | CUDA PyTorch peak allocated / reserved MiB | Sampled whole-device VRAM peak MiB |
|---|---:|---:|---:|---:|
| Diem Trinh (`diem_trinh`) | 8.483 / 0.658 | 0.848 / 0.066 | 424.358 / 492 | 587 |
| Thanh Dat (`thanh_dat`) | 9.244 / 0.639 | 0.843 / 0.058 | 426.434 / 480 | 575 |
| Mai Linh (`mai_linh`) | 8.981 / 0.713 | 0.833 / 0.066 | 422.404 / 536 | 631 |

All 6 voice/device runs completed, and 42 WAV files were generated. The PyTorch allocator figures are process-level CUDA measurements. The `nvidia-smi` values are sampled whole-device memory, including other GPU usage; they are not process-attributed. Per-process peak working set is separately included in the JSON report.

## Existing-candidate comparison

The benchmark report records Piper CPU at RTF 0.034011 and KorvaTTS CPU at RTF 2.680557 on the existing fixture. The earlier Korva CUDA attempt failed because ONNX Runtime assigned graph nodes to its CPU execution provider while CPU fallback was disabled; it was not an out-of-memory failure. These are benchmark results, not a quality ranking: synthesis speed alone does not determine which voice sounds best or is suitable for dubbing.

## Human review and outcome

The 30 voice/device/case combinations in the listening sheet are marked `NOT_REVIEWED`. Human listening is needed for Vietnamese tone accuracy, pronunciation of Japanese writing-system names, naturalness, artifacts, pauses, sentence rhythm, and voice suitability. No production provider or behavior changed. The next decision depends on reviewing the generated audio and selecting a preferred voice (or deciding not to use this candidate).
