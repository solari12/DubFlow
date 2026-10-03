# Phase 1B.1 — Speaker Diarization Benchmark

## Scope and VoiceStudio review

This phase adds a local diarization benchmark and a small schema/merge prototype. It does not extend the transcription pipeline. The Phase 0 audit in the project README identified VoiceStudio's WhisperX/pyannote path as an adaptation candidate and warned that VoiceStudio is AGPL-3.0-only. The focused [VoiceStudio diarization notes](https://github.com/debpalash/VoiceStudio/blob/main/docs/features/diarization.md) describe its `pyannote/speaker-diarization-3.1` integration, gated Hugging Face access, local model installation, and a silence-gap fallback. No VoiceStudio code was copied.

The benchmark candidate is the newer `pyannote/speaker-diarization-community-1` pipeline from `pyannote.audio` 4.0.7. It provides speaker turns and an exclusive diarization view, and supports mapping its annotation output into the simple DubFlow result. The pipeline bundle includes its configuration, a speaker segmentation checkpoint, a speaker embedding checkpoint, and PLDA/clustering data; these are downloaded together from the gated repository. The model card declares CC BY 4.0 and requires accepting Hugging Face's access conditions before downloading. The library is MIT-licensed. Model and library licenses are separate from the VoiceStudio application license.

## Phase 1B.1.1 — Environment and model access

Before this setup, PyTorch, torchvision, torchaudio, and pyannote.audio were absent from the worker environment. Python 3.11 is supported: pyannote.audio 4.0.7 requires Python 3.10 or newer, Torch and TorchAudio 2.8 or newer, and TorchCodec 0.7 or newer; the other pipeline dependencies are installed transitively. The chosen exact set is Torch 2.11.0, TorchAudio 2.11.0, and TorchCodec 0.15.0. TorchCodec 0.15 supports Torch 2.11 or newer. CUDA 13.0 Windows wheels exist for this Python version. [pyannote 4.0.7 requirements](https://raw.githubusercontent.com/pyannote/pyannote-audio/4.0.7/pyproject.toml), [TorchCodec compatibility](https://github.com/meta-pytorch/torchcodec/releases), [PyTorch CUDA 13.0 wheels](https://download.pytorch.org/whl/cu130/torch/).

The diarization set is installed in ignored `worker/.venv-diarization/`, separate from the existing ASR environment. The optional `diarization` extra pins the matched CUDA packages and pyannote.audio 4.0.7; `worker/uv.lock` is updated. Run setup from the project root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) 'worker/.venv-diarization'
uv sync --project worker --extra diarization --no-install-project
```

Verified in that isolated environment:

- Torch: `2.11.0+cu130`; `torch.version.cuda`: `13.0`.
- `torch.cuda.is_available()`: `True`; device: NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB), driver 610.62.
- `pyannote.audio`: `4.0.7`; `from pyannote.audio import Pipeline` succeeds.
- torchvision is not installed and is not required by pyannote.audio.
- Hugging Face authentication: **not configured** (checked environment variables and local Hugging Face auth cache without printing token contents).

The original `worker/.venv/` remains at faster-whisper 1.2.1, CTranslate2 4.8.2, imageio-ffmpeg 0.6.0, and nvidia-cublas-cu12 12.6.4.1. No machine-wide CUDA Toolkit or driver was installed or changed.

Pyannote’s import emits a TorchCodec warning: Windows could not load `libtorchcodec_core8.dll` or one of its dependencies, so file-based audio decoding is not verified. Pyannote supports in-memory waveform input; the current adapter still passes a file path. This must be resolved or the adapter changed before running the audio fixture. No model load or inference was attempted.

### Hugging Face access

Community-1 is gated and licensed CC BY 4.0. To configure access without putting a token in this repository or chat:

1. In a browser, sign in to the Hugging Face account and accept the [Community-1 model conditions](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. In PowerShell from the DubFlow root, set `UV_PROJECT_ENVIRONMENT` as shown above and run `worker/.venv-diarization/Scripts/hf.exe auth login`. Complete its browser/token prompt; Hugging Face stores the credential in its local user cache, outside the repository.
3. Verify only the presence of authentication with `worker/.venv-diarization/Scripts/python.exe -c "from huggingface_hub import get_token; print('HF_TOKEN configured:', bool(get_token()))"`.

Current status: **Environment is ready, but Hugging Face model access is not configured.** Per the phase stop condition, the model was not loaded. There is no model initialization time or post-load GPU memory observation.

### Future benchmark command (not run in this phase)

After model access is configured and the TorchCodec audio-path warning is resolved, the separate environment can run the Phase 1B.1 benchmark:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) 'worker/.venv-diarization'
uv run --project worker --extra diarization python worker/scripts/benchmark_diarization.py `
  input/phase1b1/two-voice-alternating.wav `
  --model pyannote/speaker-diarization-community-1 `
  --device cuda --min-speakers 2 --max-speakers 2 `
  --output output/phase1b1/diarization-benchmark.json
```

That benchmark is intentionally outside Phase 1B.1.1. It records model-load and inference times, RTF, sampled whole-device memory separately from PyTorch process allocation/peaks, and structured speaker turns. It does not silently fall back to CPU.

## Sample

`input/phase1b1/two-voice-alternating.wav` is 33.2 seconds and is gitignored. It was assembled locally from the existing Phase 1A.2 LibriSpeech test-clean English utterances and an FPT Open Speech Dataset Vietnamese utterance already present in `input/phase1a2/`. Both sources were previously documented as CC BY 4.0 in [Phase 1A.2](phase-1a2.md). Four 8-second excerpts alternate between source recordings with 400 ms silences. Expected boundaries and source-group labels are in the ignored `input/phase1b1/two-voice-alternating.references.json`.

This is a controlled two-voice engineering fixture, not natural dialogue. It permits a basic check of speaker count, turn ordering, and boundary behavior. The speaker labels mark the two source recordings; they are not a formal independently annotated conversational reference and do not support an accuracy score.

## Implementation and merge prototype

`dubflow_worker.diarization.base` validates finite positive turn ranges, stable speaker labels, and audio duration. The optional pyannote adapter loads `pyannote/speaker-diarization-community-1` on the requested device. The benchmark result is versioned JSON with audio metadata, engine/model/device, speaker labels, and ordered turns.

`assign_speakers` is a small pure helper that copies ASR segments and assigns the label with the largest temporal overlap. Segments with no overlap receive `speaker: null`. The ASR pipeline and transcript schema are not changed.

## Hardware and benchmark status

Target hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM, driver 610.62, Windows, Python 3.11.17. Torch CUDA is now verified in the isolated diarization environment, but the model remains gated and no Hugging Face authentication is configured. Therefore **no model-load, model VRAM, inference, RTF, or detected-speaker result is claimed**.

Sampled `nvidia-smi` memory is whole-device usage and may miss short peaks; it is not process allocation. PyTorch's peak allocated/reserved counters are separate process-level observations and are only populated for CUDA runs. No CUDA outcome should be inferred from a model access/authentication failure.

## Limitations and next-step suitability

- Community-1 requires a Hugging Face account, accepted model conditions, and a token or already accessible local cache.
- CUDA requires a CUDA-enabled PyTorch build compatible with the installed NVIDIA driver. The pyannote package alone does not prove CUDA support or fit within 4 GB.
- The benchmark has not yet established GPU fit, speed, memory, or diarization quality on this laptop because the model could not be loaded without model access and a PyTorch runtime.
- The synthetic, alternating English/Vietnamese sample is much easier and less representative than overlapping natural conversation.
- Maximum-overlap ASR assignment is a timestamp-based prototype; it does not resolve word-level overlap or diarization errors.

This candidate is suitable for continuing the benchmark only after Hugging Face model access and CUDA PyTorch are available. The next phase should choose a production approach only after a successful local run and listening-based inspection; this document does not start that phase.

### Earlier Phase 1B.1 attempt

The exact run command above was attempted on the 33.2-second fixture. It failed before loading the model because `pyannote.audio` and PyTorch are not installed:

```text
RuntimeError: Diarization dependencies are missing. Install the worker's diarization extra and a CUDA-enabled PyTorch build for GPU inference.
```

The JSON report at `output/phase1b1/diarization-benchmark.json` records `success: false` and the pre-install missing-runtime error. That report predates this environment setup; no benchmark was rerun in Phase 1B.1.1.

## Verification

GPU-independent unit tests cover result parsing/serialization, timestamp validation, and maximum-overlap ASR assignment. Full-suite verification after environment setup: **22 passed**.
