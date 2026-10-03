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
- Hugging Face access: configured locally; Community-1 model loading and inference completed. No token is stored in this repository.

The original `worker/.venv/` remains at faster-whisper 1.2.1, CTranslate2 4.8.2, imageio-ffmpeg 0.6.0, and nvidia-cublas-cu12 12.6.4.1. No machine-wide CUDA Toolkit or driver was installed or changed.

Pyannote emits a TorchCodec warning on Windows because its DLL loader cannot load `libtorchcodec_core8.dll` (or a required dependency). The diagnostic names a full-shared FFmpeg build, Torch/TorchCodec compatibility, or another runtime DLL as possible causes; the exact missing dependency was not identified. DubFlow keeps the matched PyTorch/pyannote stack and uses its FFmpeg extractor to decode and resample audio, then supplies pyannote an in-memory `{"waveform": tensor, "sample_rate": 16000}` mapping. This avoids TorchCodec file decoding while the import warning remains.

### Hugging Face access

Community-1 is gated and licensed CC BY 4.0. To configure access without putting a token in this repository or chat:

1. In a browser, sign in to the Hugging Face account and accept the [Community-1 model conditions](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. In PowerShell from the DubFlow root, set `UV_PROJECT_ENVIRONMENT` as shown above and run `worker/.venv-diarization/Scripts/hf.exe auth login`. Complete its browser/token prompt; Hugging Face stores the credential in its local user cache, outside the repository.
3. Verify only the presence of authentication with `worker/.venv-diarization/Scripts/python.exe -c "from huggingface_hub import get_token; print('HF_TOKEN configured:', bool(get_token()))"`.

Current status: isolated environment setup, gated model access, model loading, and real CUDA inference are verified. Earlier statements about unavailable model access describe the initial setup attempt and have been superseded by the benchmark below.

## Windows decoding workaround and benchmark result

`AudioExtractor` asks FFmpeg to decode the input directly to a temporary mono, 16 kHz, 16-bit PCM WAV. The new waveform loader converts that normalized WAV to a float32 tensor shaped `(channel, time)` and passes the tensor mapping to pyannote. This keeps the audio-loading route inside DubFlow's existing FFmpeg infrastructure and does not read the original media into Python before conversion. The diarization model, CUDA device, and speaker constraints remain unchanged.

The requested command completed successfully on `input/phase1b1/two-voice-alternating.wav`:

| Measurement | Result |
| --- | ---: |
| Model load | 13.671 s |
| Diarization inference | 46.895 s |
| Total runtime, including decode | 62.658 s |
| RTF | 1.4125 |
| Detected speakers | 2 |
| Diarization segments | 9 |
| Sampled whole-device GPU memory | 3925 MiB / 4096 MiB |
| Error | None |

The pipeline loaded successfully and then performed real CUDA inference; these are separate successes, and both occurred. The report is at `output/phase1b1/diarization-benchmark.json`. The model's output included a final turn extending past the waveform duration, so the benchmark clips returned turns to the audio bounds before validating and serializing them.

Re-run from the DubFlow root with:

```powershell
worker/.venv-diarization/Scripts/python.exe worker/scripts/benchmark_diarization.py input/phase1b1/two-voice-alternating.wav --model pyannote/speaker-diarization-community-1 --device cuda --min-speakers 2 --max-speakers 2 --output output/phase1b1/diarization-benchmark.json
```

## Sample

`input/phase1b1/two-voice-alternating.wav` is 33.2 seconds and is gitignored. It was assembled locally from the existing Phase 1A.2 LibriSpeech test-clean English utterances and an FPT Open Speech Dataset Vietnamese utterance already present in `input/phase1a2/`. Both sources were previously documented as CC BY 4.0 in [Phase 1A.2](phase-1a2.md). Four 8-second excerpts alternate between source recordings with 400 ms silences. Expected boundaries and source-group labels are in the ignored `input/phase1b1/two-voice-alternating.references.json`.

This is a controlled two-voice engineering fixture, not natural dialogue. It permits a basic check of speaker count, turn ordering, and boundary behavior. The speaker labels mark the two source recordings; they are not a formal independently annotated conversational reference and do not support an accuracy score.

## Implementation and merge prototype

`dubflow_worker.diarization.base` validates finite positive turn ranges, stable speaker labels, and audio duration. The optional pyannote adapter loads `pyannote/speaker-diarization-community-1` on the requested device. The benchmark result is versioned JSON with audio metadata, engine/model/device, speaker labels, and ordered turns.

`assign_speakers` is a small pure helper that copies ASR segments and assigns the label with the largest temporal overlap. Segments with no overlap receive `speaker: null`. The ASR pipeline and transcript schema are not changed.

## Hardware and benchmark status

Target hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB VRAM, driver 610.62, Windows, Python 3.11.17. The benchmark measured 3925 MiB of whole-device GPU memory at its sampled peak, close to device capacity. Whole-device samples may miss short peaks and include other GPU users. The JSON also reports PyTorch allocator counters separately. In this run, their reserved/peak values (9470 MiB) exceeded physical VRAM and disagreed with the sampled device measurement; treat those allocator figures as unreliable on this Windows run.

## Limitations and next-step suitability

- Community-1 requires a Hugging Face account, accepted model conditions, and a token or already accessible local cache.
- GPU inference completed on this fixture, but the sampled whole-device peak was close to the RTX 3050 Laptop GPU capacity. This does not establish fit for longer media or concurrent workloads.
- The benchmark establishes successful GPU inference and runtime on this fixture. The sampled whole-device peak was close to the RTX 3050 Laptop GPU capacity; the synthetic fixture does not establish diarization quality for natural conversation.
- The synthetic, alternating English/Vietnamese sample is much easier and less representative than overlapping natural conversation.
- Maximum-overlap ASR assignment is a timestamp-based prototype; it does not resolve word-level overlap or diarization errors.

The benchmark is ready for later product evaluation after listening-based inspection on representative dialogue. This document does not start that next phase.

### Earlier Phase 1B.1 attempt

An earlier setup attempt failed before model loading because the diarization dependencies were not yet installed. That historical failure was superseded by the successful run above; the JSON report at `output/phase1b1/diarization-benchmark.json` now contains the successful inference results.

## Verification

Full worker test suite, including FFmpeg waveform decoding and tensor shape/sample-rate checks: **23 passed**. The helper test uses CPU audio tensors and does not require CUDA inference.
