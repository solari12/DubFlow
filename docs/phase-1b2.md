# Phase 1B.2 - Speaker-aware transcription benchmark

## Scope

This phase runs faster-whisper `base` ASR and pyannote Community-1 diarization sequentially on `input/phase1b1/two-voice-alternating.wav`, then assigns a speaker to each ASR segment by temporal overlap. It does not change the Phase 1B.1 diarization implementation, ASR model stack, transcript model, or add later product infrastructure.

The ASR model runs in `worker/.venv/` on CUDA with float16. The ASR benchmark saves its transcript and closes the ASR engine; the ASR subprocess then exits. Only after that process has terminated does the runner start the Phase 1B.1 diarization benchmark in `worker/.venv-diarization/`. The diarization stage uses CUDA and Community-1 with minimum and maximum speakers both set to 2. Its FFmpeg-based in-memory waveform input avoids TorchCodec file decoding. Thus the two GPU models do not hold their resources at the same time.

`dubflow_worker.pipeline.speaker_transcription.merge_speaker_transcript` compares each ASR interval against every diarization interval. It selects the single diarization interval with the largest temporal overlap and computes `overlap_seconds / ASR_segment_duration`. The default threshold is 0.20; a speaker is assigned when the best overlap is positive and the ratio is at least 0.20. Zero overlap and ratios below threshold produce `speaker: null`. The output retains the selected overlap seconds and ratio for auditing.

## Run

From the DubFlow root, after installing the ASR and diarization environments and accepting Community-1's gated model conditions:

```powershell
worker/.venv/Scripts/python.exe worker/scripts/benchmark_speaker_transcription.py `
  input/phase1b1/two-voice-alternating.wav `
  --output output/phase1b2/speaker-transcription-benchmark.json
```

The runner uses the existing ASR and diarization benchmark scripts as separate child processes. It creates the output directory if needed. The report's stage runtimes are the ASR processing time (model load plus transcription) and the pyannote inference time; total runtime includes both subprocesses, their setup/model loading and waveform preparation, and speaker merging. Each stage RTF divides its reported runtime by the 33.2-second audio duration. Whole-GPU memory is the largest sampled `nvidia-smi` reading from either stage.

## Benchmark result

The command above completed successfully on the two-voice fixture:

| Measurement | Result |
| --- | ---: |
| ASR model | faster-whisper `base`, CUDA / float16 |
| ASR runtime | 8.016 s |
| ASR RTF | 0.2414 |
| Diarization model | pyannote Community-1, CUDA |
| Diarization inference runtime | 42.078 s |
| Diarization RTF | 1.2674 |
| Speaker merge runtime | 0.000057 s |
| Total runtime | 76.640 s |
| Detected speakers | 2 |
| ASR segments | 4 |
| Assigned / unassigned | 3 / 1 |
| Sampled whole-GPU memory | 3925 MiB / 4096 MiB |
| OOM | No |

The JSON report is `output/phase1b2/speaker-transcription-benchmark.json`. It contains the actual ASR text and speaker label, selected overlap seconds, and overlap ratio for each transcript segment. Segment 2 had no diarization overlap and remains unassigned. Segment 3 contains Vietnamese ASR text and was assigned `SPEAKER_01` at a 0.45 overlap ratio. This synthetic alternating fixture confirms execution and merge behavior; it is not a natural-dialogue accuracy evaluation. The sampled GPU usage was close to capacity, so this result does not establish fit for longer media or concurrent GPU workloads.

## Verification

The full worker test suite passes in the diarization environment: **30 passed**. The suite covers clear overlap, competing speakers, zero overlap, below-threshold overlap, touching boundaries, multiple transcript segments, and assignment exactly at the threshold. In the default ASR environment, **29 passed and one Torch-dependent waveform test was skipped** because Torch intentionally remains isolated in the diarization environment.

The ASR and diarization subprocesses are run synchronously in that order. The benchmark does not launch diarization unless ASR completed successfully and produced its transcript. Successful completion of both child benchmarks and creation of the merged four-segment JSON validate the sequential execution and speaker assignment on the real fixture.
