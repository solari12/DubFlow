# Phase 1A.2 — Real-World ASR Benchmark

## Purpose and setup

This evaluation compares faster-whisper `tiny`, `base`, and `small` on Vietnamese and English speech, checks a nearly three-minute input, and inspects the transcript timestamps produced by the existing worker. It does not add ASR models or change the transcription architecture.

Run date: 2026-10-03. Machine: Windows, NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, driver 610.62 (`nvidia-smi` CUDA UMD 13.3). Project environment: Python 3.11.17, faster-whisper 1.2.1, CTranslate2 4.8.2, imageio-ffmpeg 0.6.0, CUDA cuBLAS package `nvidia-cublas-cu12==12.6.4.1`. All cases used CUDA and float16. The runtime setup is documented in [worker/README.md](../worker/README.md).

## Samples

The workspace previously had only the 11-second JFK English fixture. The benchmark fixtures below were downloaded as individual utterances, combined locally into mono 16 kHz WAV files, and kept under the gitignored `input/phase1a2/` directory. No corpus archive was downloaded and no media or transcript output is tracked.

| Case | Source and license | Construction | Duration |
| --- | --- | --- | ---: |
| `vietnamese` | [FPT Open Speech Dataset mirror](https://huggingface.co/datasets/doof-ferb/fpt_fosd), dataset card declares CC BY 4.0 | Four Vietnamese utterance records (viewer rows 36, 51, 60, 70), concatenated in row order; dataset reference text retained | 55.859 s |
| `english-short` | [LibriSpeech ASR](https://huggingface.co/datasets/openslr/librispeech_asr), CC BY 4.0 | First ten consecutive clips from one speaker/chapter in `test.clean`; reference text retained | 91.525 s |
| `english-long` | Same LibriSpeech source and license | First 21 consecutive clips from that speaker/chapter; reference text retained | 178.885 s |

These are read or sentence-level recordings, not conversational recordings. The concatenated Vietnamese sample joins utterances from different records, and the English samples join pre-segmented clips. They provide a longer input and known text for this initial check, but do not represent spontaneous dialogue, noise, or all Vietnamese accents. Dataset reference text is used as supplied and was not independently corrected by hand. No noisy sample was tested.

## Reproduction

Run from `D:\Project\DubFlow` after placing the local fixture WAVs and reference files in the paths shown:

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

The runner saves one transcript JSON per case/model and records the environment, success or error, duration, timing, RTF, segment count, transcript character count, timestamp checks, and sampled GPU memory. `NAME.txt` in the references directory enables WER/CER for that case. WER ignores case and punctuation and splits on whitespace; Vietnamese WER therefore counts space-separated orthographic syllables. CER ignores case, punctuation, and whitespace while retaining Vietnamese diacritics. The report is in `output/asr-real-world-benchmark.json`; transcripts are in `output/phase1a2/transcripts/`.

Audio extraction is excluded from processing time. Models run in tiny/base/small order in one process per case; the first inference initializes the CUDA context, so times are from this ordered run rather than equivalent cold-process starts. RTF is processing time divided by audio duration.

## Results

All nine model/case combinations succeeded on CUDA / float16. No CUDA, memory, or other runtime failure occurred. Times and WER/CER are measured results against the supplied dataset references.

| Case | Model | Duration | Time | RTF | Segments | Transcript chars | Sampled whole-GPU memory / 4096 MiB | WER | CER |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vietnamese (`vi`) | tiny | 55.859 s | 3.249 s | 0.0582 | 8 | 499 | 241 MiB | 0.6114 | 0.4895 |
| Vietnamese (`vi`) | base | 55.859 s | 1.779 s | 0.0318 | 13 | 747 | 345 MiB | 0.2857 | 0.1561 |
| Vietnamese (`vi`) | small | 55.859 s | 2.608 s | 0.0467 | 4 | 528 | 833 MiB | 0.4229 | 0.3491 |
| English (`en`), short | tiny | 91.525 s | 1.562 s | 0.0171 | 24 | 1409 | 265 MiB | 0.0159 | 0.0115 |
| English (`en`), short | base | 91.525 s | 1.933 s | 0.0211 | 22 | 1406 | 369 MiB | 0.0159 | 0.0098 |
| English (`en`), short | small | 91.525 s | 3.634 s | 0.0397 | 22 | 1410 | 921 MiB | 0.0120 | 0.0080 |
| English (`en`), long | tiny | 178.885 s | 2.799 s | 0.0156 | 44 | 2603 | 289 MiB | 0.0644 | 0.0443 |
| English (`en`), long | base | 178.885 s | 3.515 s | 0.0196 | 42 | 2690 | 425 MiB | 0.0353 | 0.0121 |
| English (`en`), long | small | 178.885 s | 6.306 s | 0.0353 | 43 | 2689 | 945 MiB | 0.0249 | 0.0070 |

The supplied references contain 175 whitespace tokens and 570 normalized characters for Vietnamese, 251 tokens and 1128 normalized characters for English short, and 481 tokens and 2146 normalized characters for English long. Reference errors can affect these scores, and the single Vietnamese case is too small to establish general language accuracy.

### Interpretation and initial model choice

On the Vietnamese sample, `base` had the lowest measured error (WER 28.6%, CER 15.6%) and was also fastest at 1.779 seconds. `small` used more sampled GPU memory and took longer, while scoring worse on this sample. `tiny` had the highest Vietnamese error.

On both English samples, `small` had the lowest reference error. It took 3.634 seconds on the short case and 6.306 seconds on the long case, versus 1.933 and 3.515 seconds for `base`. On the long case, `small` improved WER from 3.53% to 2.49% and CER from 1.21% to 0.70%; the absolute WER difference was about one percentage point. All models processed the 178.885-second file successfully.

**Recommendation: keep `base` as the initial DubFlow default.** Vietnamese is the initial target, and `base` was both faster and more accurate than `tiny` and `small` on the Vietnamese reference sample. This is a provisional engineering choice based on one assembled Vietnamese sample. The English results show a measured accuracy advantage for `small`; a larger collection of natural Vietnamese recordings could change the recommendation.

## Timestamp inspection

Across all nine outputs, segment start/end ranges were valid, segment starts were monotonic, and there were no segment overlaps greater than 50 ms. The maximum gaps between adjacent emitted segments were:

| Case | tiny | base | small |
| --- | ---: | ---: | ---: |
| Vietnamese | 0.32 s | 1.90 s | 20.26 s |
| English short | 1.00 s | 1.04 s | 1.02 s |
| English long | 1.00 s | 1.04 s | 1.02 s |

The large Vietnamese `small` gap is an omission, not a pipeline-created pause: its transcript skipped most of the fourth utterance and only resumed near the clip's end. `base` covered the full sample with no gap above 1.90 seconds. On Vietnamese, tiny also emitted four zero-duration word timestamps. English tiny emitted one on the short case and two on the long case. Base and small emitted no zero-duration or out-of-segment word ranges in these cases.

The base Vietnamese transcript segments were ordered and within the 55.859-second audio range. Manual text inspection found the broad meaning in each utterance but also clear word substitutions and malformed phrases, consistent with its nonzero reference error. Segment timing checks establish structural consistency only; this was not forced alignment or a detailed listening-based timing study. The local JSON files preserve all segments and word timestamps for manual review.

## GPU memory observations

The values in the results table are maximum sampled `nvidia-smi memory.used` readings for the whole GPU during each model run. They include any other GPU use and can miss short peaks; they are not exact per-process VRAM peaks. PyTorch is not installed in this worker, so allocated and reserved allocator values were not available. The 4096 MiB value is total reported device memory. No model produced an out-of-memory error in these runs.

## Limitations

- Vietnamese quality is measured on one 55.859-second assembled sample of four utterances, not spontaneous conversation.
- English samples are read audiobook speech from one speaker/chapter and overlap in their first 91.525 seconds.
- WER/CER use provided corpus references, not independently verified transcripts; Vietnamese WER counts whitespace-delimited syllables.
- Word timestamp checks detect invalid or zero-length ranges and ordering issues, not acoustic alignment accuracy.
- GPU memory readings are sampled whole-device observations, not allocated/reserved values or exact peak VRAM.
- Results describe this RTX 3050, driver, runtime, and short sample set; they do not establish universal quality or performance.

## Verification

The existing worker test suite passed: **14 passed**. The benchmark does not require GPU-dependent tests and does not change the installed ASR/CUDA package versions.
