# Phase 1F: Video Rendering

Phase 1F muxes the original video's first video stream with the Phase 1E dubbed timeline WAV. The original audio is omitted. `FFmpegVideoRenderer` is a small renderer abstraction backed by the project's existing `AudioExtractor` FFmpeg resolver, including its bundled `imageio-ffmpeg` fallback. No new media package or CUDA path was added.

## Render policy

- The input video must exist, be readable by FFmpeg, have a video stream, and report a positive duration.
- The dubbed input must be a readable nonempty PCM WAV. By default it must match the Phase 1E format: 16 kHz mono.
- The default `--video-mode copy` maps only the first video stream and copies its encoded packets into MP4. If the source codec/container combination cannot be muxed safely, rendering fails clearly; use `--video-mode h264` to explicitly re-encode with libx264.
- The first source audio stream is not mapped. The first dubbed WAV stream is encoded as AAC at 192 kbps by default.
- The output duration follows the source video. Longer dubbed audio is trimmed for the MP4 and its full Phase 1E WAV remains untouched. Shorter dubbed audio is padded with silence through the video end. Both policies and the signed duration mismatch are reported.
- FFmpeg opens and inspects the source and rendered media; the renderer requires both output streams, then decodes both streams to a null sink before accepting the MP4. The temporary output is renamed into place only after validation.

The render result records source video/audio duration, dubbed audio duration, final duration, codecs, whether the video stream was copied, audio re-encoding, mismatch, policy, runtime, status, output size, stream presence, and errors. The benchmark adds input/output sizes and RTF.

## Reproduction

Render a local source video:

```powershell
worker/.venv/Scripts/python.exe worker/scripts/render_video.py `
  input/sample.mp4 `
  output/phase1e/dubbed_timeline.wav `
  --output output/phase1f/dubbed_video.mp4 `
  --audio-codec aac --audio-bitrate 192k --video-mode copy
```

Run and record the benchmark:

```powershell
worker/.venv/Scripts/python.exe worker/scripts/benchmark_render.py `
  input/sample.mp4 `
  output/phase1e/dubbed_timeline.wav `
  --output-dir output/phase1f
```

The CLI writes `render-result.json` beside the requested MP4. The benchmark writes both `render-result.json` and `render-benchmark.json` under its output directory, as well as `dubbed_video.mp4`.

## Render validation

No original video was present in the repository, so the render validation used a generated 33.18-second 320x180, 24 fps MPEG-4 test-pattern MP4 with a 220 Hz source audio track, together with the real `output/phase1e/dubbed_timeline.wav` produced from the Phase 1D clips. This is a Phase 1F render validation using existing Phase 1E artifacts; Phase 1A–1E AI inference was not rerun as one fresh end-to-end job.

| Metric | Result |
| --- | ---: |
| Source video duration | 33.180 s |
| Original audio duration | 33.181 s |
| Dubbed WAV duration | 34.8645 s |
| Output duration | 33.180 s |
| Render runtime / RTF | 0.514777 s / 0.015515 |
| Input / output MP4 size | 1,319,063 / 996,592 bytes |
| Video / audio codec | MPEG-4 Part 2 copied / AAC encoded |
| Video stream copied | Yes |
| Duration mismatch / policy | +1.6845 s / trim dubbed audio to video duration |
| Output streams decoded | Video and audio: passed |

The focused test suite also verifies source audio replacement by decoding the output and measuring the dubbed test tone; it confirms that the 880 Hz dubbed track replaces the 220 Hz source track. The render validates muxing, stream readability, codec selection, and duration policy, not speech naturalness.

## Regression and limitations

After Phase 1F, the complete worker suite passed in the ASR environment (**70 passed, 1 skipped**) and in the diarization/translation environment (**71 passed**). The Phase 1D TTS smoke and Phase 1E alignment benchmark were rerun successfully. Piper's Vietnamese x-low voice still reports skipped phonemes; pronunciation and naturalness were not evaluated. The current TTS voice also does not vary by speaker label.

The generated test-pattern video is not a real recorded source clip. Before relying on Phase 1F for a particular media collection, test representative source codecs and containers; unsupported copy combinations require the explicit H.264 mode. No video from the full Phase 1A–1E processing chain was provided for visual review.
