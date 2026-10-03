from __future__ import annotations

import math
import subprocess
import wave
from array import array
from pathlib import Path

import pytest

from dubflow_worker.audio.extractor import AudioExtractor
from dubflow_worker.video.ffmpeg import FFmpegVideoRenderer, RenderSettings


@pytest.fixture(scope="module")
def ffmpeg() -> str:
    return AudioExtractor()._resolve_executable("ffmpeg")


def _make_wav(path: Path, duration: float, frequency: float = 880.0) -> None:
    rate = 16000
    count = round(duration * rate)
    samples = array(
        "h",
        (int(10000 * math.sin(2 * math.pi * frequency * index / rate)) for index in range(count)),
    )
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())


def _make_video(ffmpeg: str, path: Path, duration: float = 2.0, with_audio: bool = True) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=160x120:rate=10:duration={duration}",
    ]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=44100:duration={duration}"]
    command += ["-map", "0:v:0"]
    if with_audio:
        command += ["-map", "1:a:0", "-c:a", "aac"]
    command += ["-c:v", "mpeg4", "-q:v", "5", "-t", str(duration), str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)


@pytest.fixture
def media(tmp_path: Path, ffmpeg: str) -> tuple[Path, Path]:
    video = tmp_path / "source.mp4"
    audio = tmp_path / "dubbed.wav"
    _make_video(ffmpeg, video)
    _make_wav(audio, 2.5)
    return video, audio


def _renderer() -> FFmpegVideoRenderer:
    return FFmpegVideoRenderer(RenderSettings())


def test_missing_video_is_reported(tmp_path: Path) -> None:
    audio = tmp_path / "dubbed.wav"
    _make_wav(audio, 1)
    result = _renderer().render(tmp_path / "missing.mp4", audio, tmp_path / "out.mp4")
    assert result.status == "failed"
    assert "video does not exist" in (result.error or "")


def test_missing_dubbed_audio_is_reported(tmp_path: Path, ffmpeg: str) -> None:
    video = tmp_path / "source.mp4"
    _make_video(ffmpeg, video)
    result = _renderer().render(video, tmp_path / "missing.wav", tmp_path / "out.mp4")
    assert result.status == "failed"
    assert "audio does not exist" in (result.error or "")


def test_invalid_video_is_reported(tmp_path: Path) -> None:
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"not a video")
    audio = tmp_path / "dubbed.wav"
    _make_wav(audio, 1)
    result = _renderer().render(video, audio, tmp_path / "out.mp4")
    assert result.status == "failed"
    assert "inspect media" in (result.error or "")


def test_invalid_wav_is_reported(tmp_path: Path, ffmpeg: str) -> None:
    video = tmp_path / "source.mp4"
    _make_video(ffmpeg, video)
    audio = tmp_path / "broken.wav"
    audio.write_bytes(b"not a wav")
    result = _renderer().render(video, audio, tmp_path / "out.mp4")
    assert result.status == "failed"
    assert "readable WAV" in (result.error or "")


def test_video_without_video_stream_is_rejected(tmp_path: Path, ffmpeg: str) -> None:
    audio_only = tmp_path / "audio-only.mp4"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "sine=frequency=220:duration=1", "-c:a", "aac", str(audio_only)],
        check=True,
        capture_output=True,
        text=True,
    )
    dubbed = tmp_path / "dubbed.wav"
    _make_wav(dubbed, 1)
    result = _renderer().render(audio_only, dubbed, tmp_path / "out.mp4")
    assert result.status == "failed"
    assert "no video stream" in (result.error or "")


def test_render_copies_video_replaces_audio_and_validates_streams(media: tuple[Path, Path], tmp_path: Path) -> None:
    video, audio = media
    output = tmp_path / "rendered.mp4"
    result = _renderer().render(video, audio, output)
    assert result.status == "success", result.error
    assert result.output_has_video and result.output_has_audio
    assert result.video_stream_copied and result.audio_reencoded
    assert result.video_codec == "mpeg4"
    assert result.audio_codec == "aac"
    assert output.is_file() and output.stat().st_size > 0
    assert result.output_duration == pytest.approx(result.source_video_duration, abs=0.12)
    assert result.duration_policy == "trim_audio_to_video_duration"

    decoded = tmp_path / "decoded.wav"
    subprocess.run(
        [AudioExtractor()._resolve_executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(output), "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(decoded)],
        check=True,
        capture_output=True,
        text=True,
    )
    with wave.open(str(decoded), "rb") as wav:
        samples = array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    crossing_count = sum((left <= 0 < right) for left, right in zip(samples, samples[1:]))
    estimated_frequency = crossing_count / (len(samples) / 16000)
    assert estimated_frequency == pytest.approx(880, abs=30)  # source track was 220 Hz


def test_short_dubbed_audio_is_padded_to_video_duration(tmp_path: Path, ffmpeg: str) -> None:
    video = tmp_path / "source.mp4"
    audio = tmp_path / "short.wav"
    _make_video(ffmpeg, video, duration=2)
    _make_wav(audio, 0.5)
    result = _renderer().render(video, audio, tmp_path / "out.mp4")
    assert result.status == "success", result.error
    assert result.duration_policy == "pad_audio_with_trailing_silence_to_video_duration"
    assert result.output_duration == pytest.approx(result.source_video_duration, abs=0.12)
    decoded = tmp_path / "decoded-short.wav"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp_path / "out.mp4"),
         "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(decoded)],
        check=True,
        capture_output=True,
        text=True,
    )
    with wave.open(str(decoded), "rb") as wav:
        samples = array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    trailing_silence = samples[int(1.5 * 16000) :]
    assert trailing_silence
    assert max(abs(sample) for sample in trailing_silence) < 100


def test_equal_duration_and_result_fields(media: tuple[Path, Path], tmp_path: Path) -> None:
    video, _ = media
    audio = tmp_path / "equal.wav"
    _make_wav(audio, 2)
    result = _renderer().render(video, audio, tmp_path / "out.mp4")
    payload = result.to_dict()
    assert result.status == "success", result.error
    assert result.duration_policy == "audio_matches_video_duration"
    assert result.runtime_seconds >= 0
    assert result.output_size_bytes and result.output_size_bytes > 0
    assert payload["input_video"] == str(video)
    assert payload["input_audio"] == str(audio)
    assert payload["output_video"] == str(tmp_path / "out.mp4")
    assert payload["duration_mismatch"] == pytest.approx(0, abs=0.02)
