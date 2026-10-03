from __future__ import annotations

import array
import math
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from dubflow_worker.audio.extractor import AudioExtractor
from dubflow_worker.models.alignment import AudioAlignmentSettings


class InvalidAudioError(ValueError):
    """A TTS segment is not a readable, non-empty PCM WAV file."""


@dataclass(frozen=True, slots=True)
class WaveData:
    samples: array.array
    sample_rate: int
    channels: int

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def read_pcm_wav(path: Path) -> WaveData:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
            if channels < 1 or sample_rate < 1 or frame_count < 1:
                raise InvalidAudioError("WAV has zero duration or invalid audio metadata")
            if compression != "NONE":
                raise InvalidAudioError(f"Unsupported WAV compression: {compression}")
            raw = wav_file.readframes(frame_count)
    except (wave.Error, OSError) as exc:
        raise InvalidAudioError(f"Could not read WAV: {exc}") from exc

    if sample_width != 2:
        raise InvalidAudioError("WAV is not 16-bit PCM")
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) != frame_count * channels:
        raise InvalidAudioError("WAV sample data is truncated")
    return WaveData(samples=samples, sample_rate=sample_rate, channels=channels)


def write_pcm_wav(path: Path, samples: array.array, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = array.array("h", samples)
    if sys.byteorder != "little":
        output.byteswap()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(output.tobytes())


def convert_wav(
    source: Path,
    destination: Path,
    *,
    settings: AudioAlignmentSettings,
    atempo: float | None = None,
) -> None:
    """Use the existing FFmpeg resolution convention for audio conversion/stretch."""
    ffmpeg = AudioExtractor(settings.ffmpeg_path)._resolve_executable(settings.ffmpeg_path)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source)]
    if atempo is not None:
        if not math.isfinite(atempo) or not 0.5 <= atempo <= 2.0:
            raise ValueError("FFmpeg atempo must be between 0.5 and 2.0")
        command.extend(["-af", f"atempo={atempo:.10f}"])
    command.extend(
        [
            "-vn",
            "-ac",
            str(settings.channels),
            "-ar",
            str(settings.sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(destination),
        ]
    )
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"Could not start FFmpeg for TTS audio: {exc}") from exc
    if completed.returncode != 0 or not destination.is_file():
        detail = completed.stderr.strip()[-1200:]
        raise RuntimeError(
            f"FFmpeg audio conversion failed (exit {completed.returncode}): "
            f"{detail or 'no diagnostic output'}"
        )
    # Reject bad output immediately rather than letting one segment break the final mix.
    read_pcm_wav(destination)


def normalize_wav(
    source: Path,
    destination: Path,
    *,
    source_info: WaveData,
    settings: AudioAlignmentSettings,
    atempo: float | None = None,
) -> Path:
    already_normalized = (
        source_info.sample_rate == settings.sample_rate
        and source_info.channels == settings.channels
    )
    if already_normalized and atempo is None:
        return source
    convert_wav(source, destination, settings=settings, atempo=atempo)
    return destination


def fit_samples_to_frames(samples: array.array, frames: int, channels: int) -> array.array:
    target_length = frames * channels
    fitted = array.array("h", samples[:target_length])
    if len(fitted) < target_length:
        fitted.extend([0] * (target_length - len(fitted)))
    return fitted
