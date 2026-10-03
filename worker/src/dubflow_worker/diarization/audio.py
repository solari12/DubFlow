from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from dubflow_worker.audio.extractor import AudioExtractor


DIARIZATION_SAMPLE_RATE = 16_000


def load_audio_waveform(
    audio_path: Path,
    extractor: AudioExtractor | None = None,
) -> dict[str, Any]:
    """Decode media through FFmpeg and return mono float32 audio for pyannote."""
    import torch

    audio_extractor = extractor or AudioExtractor()
    with audio_extractor.extract(Path(audio_path)) as extracted:
        try:
            with wave.open(str(extracted.path), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, OSError) as exc:
            raise ValueError(f"Could not read extracted WAV audio: {exc}") from exc

    if channels != 1:
        raise ValueError(f"Expected mono audio after FFmpeg conversion; got {channels} channels.")
    if sample_rate != DIARIZATION_SAMPLE_RATE:
        raise ValueError(
            f"Expected {DIARIZATION_SAMPLE_RATE} Hz audio after FFmpeg conversion; got {sample_rate} Hz."
        )
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM audio after FFmpeg conversion; got {sample_width * 8}-bit.")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        raise ValueError("Audio contains no samples.")
    waveform = torch.from_numpy(samples.copy()).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}
