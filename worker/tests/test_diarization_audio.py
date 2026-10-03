from __future__ import annotations

import wave
from pathlib import Path

import pytest

from dubflow_worker.diarization.audio import load_audio_waveform


def test_load_audio_waveform_decodes_mono_16khz_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from dubflow_worker.diarization.audio import load_audio_waveform

    source = tmp_path / "stereo-8khz.wav"
    samples = bytearray()
    for index in range(8_000):
        value = int(12_000 * ((index % 80) / 80 - 0.5))
        samples.extend(value.to_bytes(2, "little", signed=True))
        samples.extend((-value).to_bytes(2, "little", signed=True))
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(samples)

    audio = load_audio_waveform(source)
    waveform = audio["waveform"]

    assert isinstance(waveform, torch.Tensor)
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0
    assert audio["sample_rate"] == 16_000
    assert waveform.dtype == torch.float32
