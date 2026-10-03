from __future__ import annotations

from pathlib import Path


class PyannoteDiarizer:
    name = "pyannote.audio"

    def __init__(self, model: str, device: str, token: str | None = None) -> None:
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Diarization dependencies are missing. Install the worker's diarization extra "
                "and a CUDA-enabled PyTorch build for GPU inference."
            ) from exc

        self.model = model
        self.device = device
        self.pipeline = Pipeline.from_pretrained(model, token=token)
        self.pipeline.to(torch.device(device))

    def diarize(self, audio_path: Path, min_speakers: int | None = None,
                max_speakers: int | None = None):
        options = {}
        if min_speakers is not None:
            options["min_speakers"] = min_speakers
        if max_speakers is not None:
            options["max_speakers"] = max_speakers
        return self.pipeline(str(audio_path), **options)

    def close(self) -> None:
        del self.pipeline
