from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dubflow_worker.models.tts import TTSResult


class TTSEngine(ABC):
    """Provider-neutral interface for generating one speech artifact."""

    name: str
    device: str

    @abstractmethod
    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Path,
        speaker: str | None = None,
    ) -> TTSResult:
        """Write speech to output_path and return metadata read from that audio file."""

