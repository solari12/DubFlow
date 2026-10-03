from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dubflow_worker.models.transcript import Transcript


class ASRError(RuntimeError):
    """An ASR engine could not load or transcribe audio."""


class ASREngine(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        """Transcribe audio into timed segments."""

    def close(self) -> None:
        """Release resources if the implementation owns any."""
