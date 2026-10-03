from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TTSResult:
    audio_path: Path
    duration: float
    sample_rate: int
    engine: str
    language: str
    speaker: str | None
    runtime: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Generated audio duration must be finite and greater than zero")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be greater than zero")
        if not math.isfinite(self.runtime) or self.runtime < 0:
            raise ValueError("Synthesis runtime must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": str(self.audio_path),
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "engine": self.engine,
            "language": self.language,
            "speaker": self.speaker,
            "runtime": self.runtime,
        }


@dataclass(frozen=True, slots=True)
class SynthesizedSegment:
    id: int
    start: float
    end: float
    speaker: str | None
    source_text: str
    target_text: str | None
    result: TTSResult | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "audio": self.result.to_dict() if self.result else None,
            "error": self.error,
        }
