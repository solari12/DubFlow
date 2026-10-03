from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TranslatedSegment:
    id: int
    start: float
    end: float
    speaker: str | None
    source_text: str
    target_text: str | None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("Segment id must be non-negative")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("Translation timestamps must be finite numbers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("Translation timestamps must satisfy 0 <= start <= end")
        if self.target_text is None and not self.error:
            raise ValueError("A failed translation must include an error message")
        if self.target_text is not None and self.error is not None:
            raise ValueError("A successful translation cannot include an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class TranslatedTranscript:
    engine: str
    source_language: str
    target_language: str
    segments: list[TranslatedSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "translation": {
                "engine": self.engine,
                "source_language": self.source_language,
                "target_language": self.target_language,
            },
            "segments": [segment.to_dict() for segment in self.segments],
        }
