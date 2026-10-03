from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    word: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("Word timestamps must be finite numbers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("Word timestamps must satisfy 0 <= start <= end")
        if not self.word.strip():
            raise ValueError("Word text must not be empty")


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[WordTimestamp] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("Segment id must be non-negative")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("Segment timestamps must be finite numbers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("Segment timestamps must satisfy 0 <= start <= end")
        if not self.text.strip():
            raise ValueError("Segment text must not be empty")


@dataclass(frozen=True, slots=True)
class Transcript:
    language: str | None
    duration: float
    segments: list[TranscriptSegment] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError("Transcript duration must be a finite, non-negative number")
        ids = [segment.id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("Segment ids must be unique")
        for segment in self.segments:
            if segment.end > self.duration + 0.05:
                raise ValueError("Segment end must not exceed transcript duration")

    def segments_as_dicts(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for segment in self.segments:
            item: dict[str, Any] = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            if segment.words:
                item["words"] = [
                    {"word": word.word, "start": word.start, "end": word.end}
                    for word in segment.words
                ]
            result.append(item)
        return result

    def to_dict(
        self,
        *,
        filename: str,
        engine: str = "faster-whisper",
        model: str,
    ) -> dict[str, Any]:
        return {
            "version": "1.0",
            "source": {"filename": filename, "duration": self.duration},
            "asr": {"engine": engine, "model": model, "language": self.language},
            "segments": self.segments_as_dicts(),
        }
