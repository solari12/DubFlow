from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("Diarization timestamps must be finite numbers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Diarization timestamps must satisfy 0 <= start < end")
        if not self.speaker.strip():
            raise ValueError("Speaker label must not be empty")


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    filename: str
    duration: float
    engine: str
    model: str
    device: str
    segments: list[DiarizationSegment]

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError("Audio duration must be a finite, non-negative number")
        if not self.filename or not self.engine or not self.model or self.device not in {"cuda", "cpu"}:
            raise ValueError("Filename, engine, model, and a valid device are required")
        for segment in self.segments:
            if segment.end > self.duration + 0.05:
                raise ValueError("Diarization segment end exceeds audio duration")

    @property
    def speakers(self) -> list[str]:
        return sorted({segment.speaker for segment in self.segments})

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "audio": {"filename": self.filename, "duration": self.duration},
            "diarization": {"engine": self.engine, "model": self.model, "device": self.device},
            "speakers": self.speakers,
            "segments": [
                {"start": item.start, "end": item.end, "speaker": item.speaker}
                for item in self.segments
            ],
        }

    @classmethod
    def from_tracks(
        cls,
        tracks: Iterable[tuple[float, float, str]],
        *,
        filename: str,
        duration: float,
        engine: str,
        model: str,
        device: str,
    ) -> DiarizationResult:
        segments = [DiarizationSegment(float(start), float(end), str(speaker)) for start, end, speaker in tracks]
        segments.sort(key=lambda item: (item.start, item.end, item.speaker))
        return cls(filename, duration, engine, model, device, segments)


def assign_speakers(
    transcript_segments: Iterable[dict[str, Any]], diarization: DiarizationResult
) -> list[dict[str, Any]]:
    """Copy ASR segments and assign the speaker with the largest time overlap."""
    merged: list[dict[str, Any]] = []
    for source in transcript_segments:
        item = dict(source)
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("ASR timestamps must satisfy 0 <= start < end")
        overlaps = [
            (max(0.0, min(end, turn.end) - max(start, turn.start)), turn.speaker)
            for turn in diarization.segments
        ]
        best = max(overlaps, default=(0.0, ""))
        item["speaker"] = best[1] if best[0] > 0 else None
        merged.append(item)
    return merged
