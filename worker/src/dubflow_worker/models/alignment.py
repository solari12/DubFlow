from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


OverflowPolicy = Literal["preserve", "trim", "fail"]


@dataclass(frozen=True, slots=True)
class AudioAlignmentSettings:
    min_stretch_factor: float = 0.85
    max_stretch_factor: float = 1.10
    pad_short_audio: bool = True
    overflow_policy: OverflowPolicy = "preserve"
    sample_rate: int = 16000
    channels: int = 1
    output_format: str = "wav"
    ffmpeg_path: str = "ffmpeg"

    def __post_init__(self) -> None:
        if not 0.75 <= self.min_stretch_factor <= 1.0:
            raise ValueError("Minimum stretch factor must be between 0.75 and 1.0")
        if not 1.0 <= self.max_stretch_factor <= 1.25:
            raise ValueError("Maximum stretch factor must be between 1.0 and 1.25")
        if self.min_stretch_factor > self.max_stretch_factor:
            raise ValueError("Minimum stretch factor must not exceed maximum")
        if self.sample_rate <= 0:
            raise ValueError("Output sample rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("Output channel count must be 1 or 2")
        if self.output_format.lower() != "wav":
            raise ValueError("Only WAV output is currently supported")
        if self.overflow_policy not in {"preserve", "trim", "fail"}:
            raise ValueError("Overflow policy must be preserve, trim, or fail")


@dataclass(frozen=True, slots=True)
class AlignedSegment:
    segment_id: int
    speaker: str | None
    target_start: float
    target_end: float
    target_duration: float
    original_tts_duration: float | None
    final_audio_duration: float | None
    duration_error: float | None
    stretch_factor: float | None
    alignment_action: str
    output_audio_path: Path | None
    status: Literal["success", "overflow", "failed"]
    error: str | None = None
    source_start: float | None = None
    source_end: float | None = None
    planned_start: float | None = None
    planned_end: float | None = None
    allowed_duration: float | None = None
    overflow_before_fitting: bool = False
    overflow_after_fitting: bool = False
    requires_concise_rephrasing: bool = False
    shortening_attempted: bool = False
    shortened: bool = False
    forcibly_truncated: bool = False
    truncated_duration: float = 0.0
    source_text: str = ""
    translated_text: str = ""
    preserved_pause_before: float = 0.0
    preserved_pause_after: float = 0.0

    def to_dict(self) -> dict:
        quality_fail = (
            self.status == "failed"
            or (self.status == "overflow" and self.output_audio_path is None)
            or self.forcibly_truncated
        )
        return {
            "segment_id": self.segment_id,
            "speaker": self.speaker,
            "target_start": self.target_start,
            "target_end": self.target_end,
            "target_duration": self.target_duration,
            "tts_duration": self.original_tts_duration,
            "final_duration": self.final_audio_duration,
            "original_tts_duration": self.original_tts_duration,
            "final_audio_duration": self.final_audio_duration,
            "duration_error": self.duration_error,
            "stretch_factor": self.stretch_factor,
            "alignment_action": self.alignment_action,
            "output_audio_path": str(self.output_audio_path) if self.output_audio_path else None,
            "status": self.status,
            "error": self.error,
            "source_start": self.source_start if self.source_start is not None else self.target_start,
            "source_end": self.source_end if self.source_end is not None else self.target_end,
            "planned_start": self.planned_start if self.planned_start is not None else self.target_start,
            "planned_end": self.planned_end if self.planned_end is not None else self.target_end,
            "allowed_duration": self.allowed_duration if self.allowed_duration is not None else self.target_duration,
            "overflow_before_fitting": self.overflow_before_fitting,
            "overflow_after_fitting": self.overflow_after_fitting,
            "requires_concise_rephrasing": self.requires_concise_rephrasing,
            "shortening_attempted": self.shortening_attempted,
            "shortened": self.shortened,
            "forcibly_truncated": self.forcibly_truncated,
            "truncated_duration": self.truncated_duration,
            "was_stretched": self.stretch_factor is not None and self.stretch_factor < 1.0,
            "was_truncated": self.forcibly_truncated,
            "truncated_seconds": self.truncated_duration,
            "quality_status": "QUALITY_FAIL" if quality_fail else "OK",
            "quality_failure_reason": "INSUFFICIENT_SPEECH_WINDOW" if quality_fail else None,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "preserved_pause_before": self.preserved_pause_before,
            "preserved_pause_after": self.preserved_pause_after,
        }


@dataclass(frozen=True, slots=True)
class AlignmentRun:
    timeline_path: Path
    segments: list[AlignedSegment]
    sample_rate: int
    channels: int
    target_timeline_duration: float
    total_duration: float
    runtime_seconds: float
    rtf: float | None
    peak_amplitude: float
    clipping_count: int
    overlap_policy: str = "non_overlapping_dialogue; global_peak_normalize_to_0.99"
    planned_overlap_count: int = 0
    actual_overlap_count: int = 0

    @property
    def successful_count(self) -> int:
        return sum(segment.status == "success" for segment in self.segments)

    @property
    def failed_count(self) -> int:
        return sum(segment.status == "failed" for segment in self.segments)

    @property
    def overflow_count(self) -> int:
        return sum(segment.status == "overflow" for segment in self.segments)

    @property
    def quality_fail_count(self) -> int:
        return sum(
            segment.status == "failed"
            or (segment.status == "overflow" and segment.output_audio_path is None)
            or segment.forcibly_truncated
            for segment in self.segments
        )

    def to_dict(self, *, input_transcript: dict | None = None) -> dict:
        return {
            "version": "1.0",
            "success": self.failed_count == 0,
            "input_transcript": input_transcript or {},
            "output_audio_path": str(self.timeline_path),
            "output_sample_rate": self.sample_rate,
            "output_channels": self.channels,
            "output_format": "wav",
            "target_timeline_duration": self.target_timeline_duration,
            "total_duration": self.total_duration,
            "segment_count": len(self.segments),
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "overflow_count": self.overflow_count,
            "quality_fail_count": self.quality_fail_count,
            "total_processing_time_seconds": self.runtime_seconds,
            "rtf": self.rtf,
            "peak_amplitude": self.peak_amplitude,
            "clipping_count": self.clipping_count,
            "overlap_policy": self.overlap_policy,
            "planned_overlap_count": self.planned_overlap_count,
            "actual_overlap_count": self.actual_overlap_count,
            "preserved_pause_count": sum(segment.preserved_pause_after > 0 for segment in self.segments),
            "overflow_before_fitting_count": sum(segment.overflow_before_fitting for segment in self.segments),
            "overflow_after_fitting_count": sum(segment.overflow_after_fitting for segment in self.segments),
            "requires_concise_rephrasing_count": sum(
                segment.requires_concise_rephrasing for segment in self.segments
            ),
            "forcibly_truncated_count": sum(segment.forcibly_truncated for segment in self.segments),
            "shortened_segment_count": sum(segment.shortened for segment in self.segments),
            "tts_duration_before_fitting": round(sum(segment.original_tts_duration or 0 for segment in self.segments), 6),
            "segments": [segment.to_dict() for segment in self.segments],
        }


def valid_timestamps(start: float, end: float) -> bool:
    return math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start
