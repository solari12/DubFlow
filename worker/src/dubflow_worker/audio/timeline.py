from __future__ import annotations

import math
import array
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TimelineSegmentPlan:
    segment_id: int
    speaker_id: str | None
    source_start: float
    source_end: float
    translated_text: str
    generated_tts_duration: float
    tts_audio_path: Path
    planned_start: float
    planned_end: float
    allowed_duration: float
    stretch_ratio: float
    overflow_before_fitting: bool
    overflow_after_fitting: bool
    requires_concise_rephrasing: bool
    preserved_pause_before: float
    preserved_pause_after: float
    source_pause_before: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "translated_text": self.translated_text,
            "generated_tts_duration": self.generated_tts_duration,
            "tts_audio_path": str(self.tts_audio_path),
            "planned_start": self.planned_start,
            "planned_end": self.planned_end,
            "allowed_duration": self.allowed_duration,
            "stretch_ratio": self.stretch_ratio,
            "overflow_before_fitting": self.overflow_before_fitting,
            "overflow_after_fitting": self.overflow_after_fitting,
            "requires_concise_rephrasing": self.requires_concise_rephrasing,
            "preserved_pause_before": self.preserved_pause_before,
            "preserved_pause_after": self.preserved_pause_after,
            "source_pause_before": self.source_pause_before,
        }


def plan_dialogue_timeline(
    segments: list[dict[str, Any]],
    *,
    min_stretch_ratio: float = 0.85,
    sample_rate: int = 16000,
    preserve_overflow: bool = False,
) -> list[TimelineSegmentPlan]:
    """Plan ordered dialogue into source windows, preserving gaps and preventing spills."""
    if not 0 < min_stretch_ratio <= 1:
        raise ValueError("min_stretch_ratio must be in (0, 1]")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    plans: list[TimelineSegmentPlan] = []
    previous_planned_end = 0.0
    for index, item in enumerate(segments):
        segment_id = int(item["segment_id"])
        source_start = float(item["source_start"])
        source_end = float(item["source_end"])
        duration = float(item["generated_tts_duration"])
        audio_path = Path(item["tts_audio_path"])
        if (
            not all(math.isfinite(value) for value in (source_start, source_end, duration))
            or source_start < 0
            or source_end <= source_start
            or duration <= 0
        ):
            raise ValueError(f"Invalid timeline input for segment {segment_id}")

        source_pause_before = min(
            max(0.0, float(item.get("source_pause_before", 0.0))),
            max(0.0, source_end - source_start - 1.0 / sample_rate),
        )
        planned_start = round(
            max(source_start + source_pause_before, previous_planned_end)
            * sample_rate
        ) / sample_rate
        window_end = source_end
        if index + 1 < len(segments):
            next_item = segments[index + 1]
            next_start = (
                float(next_item["source_start"])
                + max(0.0, float(next_item.get("source_pause_before", 0.0)))
            )
            window_end = min(window_end, next_start)
        window_end = max(planned_start, round(window_end * sample_rate) / sample_rate)
        allowed_duration = round(max(0.0, window_end - planned_start) * sample_rate) / sample_rate
        if allowed_duration < 1.0 / sample_rate:
            # Earlier dialogue may have consumed this source window. Keep the
            # segment in the ordered plan with a one-frame nominal window so
            # alignment can either preserve/spill its audio or report an
            # explicit QUALITY_FAIL under a fail policy.
            allowed_duration = 1.0 / sample_rate

        stretch_ratio = allowed_duration / duration
        tolerance = 0.5 / sample_rate
        overflow_before = duration > allowed_duration + tolerance
        overflow_after = overflow_before and stretch_ratio < min_stretch_ratio
        if preserve_overflow and overflow_before:
            # Severe overruns still receive the configured bounded speed-up;
            # reserve the resulting duration while keeping the complete audio.
            # atempo output can exceed its nominal ratio by a few milliseconds;
            # leave a small guard gap so the subsequent clips cannot overlap.
            planned_end = planned_start + duration * min_stretch_ratio + 0.02
        else:
            planned_end = planned_start + allowed_duration
        pause_before = max(0.0, planned_start - previous_planned_end)
        next_start = (
            float(segments[index + 1]["source_start"])
            + max(0.0, float(segments[index + 1].get("source_pause_before", 0.0)))
            if index + 1 < len(segments)
            else planned_end
        )
        pause_after = max(0.0, next_start - planned_end)
        plans.append(
            TimelineSegmentPlan(
                segment_id=segment_id,
                speaker_id=item.get("speaker_id"),
                source_start=source_start,
                source_end=source_end,
                translated_text=str(item.get("translated_text", "")),
                generated_tts_duration=duration,
                tts_audio_path=audio_path,
                planned_start=planned_start,
                planned_end=planned_end,
                allowed_duration=allowed_duration,
                stretch_ratio=stretch_ratio,
                overflow_before_fitting=overflow_before,
                overflow_after_fitting=overflow_after,
                requires_concise_rephrasing=overflow_after,
                preserved_pause_before=round(pause_before, 9),
                preserved_pause_after=round(pause_after, 9),
                source_pause_before=round(source_pause_before, 9),
            )
        )
        previous_planned_end = planned_end
    return plans


def detect_boundary_pauses(
    source_wav: Path,
    segments: list[dict[str, Any]],
    *,
    rms_threshold: float = 0.005,
    min_pause_seconds: float = 0.08,
    search_seconds: float = 0.6,
) -> dict[int, float]:
    """Find sustained quiet immediately after transcript segment starts in source PCM."""
    with wave.open(str(source_wav), "rb") as wav_file:
        rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        total_frames = wav_file.getnframes()
        raw = wav_file.readframes(total_frames)
    if rate < 1 or channels < 1 or width != 2:
        raise ValueError("Source pause detection requires 16-bit PCM audio")
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    frame_samples = channels
    chunk_frames = max(1, round(rate * 0.01))
    pauses: dict[int, float] = {}
    for index, item in enumerate(segments):
        if index == 0:
            continue
        segment_id = int(item.get("id", item.get("segment_id", -1)))
        start = float(item.get("start", item.get("source_start", 0.0)))
        end = float(item.get("end", item.get("source_end", start)))
        first = max(0, round(start * rate))
        last = min(total_frames, round((start + search_seconds) * rate))
        quiet_start: int | None = None
        detected_end: int | None = None
        cursor = first
        while cursor < last:
            chunk_end = min(last, cursor + chunk_frames)
            block = samples[cursor * frame_samples:chunk_end * frame_samples]
            rms = math.sqrt(sum(value * value for value in block) / max(1, len(block))) / 32768.0
            if rms < rms_threshold:
                if quiet_start is None:
                    quiet_start = cursor
            elif quiet_start is not None:
                if cursor - quiet_start >= round(min_pause_seconds * rate):
                    detected_end = cursor
                    break
                quiet_start = None
            cursor = chunk_end
        if quiet_start is not None and detected_end is None:
            if last - quiet_start >= round(min_pause_seconds * rate):
                detected_end = last
        if detected_end is not None:
            pause = max(0.0, detected_end / rate - start)
            pauses[segment_id] = round(min(pause, max(0.0, end - start - 1.0 / rate)), 9)
    return pauses
