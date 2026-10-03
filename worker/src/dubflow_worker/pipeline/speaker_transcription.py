from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def merge_speaker_transcript(
    transcript_segments: Sequence[Mapping[str, Any]],
    diarization_segments: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.20,
) -> list[dict[str, Any]]:
    """Attach the label of the single diarization turn with greatest overlap."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Speaker overlap threshold must be between 0 and 1")

    merged: list[dict[str, Any]] = []
    for transcript in transcript_segments:
        start = float(transcript["start"])
        end = float(transcript["end"])
        duration = end - start
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or duration <= 0:
            raise ValueError("ASR timestamps must satisfy 0 <= start < end")

        best_overlap = 0.0
        best_speaker: str | None = None
        for turn in diarization_segments:
            turn_start = float(turn["start"])
            turn_end = float(turn["end"])
            speaker = str(turn["speaker"])
            overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        ratio = best_overlap / duration
        if best_overlap == 0.0 or ratio < threshold:
            best_speaker = None
        merged.append(
            {
                "id": int(transcript["id"]),
                "start": start,
                "end": end,
                "speaker": best_speaker,
                "speaker_overlap_seconds": best_overlap,
                "speaker_overlap_ratio": ratio,
                "text": str(transcript["text"]),
            }
        )
    return merged
