import math

import pytest

from dubflow_worker.diarization.base import (
    DiarizationResult,
    DiarizationSegment,
    assign_speakers,
)


def result() -> DiarizationResult:
    return DiarizationResult.from_tracks(
        [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")],
        filename="sample.wav", duration=4.0, engine="pyannote.audio",
        model="test", device="cpu",
    )


def test_parse_tracks_sorts_and_serializes_speakers() -> None:
    parsed = DiarizationResult.from_tracks(
        [(2.0, 4.0, "SPEAKER_01"), (0.0, 2.0, "SPEAKER_00")],
        filename="sample.wav", duration=4.0, engine="test", model="test", device="cpu",
    )
    assert parsed.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert [segment.start for segment in parsed.segments] == [0.0, 2.0]
    assert parsed.to_dict()["segments"][1]["speaker"] == "SPEAKER_01"


@pytest.mark.parametrize(
    "start,end,speaker",
    [(-1, 1, "S"), (1, 1, "S"), (0, math.inf, "S"), (0, 1, " ")],
)
def test_segment_rejects_invalid_timestamps_or_speaker(start, end, speaker) -> None:
    with pytest.raises(ValueError):
        DiarizationSegment(start, end, speaker)


def test_result_rejects_segment_past_audio_duration() -> None:
    with pytest.raises(ValueError, match="exceeds audio duration"):
        DiarizationResult.from_tracks(
            [(0.0, 4.2, "SPEAKER_00")], filename="sample.wav", duration=4.0,
            engine="test", model="test", device="cpu",
        )


def test_assign_speaker_by_maximum_temporal_overlap_and_keep_unmatched() -> None:
    source = [
        {"start": 0.2, "end": 2.6, "text": "Hello"},
        {"start": 3.8, "end": 4.0, "text": "there"},
        {"start": 5.0, "end": 6.0, "text": "unknown"},
    ]
    merged = assign_speakers(source, result())
    assert merged == [
        {"start": 0.2, "end": 2.6, "text": "Hello", "speaker": "SPEAKER_00"},
        {"start": 3.8, "end": 4.0, "text": "there", "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 6.0, "text": "unknown", "speaker": None},
    ]
    assert source[0] == {"start": 0.2, "end": 2.6, "text": "Hello"}


def test_assign_rejects_invalid_asr_timestamps() -> None:
    with pytest.raises(ValueError, match="ASR timestamps"):
        assign_speakers([{"start": 2.0, "end": 1.0}], result())
