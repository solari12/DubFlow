import pytest

from dubflow_worker.pipeline.speaker_transcription import merge_speaker_transcript


def transcript(*segments):
    return [
        {"id": index, "start": start, "end": end, "text": text}
        for index, (start, end, text) in enumerate(segments)
    ]


def test_clear_single_speaker_overlap_assigns_speaker() -> None:
    merged = merge_speaker_transcript(
        transcript((1.0, 3.0, "Hello there.")),
        [{"start": 0.5, "end": 3.5, "speaker": "SPEAKER_00"}],
    )
    assert merged[0]["speaker"] == "SPEAKER_00"
    assert merged[0]["speaker_overlap_seconds"] == 2.0
    assert merged[0]["speaker_overlap_ratio"] == 1.0


def test_larger_of_two_speaker_overlaps_wins() -> None:
    merged = merge_speaker_transcript(
        transcript((0.0, 4.0, "A sentence.")),
        [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 4.0, "speaker": "SPEAKER_01"},
        ],
    )
    assert merged[0]["speaker"] == "SPEAKER_01"
    assert merged[0]["speaker_overlap_seconds"] == 3.0
    assert merged[0]["speaker_overlap_ratio"] == 0.75


def test_zero_overlap_is_unassigned() -> None:
    merged = merge_speaker_transcript(
        transcript((0.0, 1.0, "No overlap.")),
        [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}],
    )
    assert merged[0]["speaker"] is None
    assert merged[0]["speaker_overlap_seconds"] == 0.0
    assert merged[0]["speaker_overlap_ratio"] == 0.0


def test_overlap_below_threshold_is_unassigned() -> None:
    merged = merge_speaker_transcript(
        transcript((0.0, 10.0, "Small overlap.")),
        [{"start": 0.0, "end": 1.9, "speaker": "SPEAKER_00"}],
    )
    assert merged[0]["speaker"] is None
    assert merged[0]["speaker_overlap_ratio"] == pytest.approx(0.19)


def test_touching_boundaries_have_zero_overlap() -> None:
    merged = merge_speaker_transcript(
        transcript((1.0, 2.0, "Touching.")),
        [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
    )
    assert merged[0]["speaker"] is None
    assert merged[0]["speaker_overlap_seconds"] == 0.0


def test_multiple_transcript_segments_receive_independent_assignments() -> None:
    merged = merge_speaker_transcript(
        transcript((0.0, 1.0, "First."), (1.0, 2.0, "Second."), (4.0, 5.0, "Third.")),
        [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        ],
    )
    assert [segment["speaker"] for segment in merged] == [
        "SPEAKER_00", "SPEAKER_01", None,
    ]
    assert [segment["text"] for segment in merged] == ["First.", "Second.", "Third."]


def test_overlap_at_threshold_is_assigned() -> None:
    merged = merge_speaker_transcript(
        transcript((0.0, 5.0, "Threshold.")),
        [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
    )
    assert merged[0]["speaker"] == "SPEAKER_00"
