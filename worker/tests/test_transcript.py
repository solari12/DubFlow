import json

import pytest

from dubflow_worker.models.transcript import Transcript, TranscriptSegment, WordTimestamp


def test_valid_transcript_creation_and_json_serialization() -> None:
    transcript = Transcript(
        language="en",
        duration=4.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=0.52,
                end=3.81,
                text="Welcome to our channel.",
                words=[WordTimestamp(word="Welcome", start=0.52, end=1.12)],
            )
        ],
    )
    payload = transcript.to_dict(filename="sample.mp4", model="base")
    round_trip = json.loads(json.dumps(payload))

    assert round_trip["version"] == "1.0"
    assert round_trip["source"] == {"filename": "sample.mp4", "duration": 4.0}
    assert round_trip["asr"]["language"] == "en"
    assert round_trip["segments"][0]["text"] == "Welcome to our channel."
    assert round_trip["segments"][0]["words"][0]["word"] == "Welcome"


@pytest.mark.parametrize(
    ("start", "end"),
    [(-0.1, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))],
)
def test_invalid_segment_timestamps_are_rejected(start: float, end: float) -> None:
    with pytest.raises(ValueError):
        TranscriptSegment(id=0, start=start, end=end, text="hello")


def test_segment_end_cannot_exceed_transcript_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        Transcript(language="en", duration=1, segments=[
            TranscriptSegment(id=0, start=0, end=2, text="late")
        ])


def test_duplicate_segment_ids_are_rejected() -> None:
    segment = TranscriptSegment(id=0, start=0, end=0.5, text="hello")
    with pytest.raises(ValueError, match="unique"):
        Transcript(language="en", duration=1, segments=[segment, segment])
