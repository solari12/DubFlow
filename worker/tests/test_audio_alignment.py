from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from dubflow_worker.models.alignment import AudioAlignmentSettings
from dubflow_worker.pipeline.align_audio import align_translated_transcript


def _write_wav(
    path: Path,
    duration: float,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = round(duration * sample_rate)
    value = max(-32768, min(32767, round(amplitude * 32767)))
    payload = struct.pack("<h", value) * frames * channels
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(payload)


def _segment(segment_id: int, path: Path, start: float, end: float, speaker: str | None) -> dict:
    return {
        "id": segment_id,
        "start": start,
        "end": end,
        "speaker": speaker,
        "source_text": "source",
        "target_text": "target",
        "audio": {"audio_path": str(path)},
    }


def _align(tmp_path: Path, segments: list[dict], settings: AudioAlignmentSettings | None = None):
    return align_translated_transcript(
        {"version": "1.0", "segments": segments},
        tts_output_dir=tmp_path / "tts",
        output_dir=tmp_path / "aligned",
        settings=settings,
    )


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def test_exact_duration_match(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.5)

    run = _align(tmp_path, [_segment(0, wav, 0.0, 0.5, "SPEAKER_00")])

    assert run.segments[0].alignment_action == "exact"
    assert run.segments[0].status == "success"
    assert run.segments[0].duration_error == 0
    assert _duration(run.segments[0].output_audio_path) == 0.5


def test_short_tts_is_padded_with_silence(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.25)

    run = _align(tmp_path, [_segment(0, wav, 0.0, 0.5, None)])
    segment = run.segments[0]

    assert segment.alignment_action == "pad_silence"
    assert segment.status == "success"
    assert _duration(segment.output_audio_path) == 0.5
    with wave.open(str(segment.output_audio_path), "rb") as audio:
        audio.readframes(4000)
        padded = audio.readframes(4000)
    assert padded and set(padded) == {0}


def test_slight_overrun_uses_bounded_pitch_preserving_stretch(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.52)

    run = _align(tmp_path, [_segment(0, wav, 0.0, 0.5, "SPEAKER_00")])
    segment = run.segments[0]

    assert segment.alignment_action == "time_stretch"
    assert segment.status == "success"
    assert 0.90 <= segment.stretch_factor <= 1.10
    assert _duration(segment.output_audio_path) == 0.5


def test_significant_overrun_is_preserved_and_marked_overflow(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.8)

    run = _align(tmp_path, [_segment(0, wav, 0.0, 0.5, "SPEAKER_00")])
    segment = run.segments[0]

    assert segment.status == "overflow"
    assert segment.alignment_action == "overflow_preserve"
    assert _duration(segment.output_audio_path) == 0.8
    assert run.total_duration == 0.8


def test_overflow_policy_can_explicitly_trim(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.8)

    run = _align(
        tmp_path,
        [_segment(0, wav, 0.0, 0.5, "SPEAKER_00")],
        AudioAlignmentSettings(overflow_policy="trim"),
    )

    assert run.segments[0].status == "overflow"
    assert run.segments[0].alignment_action == "overflow_trim"
    assert "truncated" in run.segments[0].error.lower()
    assert _duration(run.segments[0].output_audio_path) == 0.5


def test_missing_tts_file_fails_segment_and_continues(tmp_path: Path) -> None:
    good = tmp_path / "tts" / "segment-0001.wav"
    _write_wav(good, 0.25)

    run = _align(
        tmp_path,
        [
            _segment(0, tmp_path / "missing.wav", 0.0, 0.5, "SPEAKER_00"),
            _segment(1, good, 1.0, 1.5, "SPEAKER_01"),
        ],
    )

    assert run.segments[0].status == "failed"
    assert "missing" in run.segments[0].error.lower()
    assert run.segments[1].status == "success"
    assert run.total_duration == 1.5


def test_corrupt_wav_fails_only_that_segment(tmp_path: Path) -> None:
    corrupt = tmp_path / "tts" / "segment-0000.wav"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not a wave")

    run = _align(tmp_path, [_segment(0, corrupt, 0.0, 0.5, None)])

    assert run.segments[0].status == "failed"
    assert run.segments[0].error
    assert run.timeline_path.is_file()
    assert run.total_duration == 0.5


def test_zero_duration_wav_fails(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.0)

    run = _align(tmp_path, [_segment(0, wav, 0.0, 0.5, None)])

    assert run.segments[0].status == "failed"
    assert "zero duration" in run.segments[0].error.lower()


def test_gap_before_segment_remains_silence(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.25, amplitude=0.5)

    run = _align(tmp_path, [_segment(0, wav, 0.5, 0.75, "SPEAKER_01")])

    with wave.open(str(run.timeline_path), "rb") as audio:
        first = audio.readframes(8000)
        placed = audio.readframes(4000)
    assert set(first) == {0}
    assert any(placed)
    assert run.total_duration == 0.75


def test_multiple_segments_keep_positions_and_speaker_metadata(tmp_path: Path) -> None:
    first = tmp_path / "tts" / "segment-0000.wav"
    second = tmp_path / "tts" / "segment-0001.wav"
    _write_wav(first, 0.25)
    _write_wav(second, 0.25)

    run = _align(
        tmp_path,
        [
            _segment(0, first, 0.0, 0.25, "SPEAKER_00"),
            _segment(1, second, 0.75, 1.0, "SPEAKER_01"),
        ],
    )

    assert [segment.segment_id for segment in run.segments] == [0, 1]
    assert [segment.speaker for segment in run.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert run.total_duration == 1.0
    with wave.open(str(run.timeline_path), "rb") as audio:
        audio.readframes(4000)
        gap = audio.readframes(8000)
    assert set(gap) == {0}


def test_overlapping_segments_are_summed_and_peak_normalized(tmp_path: Path) -> None:
    first = tmp_path / "tts" / "segment-0000.wav"
    second = tmp_path / "tts" / "segment-0001.wav"
    _write_wav(first, 0.5, amplitude=0.8)
    _write_wav(second, 0.5, amplitude=0.8)

    run = _align(
        tmp_path,
        [
            _segment(0, first, 0.0, 0.5, "SPEAKER_00"),
            _segment(1, second, 0.0, 0.5, "SPEAKER_01"),
        ],
    )

    assert run.overlap_policy == "sum_then_global_peak_normalize_to_0.99"
    assert run.peak_amplitude <= 0.99
    assert run.clipping_count == 0
    with wave.open(str(run.timeline_path), "rb") as audio:
        values = struct.unpack("<" + "h" * audio.getnframes(), audio.readframes(audio.getnframes()))
    assert max(abs(value) for value in values) < 32767


def test_empty_transcript_creates_empty_timeline(tmp_path: Path) -> None:
    run = _align(tmp_path, [])

    assert run.segments == []
    assert run.total_duration == 0
    with wave.open(str(run.timeline_path), "rb") as audio:
        assert audio.getnframes() == 0
        assert audio.getframerate() == 16000


def test_one_segment_final_duration_matches_timeline(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0004.wav"
    _write_wav(wav, 0.4)

    run = _align(tmp_path, [_segment(4, wav, 0.2, 0.6, "SPEAKER_00")])

    assert run.segments[0].segment_id == 4
    assert math.isclose(run.total_duration, 0.6, abs_tol=1 / 16000)
    assert math.isclose(_duration(run.timeline_path), 0.6, abs_tol=1 / 16000)


def test_output_rate_and_channels_are_normalized(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.25, sample_rate=8000, channels=1)
    settings = AudioAlignmentSettings(sample_rate=16000, channels=2)

    run = _align(tmp_path, [_segment(0, wav, 0.0, 0.25, "SPEAKER_00")], settings)

    with wave.open(str(run.timeline_path), "rb") as audio:
        assert audio.getframerate() == 16000
        assert audio.getnchannels() == 2
        assert audio.getsampwidth() == 2


def test_no_pad_short_is_an_explicit_configurable_behavior(tmp_path: Path) -> None:
    wav = tmp_path / "tts" / "segment-0000.wav"
    _write_wav(wav, 0.25)

    run = _align(
        tmp_path,
        [_segment(0, wav, 0.0, 0.5, None)],
        AudioAlignmentSettings(pad_short_audio=False),
    )

    assert run.segments[0].alignment_action == "leave_short"
    assert _duration(run.segments[0].output_audio_path) == 0.25
    assert run.total_duration == 0.5
