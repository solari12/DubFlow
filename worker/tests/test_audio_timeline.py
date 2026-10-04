from __future__ import annotations

import struct
import wave
from pathlib import Path

from dubflow_worker.audio.timeline import detect_boundary_pauses, plan_dialogue_timeline
from dubflow_worker.models.alignment import AudioAlignmentSettings
from dubflow_worker.pipeline.align_audio import align_translated_transcript


def _inputs(
    tmp_path: Path,
    windows: list[tuple[float, float, float, str]],
) -> list[dict]:
    rows = []
    for index, (start, end, duration, speaker) in enumerate(windows):
        audio_path = tmp_path / f"segment-{index:04d}.wav"
        frames = round(duration * 16000)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack("<h", 5000) * frames)
        rows.append({
            "segment_id": index,
            "speaker_id": speaker,
            "source_start": start,
            "source_end": end,
            "translated_text": f"Line {index}",
            "generated_tts_duration": duration,
            "tts_audio_path": audio_path,
        })
    return rows


def test_two_adjacent_segments_get_non_overlapping_plans(tmp_path: Path) -> None:
    plans = plan_dialogue_timeline(_inputs(tmp_path, [
        (0, 1, 0.9, "SPEAKER_00"), (1, 2, 0.9, "SPEAKER_01"),
    ]))
    assert plans[0].planned_end <= plans[1].planned_start
    assert plans[0].preserved_pause_after == 0
    assert plans[1].preserved_pause_before == 0


def test_source_pause_is_preserved_in_plan(tmp_path: Path) -> None:
    plans = plan_dialogue_timeline(_inputs(tmp_path, [
        (0, 2.8, 2.0, "SPEAKER_00"), (3.6, 6.0, 2.0, "SPEAKER_01"),
    ]))
    assert plans[0].preserved_pause_after == 0.8
    assert plans[1].preserved_pause_before == 0.8
    assert plans[1].planned_start == 3.6


def test_quiet_pause_at_transcript_boundary_is_detected_and_planned(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    samples = [5000] * round(0.7 * 16000)
    samples += [0] * round(0.2 * 16000)
    samples += [5000] * round(1.1 * 16000)
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    transcript = [{"id": 0, "start": 0.0, "end": 0.7}, {"id": 1, "start": 0.7, "end": 2.0}]
    pauses = detect_boundary_pauses(source, transcript)
    assert pauses[1] == 0.2

    inputs = _inputs(tmp_path, [(0.0, 0.7, 0.6, "SPEAKER_00"), (0.7, 2.0, 1.0, "SPEAKER_01")])
    inputs[1]["source_pause_before"] = pauses[1]
    plans = plan_dialogue_timeline(inputs)
    assert plans[1].planned_start == 0.9
    assert plans[0].preserved_pause_after == 0.2


def test_slight_overflow_is_marked_for_bounded_stretch(tmp_path: Path) -> None:
    plan = plan_dialogue_timeline(_inputs(tmp_path, [(0, 1, 1.05, "SPEAKER_00")]))[0]
    assert plan.overflow_before_fitting is True
    assert plan.overflow_after_fitting is False
    assert plan.requires_concise_rephrasing is False
    assert 0.85 <= plan.stretch_ratio < 1


def test_severe_overflow_requires_concise_rephrasing(tmp_path: Path) -> None:
    plan = plan_dialogue_timeline(_inputs(tmp_path, [(0, 1, 2.0, "SPEAKER_00")]))[0]
    assert plan.overflow_before_fitting is True
    assert plan.overflow_after_fitting is True
    assert plan.requires_concise_rephrasing is True


def test_exhausted_source_window_spills_forward_instead_of_failing(tmp_path: Path) -> None:
    plans = plan_dialogue_timeline(
        _inputs(tmp_path, [
            (0.0, 0.1, 1.0, "SPEAKER_00"),
            (0.1, 0.2, 0.3, "SPEAKER_01"),
        ]),
        preserve_overflow=True,
    )

    assert plans[1].allowed_duration == 1 / 16000
    assert plans[1].overflow_before_fitting is True
    assert plans[1].planned_start >= plans[0].planned_end
    assert plans[1].planned_end > plans[1].planned_start


def test_multiple_speakers_and_consecutive_segments_keep_order(tmp_path: Path) -> None:
    plans = plan_dialogue_timeline(_inputs(tmp_path, [
        (0, 0.7, 0.6, "SPEAKER_00"),
        (0.7, 1.4, 0.6, "SPEAKER_01"),
        (1.4, 2.1, 0.6, "SPEAKER_00"),
    ]))
    assert [plan.segment_id for plan in plans] == [0, 1, 2]
    assert [plan.speaker_id for plan in plans] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert all(left.planned_end <= right.planned_start for left, right in zip(plans, plans[1:]))


def test_final_timeline_has_no_overlap_and_keeps_pause_silent(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, [
        (0, 0.8, 1.0, "SPEAKER_00"),
        (1.2, 2.0, 1.0, "SPEAKER_01"),
        (2.4, 3.2, 1.0, "SPEAKER_00"),
    ])
    transcript = {"segments": [
        {
            "id": row["segment_id"], "start": row["source_start"], "end": row["source_end"],
            "speaker": row["speaker_id"], "target_text": row["translated_text"],
            "audio_path": str(row["tts_audio_path"]),
        }
        for row in inputs
    ]}
    run = align_translated_transcript(
        transcript,
        tts_output_dir=tmp_path,
        output_dir=tmp_path / "aligned",
        settings=AudioAlignmentSettings(),
    )
    assert run.planned_overlap_count == 0
    assert run.actual_overlap_count == 0
    assert [item.planned_start for item in run.segments] == [0, 1.2, 2.4]
    with wave.open(str(run.timeline_path), "rb") as wav:
        rate = wav.getframerate()
        samples = struct.unpack("<" + "h" * wav.getnframes(), wav.readframes(wav.getnframes()))
    assert not any(samples[round(1.0 * rate):round(1.2 * rate)])
    assert not any(samples[round(2.2 * rate):round(2.4 * rate)])
