from __future__ import annotations

import array
import math
import time
import wave
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dubflow_worker.audio.alignment import (
    InvalidAudioError,
    WaveData,
    fit_samples_to_frames,
    normalize_wav,
    read_pcm_wav,
    write_pcm_wav,
)
from dubflow_worker.audio.extractor import AudioExtractor
from dubflow_worker.audio.timeline import (
    TimelineSegmentPlan,
    detect_boundary_pauses,
    plan_dialogue_timeline,
)
from dubflow_worker.models.alignment import (
    AlignedSegment,
    AlignmentRun,
    AudioAlignmentSettings,
    valid_timestamps,
)


def _find_audio_path(
    segment: Mapping[str, Any], tts_output_dir: Path, transcript_path: Path | None
) -> Path:
    audio_info = segment.get("audio")
    value = audio_info.get("audio_path") if isinstance(audio_info, Mapping) else None
    value = value or segment.get("audio_path")
    candidates: list[Path] = []
    if value:
        path = Path(str(value))
        candidates.append(path if path.is_absolute() else Path.cwd() / path)
        candidates.append(tts_output_dir / path.name)
        if transcript_path:
            candidates.append(transcript_path.parent / path)
    segment_id = int(segment.get("id", segment.get("segment_id", -1)))
    if segment_id >= 0:
        candidates.append(tts_output_dir / f"segment-{segment_id:04d}.wav")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No Phase 1D audio mapping for segment {segment_id}")


def _audio_metadata(path: Path) -> tuple[int, int, int, int, str]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                wav_file.getnframes(),
                wav_file.getframerate(),
                wav_file.getnchannels(),
                wav_file.getsampwidth(),
                wav_file.getcomptype(),
            )
    except (wave.Error, OSError) as exc:
        raise InvalidAudioError(f"Could not read WAV metadata: {exc}") from exc


def _convert_source(
    source: Path,
    temporary_path: Path,
    *,
    source_metadata: tuple[int, int, int, int, str],
    settings: AudioAlignmentSettings,
    atempo: float | None = None,
) -> Path:
    frames, sample_rate, channels, sample_width, compression = source_metadata
    if frames < 1 or sample_rate < 1 or channels < 1:
        raise InvalidAudioError("WAV has zero duration or invalid audio metadata")
    if compression != "NONE":
        raise InvalidAudioError(f"Unsupported WAV compression: {compression}")
    if sample_width == 2 and sample_rate == settings.sample_rate and channels == settings.channels and atempo is None:
        return source

    from dubflow_worker.audio.alignment import convert_wav

    convert_wav(source, temporary_path, settings=settings, atempo=atempo)
    return temporary_path


def _failed_segment(segment: Mapping[str, Any], exc: Exception) -> AlignedSegment:
    segment_id = int(segment.get("id", segment.get("segment_id", -1)))
    start = float(segment.get("start", segment.get("target_start", 0.0)))
    end = float(segment.get("end", segment.get("target_end", start)))
    return AlignedSegment(
        segment_id=segment_id,
        speaker=segment.get("speaker"),
        target_start=start,
        target_end=end,
        target_duration=round(max(0.0, end - start), 9),
        original_tts_duration=None,
        final_audio_duration=None,
        duration_error=None,
        stretch_factor=None,
        alignment_action="failed",
        output_audio_path=None,
        status="failed",
        error=f"{type(exc).__name__}: {exc}",
    )


def _align_segment(
    segment: Mapping[str, Any],
    plan: TimelineSegmentPlan,
    *,
    output_dir: Path,
    settings: AudioAlignmentSettings,
    temporary_dir: Path,
) -> AlignedSegment:
    segment_id = plan.segment_id
    source_start = plan.source_start
    source_end = plan.source_end
    start = plan.planned_start
    target_duration = plan.allowed_duration
    end = plan.planned_end
    source_path = plan.tts_audio_path
    if not source_path.is_file():
        raise FileNotFoundError(f"TTS WAV is missing: {source_path}")

    metadata = _audio_metadata(source_path)
    source_frames, source_rate, source_channels, source_width, source_compression = metadata
    if source_frames < 1 or source_rate < 1 or source_channels < 1:
        raise InvalidAudioError("WAV has zero duration or invalid audio metadata")
    if source_compression != "NONE":
        raise InvalidAudioError(f"Unsupported WAV compression: {source_compression}")
    original_duration = source_frames / source_rate
    factor = target_duration / original_duration
    target_frames = max(1, round(target_duration * settings.sample_rate))
    temporary_wav = temporary_dir / f"{segment_id}-normalized.wav"
    output_path = output_dir / "aligned_audio" / f"segment-{segment_id:04d}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shortening_attempted = bool(segment.get("shortening_attempted"))
    shortened = bool(segment.get("shortened"))
    overflow_before = original_duration > target_duration + 0.5 / settings.sample_rate
    requires_rephrase = overflow_before and factor < settings.min_stretch_factor
    forcibly_truncated = False
    truncated_duration = 0.0
    overflow_after = False

    if math.isclose(original_duration, target_duration, rel_tol=0.0, abs_tol=0.5 / settings.sample_rate):
        normalized_path = _convert_source(
            source_path,
            temporary_wav,
            source_metadata=metadata,
            settings=settings,
        )
        audio = read_pcm_wav(normalized_path)
        samples = fit_samples_to_frames(audio.samples, target_frames, settings.channels)
        action = "exact"
        status = "success"
        stretch_factor = 1.0
        error = None
    elif original_duration < target_duration:
        normalized_path = _convert_source(
            source_path,
            temporary_wav,
            source_metadata=metadata,
            settings=settings,
        )
        audio = read_pcm_wav(normalized_path)
        if settings.pad_short_audio:
            samples = fit_samples_to_frames(audio.samples, target_frames, settings.channels)
            action = "pad_silence"
        else:
            samples = array.array("h", audio.samples)
            action = "leave_short"
        status = "success"
        stretch_factor = 1.0
        error = None
    elif settings.min_stretch_factor <= factor <= settings.max_stretch_factor:
        normalized_path = _convert_source(
            source_path,
            temporary_wav,
            source_metadata=metadata,
            settings=settings,
            atempo=1.0 / factor,
        )
        audio = read_pcm_wav(normalized_path)
        # Bound FFmpeg rounding differences to at most one sample at the requested rate.
        samples = fit_samples_to_frames(audio.samples, target_frames, settings.channels)
        action = "time_stretch"
        status = "success"
        stretch_factor = factor
        error = None
    elif settings.overflow_policy == "fail":
        return AlignedSegment(
            segment_id=segment_id,
            speaker=segment.get("speaker"),
            target_start=start,
            target_end=end,
            target_duration=target_duration,
            original_tts_duration=original_duration,
            final_audio_duration=None,
            duration_error=None,
            stretch_factor=factor,
            alignment_action="overflow_fail",
            output_audio_path=None,
            status="overflow",
            error=(
                f"Required stretch ratio {factor:.6f} is below the safe minimum "
                f"{settings.min_stretch_factor:.3f}"
            ),
            source_start=source_start,
            source_end=source_end,
            planned_start=start,
            planned_end=end,
            allowed_duration=target_duration,
            overflow_before_fitting=overflow_before,
            overflow_after_fitting=True,
            requires_concise_rephrasing=True,
            shortening_attempted=shortening_attempted,
            shortened=shortened,
            preserved_pause_before=plan.preserved_pause_before,
            preserved_pause_after=plan.preserved_pause_after,
        )
    else:
        normalized_path = _convert_source(
            source_path,
            temporary_wav,
            source_metadata=metadata,
            settings=settings,
            atempo=1.0 / settings.min_stretch_factor
            if settings.overflow_policy == "trim"
            else None,
        )
        audio = read_pcm_wav(normalized_path)
        if settings.overflow_policy == "preserve":
            samples = array.array("h", audio.samples)
            action = "overflow_preserve"
            error = None
        else:
            samples = fit_samples_to_frames(audio.samples, target_frames, settings.channels)
            forcibly_truncated = len(audio.samples) > target_frames * settings.channels
            truncated_duration = round(max(0.0, audio.duration - target_duration), 6)
            overflow_after = forcibly_truncated
            action = "overflow_truncate_after_bounded_stretch"
            error = (
                "TTS remained longer than its source window after bounded time-stretch; "
                f"the final {truncated_duration:.3f} seconds were explicitly truncated"
            )
        status = "overflow"
        stretch_factor = settings.min_stretch_factor if settings.overflow_policy == "trim" else 1.0

    write_pcm_wav(output_path, samples, settings.sample_rate, settings.channels)
    final_audio = read_pcm_wav(output_path)
    final_duration = final_audio.duration
    return AlignedSegment(
        segment_id=segment_id,
        speaker=segment.get("speaker"),
        target_start=start,
        target_end=end,
        target_duration=target_duration,
        original_tts_duration=original_duration,
        final_audio_duration=final_duration,
        duration_error=round(final_duration - target_duration, 9),
        stretch_factor=stretch_factor,
        alignment_action=action,
        output_audio_path=output_path,
        status=status,
        error=error,
        source_start=source_start,
        source_end=source_end,
        planned_start=start,
        planned_end=end,
        allowed_duration=target_duration,
        overflow_before_fitting=overflow_before,
        overflow_after_fitting=overflow_after,
        requires_concise_rephrasing=requires_rephrase,
        shortening_attempted=shortening_attempted,
        shortened=shortened,
        forcibly_truncated=forcibly_truncated,
        truncated_duration=truncated_duration,
        preserved_pause_before=plan.preserved_pause_before,
        preserved_pause_after=plan.preserved_pause_after,
    )


def _mix_timeline(
    segments: list[AlignedSegment], *, target_duration: float, settings: AudioAlignmentSettings, output_path: Path
) -> tuple[float, float, int]:
    sample_rate = settings.sample_rate
    channels = settings.channels
    timeline_frames = max(0, round(target_duration * sample_rate))
    valid_audio: list[tuple[int, WaveData]] = []
    for segment in sorted(segments, key=lambda item: (item.target_start, item.segment_id)):
        if segment.output_audio_path is None:
            continue
        audio = read_pcm_wav(segment.output_audio_path)
        offset = round(segment.target_start * sample_rate)
        valid_audio.append((offset, audio))
        timeline_frames = max(timeline_frames, offset + audio.frames)

    mixed = [0.0] * (timeline_frames * channels)
    for offset, audio in valid_audio:
        for index, sample in enumerate(audio.samples):
            mixed[offset * channels + index] += sample / 32768.0

    peak_before = max((abs(sample) for sample in mixed), default=0.0)
    gain = min(1.0, 0.99 / peak_before) if peak_before > 0 else 1.0
    peak_after = 0.0
    clipping_count = 0
    final_samples = array.array("h")
    for value in mixed:
        value *= gain
        peak_after = max(peak_after, abs(value))
        sample = round(value * 32768.0)
        if sample > 32767 or sample < -32768:
            clipping_count += 1
            sample = max(-32768, min(32767, sample))
        final_samples.append(sample)
    write_pcm_wav(output_path, final_samples, sample_rate, channels)
    if timeline_frames == 0:
        return 0.0, 0.0, 0
    final_audio = read_pcm_wav(output_path)
    return final_audio.duration, peak_after, clipping_count


def align_translated_transcript(
    transcript: Mapping[str, Any],
    *,
    tts_output_dir: Path,
    output_dir: Path,
    settings: AudioAlignmentSettings | None = None,
    transcript_path: Path | None = None,
    source_media: Path | None = None,
) -> AlignmentRun:
    settings = settings or AudioAlignmentSettings()
    tts_output_dir = Path(tts_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_segments = list(transcript.get("segments", []))
    ids = [int(item.get("id", item.get("segment_id", -1))) for item in input_segments]
    if len(ids) != len(set(ids)):
        raise ValueError("Transcript segment IDs must be unique")
    starts = [float(item.get("start", item.get("target_start", 0.0))) for item in input_segments]
    ends = [float(item.get("end", item.get("target_end", 0.0))) for item in input_segments]
    for start, end in zip(starts, ends):
        if not valid_timestamps(start, end):
            raise ValueError("Transcript timestamps must satisfy finite 0 <= start < end")
    target_timeline_duration = max(ends, default=0.0)

    started = time.perf_counter()
    segments: list[AlignedSegment] = []
    source_pauses: dict[int, float] = {}
    if source_media is not None:
        with AudioExtractor(settings.ffmpeg_path).extract(Path(source_media)) as extracted:
            source_pauses = detect_boundary_pauses(extracted.path, input_segments)
    planning_inputs: list[dict[str, Any]] = []
    planning_failures: dict[int, Exception] = {}
    for item in input_segments:
        segment_id = int(item.get("id", item.get("segment_id", -1)))
        try:
            source_path = _find_audio_path(item, tts_output_dir, transcript_path)
            metadata = _audio_metadata(source_path)
            frames, rate, channels, width, compression = metadata
            if frames < 1:
                raise InvalidAudioError("TTS WAV has zero duration")
            if rate < 1 or channels < 1 or width != 2 or compression != "NONE":
                raise InvalidAudioError("TTS WAV must contain non-empty 16-bit PCM samples")
            planning_inputs.append(
                {
                    "segment_id": segment_id,
                    "speaker_id": item.get("speaker"),
                    "source_start": float(item.get("start", item.get("target_start", 0.0))),
                    "source_end": float(item.get("end", item.get("target_end", 0.0))),
                    "translated_text": str(item.get("target_text", "")),
                    "generated_tts_duration": frames / rate,
                    "tts_audio_path": source_path,
                    "source_pause_before": source_pauses.get(segment_id, 0.0),
                }
            )
        except Exception as exc:
            planning_failures[segment_id] = exc
    plans = plan_dialogue_timeline(
        planning_inputs,
        min_stretch_ratio=settings.min_stretch_factor,
        sample_rate=settings.sample_rate,
        preserve_overflow=settings.overflow_policy == "preserve",
    )
    plan_by_id = {plan.segment_id: plan for plan in plans}
    with TemporaryDirectory(prefix="dubflow-align-") as temporary_name:
        temporary_dir = Path(temporary_name)
        for item in input_segments:
            segment_id = int(item.get("id", item.get("segment_id", -1)))
            try:
                if segment_id in planning_failures:
                    raise planning_failures[segment_id]
                aligned = _align_segment(item, plan_by_id[segment_id],
                                         output_dir=output_dir, settings=settings,
                                         temporary_dir=temporary_dir)
            except Exception as exc:
                aligned = _failed_segment(item, exc)
            segments.append(aligned)
        ordered_successful = sorted(
            (item for item in segments if item.output_audio_path is not None),
            key=lambda item: (item.planned_start or item.target_start, item.segment_id),
        )
        actual_overlap_count = 0
        last_end = -math.inf
        for item in ordered_successful:
            item_start = item.planned_start if item.planned_start is not None else item.target_start
            item_end = item_start + (item.final_audio_duration or 0.0)
            if item_start < last_end - 0.5 / settings.sample_rate:
                actual_overlap_count += 1
            last_end = max(last_end, item_end)
        if actual_overlap_count:
            raise RuntimeError("Timeline planner produced overlapping ordinary dialogue audio")
        total_duration, peak_amplitude, clipping_count = _mix_timeline(
            segments,
            target_duration=target_timeline_duration,
            settings=settings,
            output_path=output_dir / "dubbed_timeline.wav",
        )
    runtime = time.perf_counter() - started
    rtf = runtime / target_timeline_duration if target_timeline_duration > 0 else None
    planned_overlap_count = 0
    previous_plan_end = -math.inf
    for plan in sorted(plans, key=lambda item: (item.planned_start, item.segment_id)):
        if plan.planned_start < previous_plan_end - 0.5 / settings.sample_rate:
            planned_overlap_count += 1
        previous_plan_end = max(previous_plan_end, plan.planned_end)
    return AlignmentRun(
        timeline_path=output_dir / "dubbed_timeline.wav",
        segments=segments,
        sample_rate=settings.sample_rate,
        channels=settings.channels,
        target_timeline_duration=target_timeline_duration,
        total_duration=total_duration,
        runtime_seconds=runtime,
        rtf=rtf,
        peak_amplitude=peak_amplitude,
        clipping_count=clipping_count,
        planned_overlap_count=planned_overlap_count,
        actual_overlap_count=0,
    )
