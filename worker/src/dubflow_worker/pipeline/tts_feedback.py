from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dubflow_worker.models.dubbing_quality import DubbingQualitySettings, is_severe_overflow
from dubflow_worker.models.tts import TTSResult
from dubflow_worker.translation.base import TranslationEngine
from dubflow_worker.tts.base import TTSEngine


@dataclass(frozen=True, slots=True)
class FeedbackSynthesis:
    segment_id: int
    speaker: str | None
    source_text: str
    original_translation: str
    final_translation: str
    target_duration: float
    tts_duration_before_fit: float | None
    tts_duration_after_fit: float | None
    translation_retry_count: int
    overflow: bool
    overflow_duration: float
    severe_overflow: bool
    audio_path: Path | None
    error: str | None = None
    shortening_attempted: bool = False
    shortened: bool = False
    requires_concise_rephrasing: bool = False
    forcibly_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "speaker": self.speaker,
            "source_text": self.source_text,
            "original_translation": self.original_translation,
            "final_translation": self.final_translation,
            "target_duration": self.target_duration,
            "tts_duration_before_fit": self.tts_duration_before_fit,
            "tts_duration_after_fit": self.tts_duration_after_fit,
            "translation_retry_count": self.translation_retry_count,
            "overflow": self.overflow,
            "overflow_duration": self.overflow_duration,
            "severe_overflow": self.severe_overflow,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "error": self.error,
            "shortening_attempted": self.shortening_attempted,
            "shortened": self.shortened,
            "requires_concise_rephrasing": self.requires_concise_rephrasing,
            "forcibly_truncated": self.forcibly_truncated,
        }


def synthesize_with_duration_feedback(
    segments: list[dict[str, Any]],
    *,
    source_language: str,
    target_language: str,
    tts_engine: TTSEngine,
    translation_engine: TranslationEngine,
    output_dir: Path,
    settings: DubbingQualitySettings | None = None,
) -> list[FeedbackSynthesis]:
    """Synthesize once, then allow at most the configured concise rephrase retries."""
    settings = settings or DubbingQualitySettings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[FeedbackSynthesis] = []
    for segment in segments:
        segment_id = int(segment["id"])
        speaker = segment.get("speaker")
        source_text = str(segment.get("source_text", segment.get("text", "")))
        original = segment.get("target_text")
        target_duration = float(segment["end"]) - float(segment["start"])
        if not isinstance(original, str) or not original.strip() or target_duration <= 0:
            results.append(FeedbackSynthesis(
                segment_id, speaker, source_text, str(original or ""), str(original or ""),
                target_duration, None, None, 0, False, 0.0, False, None,
                "Translation is empty or target duration is invalid",
            ))
            continue

        chosen_text = original
        chosen_result: TTSResult | None = None
        before_duration: float | None = None
        retries = 0
        shortening_attempted = False
        try:
            initial_path = output_dir / f"segment-{segment_id:04d}-initial.wav"
            chosen_result = tts_engine.synthesize(original, target_language, initial_path, speaker)
            before_duration = chosen_result.duration
            # Alignment can fit through this limit without changing the words.
            fit_limit = target_duration / settings.min_time_stretch_ratio
            while (
                chosen_result.duration > fit_limit
                and retries < settings.max_translation_retries
            ):
                shortening_attempted = True
                candidate = translation_engine.shorten_for_duration(
                    chosen_text, fit_limit, target_language
                )
                if candidate == chosen_text:
                    try:
                        candidate = translation_engine.shorten_for_dubbing(
                            chosen_text,
                            source_text,
                            source_language,
                            target_language,
                            settings.max_translation_expansion_ratio,
                        )
                    except NotImplementedError:
                        candidate = chosen_text
                if (
                    not isinstance(candidate, str)
                    or not candidate.strip()
                    or len(candidate) >= len(chosen_text)
                    or len(candidate) > len(chosen_text) * settings.max_translation_expansion_ratio
                ):
                    break

                retries += 1
                candidate_path = output_dir / f"segment-{segment_id:04d}-retry-{retries}.wav"
                retry_result = tts_engine.synthesize(
                    candidate, target_language, candidate_path, speaker
                )
                if retry_result.duration >= chosen_result.duration:
                    candidate_path.unlink(missing_ok=True)
                    break
                chosen_text = candidate
                chosen_result = retry_result

            final_path = output_dir / f"segment-{segment_id:04d}.wav"
            if chosen_result.audio_path != final_path:
                shutil.copyfile(chosen_result.audio_path, final_path)
            for candidate_path in output_dir.glob(f"segment-{segment_id:04d}-*.wav"):
                candidate_path.unlink(missing_ok=True)
            after_duration = chosen_result.duration
            fit_limit = target_duration / settings.min_time_stretch_ratio
            overflow = after_duration > fit_limit + 0.5 / chosen_result.sample_rate
            overflow_duration = max(0.0, after_duration - target_duration) if overflow else 0.0
            results.append(FeedbackSynthesis(
                segment_id,
                speaker,
                source_text,
                original,
                chosen_text,
                target_duration,
                before_duration,
                after_duration,
                retries,
                overflow,
                round(overflow_duration, 6),
                is_severe_overflow(after_duration, target_duration, settings),
                final_path,
                shortening_attempted=shortening_attempted,
                shortened=chosen_text != original,
                requires_concise_rephrasing=after_duration > fit_limit + 0.5 / chosen_result.sample_rate,
            ))
        except Exception as exc:
            results.append(FeedbackSynthesis(
                segment_id,
                speaker,
                source_text,
                original,
                chosen_text,
                target_duration,
                before_duration,
                chosen_result.duration if chosen_result else None,
                retries,
                True,
                max(0.0, (chosen_result.duration if chosen_result else 0.0) - target_duration),
                is_severe_overflow(chosen_result.duration, target_duration, settings)
                if chosen_result else False,
                chosen_result.audio_path if chosen_result else None,
                f"{type(exc).__name__}: {exc}",
                shortening_attempted=shortening_attempted,
                shortened=chosen_text != original,
                requires_concise_rephrasing=bool(
                    chosen_result
                    and chosen_result.duration > target_duration / settings.min_time_stretch_ratio
                ),
            ))
    return results
