from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DubbingQualitySettings:
    """Conservative controls for duration fitting and one bounded rephrase retry."""

    min_time_stretch_ratio: float = 0.85
    max_time_stretch_ratio: float = 1.10
    max_translation_expansion_ratio: float = 1.25
    max_overflow_ratio: float = 1.50
    max_translation_retries: int = 1

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.min_time_stretch_ratio,
            self.max_time_stretch_ratio,
            self.max_translation_expansion_ratio,
            self.max_overflow_ratio,
        )):
            raise ValueError("Dubbing quality ratios must be finite numbers")
        if not 0.75 <= self.min_time_stretch_ratio <= 1.0:
            raise ValueError("min_time_stretch_ratio must be between 0.75 and 1.0")
        if not 1.0 <= self.max_time_stretch_ratio <= 1.25:
            raise ValueError("max_time_stretch_ratio must be between 1.0 and 1.25")
        if not 1.0 <= self.max_translation_expansion_ratio <= 2.0:
            raise ValueError("max_translation_expansion_ratio must be between 1.0 and 2.0")
        if self.max_overflow_ratio <= 1.0:
            raise ValueError("max_overflow_ratio must exceed 1.0")
        if not 0 <= self.max_translation_retries <= 2:
            raise ValueError("max_translation_retries must be between 0 and 2")


def is_severe_overflow(generated_duration: float, target_duration: float, settings: DubbingQualitySettings) -> bool:
    if target_duration <= 0:
        return True
    return generated_duration / target_duration > settings.max_overflow_ratio
