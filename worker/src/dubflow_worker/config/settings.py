from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    asr_model: str = "base"
    asr_device: str = "cuda"
    asr_compute_type: str | None = None
    ffmpeg_path: str = "ffmpeg"

    def __post_init__(self) -> None:
        if not self.asr_model.strip():
            raise ValueError("ASR model must not be empty")
        if self.asr_device not in {"cuda", "cpu"}:
            raise ValueError("ASR device must be 'cuda' or 'cpu'")
        if self.asr_compute_type is None:
            object.__setattr__(
                self, "asr_compute_type", "float16" if self.asr_device == "cuda" else "int8"
            )
        if not self.asr_compute_type or not self.asr_compute_type.strip():
            raise ValueError("ASR compute type must not be empty")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            asr_model=os.getenv("DUBFLOW_ASR_MODEL", "base"),
            asr_device=os.getenv("DUBFLOW_ASR_DEVICE", "cuda").strip().lower(),
            asr_compute_type=os.getenv("DUBFLOW_ASR_COMPUTE_TYPE") or None,
            ffmpeg_path=os.getenv("DUBFLOW_FFMPEG", "ffmpeg"),
        )

    def with_overrides(
        self,
        *,
        asr_model: str | None = None,
        asr_device: str | None = None,
        asr_compute_type: str | None = None,
    ) -> Settings:
        device = asr_device or self.asr_device
        return Settings(
            asr_model=asr_model or self.asr_model,
            asr_device=device,
            asr_compute_type=asr_compute_type or (
                None if asr_device and asr_device != self.asr_device else self.asr_compute_type
            ),
            ffmpeg_path=self.ffmpeg_path,
        )

    def model_cache_path(self) -> Path | None:
        """Return an explicit local model path, if the configured value is one."""
        candidate = Path(self.asr_model).expanduser()
        return candidate if candidate.exists() else None
