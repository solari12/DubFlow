from __future__ import annotations

from dubflow_worker.asr.base import ASREngine
from dubflow_worker.asr.faster_whisper import FasterWhisperASR
from dubflow_worker.config.settings import Settings


def create_asr_engine(settings: Settings) -> ASREngine:
    """Create the selected ASR implementation behind the engine interface."""
    return FasterWhisperASR(settings)
