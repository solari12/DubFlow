from __future__ import annotations

import os

from dubflow_worker.translation.base import TranslationEngine


def create_translation_engine(
    provider: str = "argos", *, model: str | None = None, device: str | None = None
) -> TranslationEngine:
    """Create the explicitly selected provider; never silently switch providers."""
    if provider == "argos":
        from dubflow_worker.translation.argos import ArgosTranslationEngine

        return ArgosTranslationEngine()
    if provider == "nllb":
        from dubflow_worker.translation.nllb import NllbTranslationEngine

        return NllbTranslationEngine(
            model_name=model or os.getenv("DUBFLOW_NLLB_MODEL") or NllbTranslationEngine.name,
            device=device or os.getenv("DUBFLOW_NLLB_DEVICE") or "cpu",
        )
    raise ValueError(f"Unknown translation provider: {provider}")
