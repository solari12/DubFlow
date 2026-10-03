from __future__ import annotations

import os
from pathlib import Path

from dubflow_worker.translation.base import TranslationEngine, UnsupportedLanguageError


class ArgosTranslationEngine(TranslationEngine):
    """CPU-only local Argos Translate adapter; language packages are installed separately."""

    name = "argos-translate"
    device = "cpu"

    def __init__(self) -> None:
        worker_root = Path(__file__).resolve().parents[3]
        cache_root = worker_root / ".model-cache"
        os.environ.setdefault("ARGOS_PACKAGES_DIR", str(cache_root / "argos-packages"))
        os.environ.setdefault("XDG_CONFIG_HOME", str(cache_root / "argos-config"))
        os.environ.setdefault("XDG_DATA_HOME", str(cache_root / "argos-data"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "argos-cache"))
        # Argos reads its device configuration when imported. Keep this workload off the GPU.
        os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
        # The bundled package's Stanza metadata can trigger a resource download. Use the
        # lightweight built-in sentence splitter so translation works without network access.
        os.environ["ARGOS_CHUNK_TYPE"] = "MINISBD"
        try:
            from argostranslate import translate
        except ImportError as exc:
            raise RuntimeError(
                "Argos Translate is not installed. Install the worker translation extra."
            ) from exc
        self._translate_api = translate
        self._translations: dict[tuple[str, str], object] = {}

    def _get_translation(self, source_language: str, target_language: str):
        source = source_language.strip().lower()
        target = target_language.strip().lower()
        key = (source, target)
        if key not in self._translations:
            languages = {language.code: language for language in self._translate_api.get_installed_languages()}
            if source not in languages or target not in languages:
                raise UnsupportedLanguageError(
                    f"Argos has no installed language package for {source!r} -> {target!r}."
                )
            translation = languages[source].get_translation(languages[target])
            if translation is None:
                raise UnsupportedLanguageError(
                    f"Argos has no installed translation route for {source!r} -> {target!r}."
                )
            self._translations[key] = translation
        return self._translations[key]

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not text.strip():
            return ""
        translation = self._get_translation(source_language, target_language)
        return translation.translate(text)  # type: ignore[attr-defined]
