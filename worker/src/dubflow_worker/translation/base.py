from __future__ import annotations

from abc import ABC, abstractmethod


class UnsupportedLanguageError(ValueError):
    """An engine has no installed route for a requested language pair."""


class TranslationEngine(ABC):
    """Provider-neutral interface for text translation."""

    name: str
    device: str

    @abstractmethod
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """Translate one text item from source_language to target_language."""
