from __future__ import annotations

from abc import ABC, abstractmethod


class UnsupportedLanguageError(ValueError):
    """An engine has no installed route for a requested language pair."""


class TranslationEngine(ABC):
    """Provider-neutral interface for text translation."""

    name: str
    device: str

    def validate_language_pair(self, source_language: str, target_language: str) -> None:
        """Fail before processing when a provider cannot route this language pair."""

    def shorten_for_dubbing(
        self,
        translated_text: str,
        source_text: str,
        source_language: str,
        target_language: str,
        max_expansion_ratio: float,
    ) -> str:
        """Return a concise, faithful rephrase when the provider supports it."""
        raise NotImplementedError(f"{self.name} does not support translation rephrasing for dubbing")

    def shorten_for_duration(self, text: str, target_duration: float, language: str) -> str:
        """Return unchanged text unless a provider implements duration-aware shortening."""
        return text

    def translate_for_dubbing(self, text: str, source_language: str, target_language: str) -> str:
        """Translate faithfully for concise spoken delivery; providers may specialize this."""
        return self.translate(text, source_language, target_language)

    @abstractmethod
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """Translate one text item from source_language to target_language."""
