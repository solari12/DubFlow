from __future__ import annotations

import re
from abc import ABC, abstractmethod


class NaturalizationProvider(ABC):
    """Provider-neutral post-translation rewrite interface."""

    name: str

    @abstractmethod
    def naturalize(self, text: str, *, language: str, source_context: str) -> str:
        """Rewrite translated text without changing its meaning or adding facts."""

    def review_reason(self, text: str, *, naturalized_text: str, language: str) -> str | None:
        del text, naturalized_text, language
        return None


class DeterministicVietnameseNaturalizer(NaturalizationProvider):
    """Conservative Vietnamese surface rewrites; unchanged or unsafe text is reviewed."""

    name = "deterministic-vi-rules-v2"
    _rules = (
        (re.compile(r"\bh\u00e3y \u0111\u1ec3 t\u00f4i\b", re.IGNORECASE), "\u0111\u1ec3 t\u00f4i"),
        (re.compile(r"t\u00f4i \u0111\u00e3 h\u1ecdc ti\u1ebfng Nh\u1eadt trong b\u1ea3y n\u0103m, v\u00e0 (?:h\u00e3y )?\u0111\u1ec3 t\u00f4i d\u1ea1y c\u00e1c b\u1ea1n l\u00e0m th\u1ebf n\u00e0o \u0111\u1ec3 l\u00e0m n\u00f3 ch\u00ednh x\u00e1c trong m\u1ed9t", re.IGNORECASE), "t\u00f4i \u0111\u00e3 h\u1ecdc ti\u1ebfng Nh\u1eadt \u0111\u01b0\u1ee3c b\u1ea3y n\u0103m; \u0111\u1ec3 t\u00f4i ch\u1ec9 cho c\u00e1c b\u1ea1n c\u00e1ch l\u00e0m ch\u00ednh x\u00e1c trong m\u1ed9t"),
        (re.compile(r"\bdownload\b", re.IGNORECASE), "t\u1ea3i v\u1ec1"),
        (re.compile(r"\b\u0111i\u1ec1u \u0111\u1ea7u ti\u00ean t\u00f4i l\u00e0m l\u00e0 \u1ee9ng d\u1ee5ng\b", re.IGNORECASE), "vi\u1ec7c \u0111\u1ea7u ti\u00ean t\u00f4i l\u00e0m l\u00e0 d\u00f9ng \u1ee9ng d\u1ee5ng"),
        (re.compile(r"\bt\u00f4i ngh\u0129 r\u1eb1ng\b", re.IGNORECASE), "t\u00f4i ngh\u0129"),
        (re.compile(r"\bv\u00ec v\u1eady ch\u1ecdn m\u1ed9t ho\u1eb7c hai\b", re.IGNORECASE), "v\u1eady h\u00e3y ch\u1ecdn m\u1ed9t ho\u1eb7c hai"),
        (re.compile(r"\bn\u1ebfu \u0111i\u1ec1u \u0111\u00f3 c\u00f3 \u00fd ngh\u0129a\b", re.IGNORECASE), "n\u1ebfu b\u1ea1n hi\u1ec3u \u00fd t\u00f4i"),
        (re.compile(r"\b\u0111i\u1ec1u \u0111\u1ea7u ti\u00ean b\u1ea1n mu\u1ed1n l\u00e0m l\u00e0 h\u1ecdc\b", re.IGNORECASE), "tr\u01b0\u1edbc ti\u00ean, b\u1ea1n n\u00ean h\u1ecdc"),
        (re.compile(r"\bhiragana\s+katakana\b", re.IGNORECASE), "Hiragana, Katakana"),
        (re.compile(r"\bba h\u1ec7 th\u1ed1ng vi\u1ebft ch\u00ednh\b", re.IGNORECASE), "ba h\u1ec7 ch\u1eef vi\u1ebft ch\u00ednh"),
        (re.compile(r"\bn\u00ean h\u00e3y h\u1ecdc khi b\u1ea1n s\u1eed d\u1ee5ng m\u1ed9t \u1ee9ng d\u1ee5ng\b", re.IGNORECASE), "b\u1ea1n c\u00f3 th\u1ec3 h\u1ecdc d\u1ea7n khi d\u00f9ng \u1ee9ng d\u1ee5ng"),
        (re.compile(r"\bv\u00e0 s\u1edbm (?:th\u00f4i )?b\u1ea1n s\u1ebd th\u1ea5y\b", re.IGNORECASE), "r\u1ed3i b\u1ea1n s\u1ebd th\u1ea5y"),
        (re.compile(r", \u0111\u00f3 l\u00e0 ng\u1eef ph\u00e1p, l\u00e0 m\u1ed9t ch\u00fat nh\u1ea7m l\u1eabn", re.IGNORECASE), " l\u00e0 ng\u1eef ph\u00e1p h\u01a1i kh\u00f3 hi\u1ec3u"),
        (re.compile(r"\b\u0111\u1ec3 l\u00e0 m\u1ed9t ch\u00fat nh\u1ea7m l\u1eabn\b", re.IGNORECASE), "c\u00f3 th\u1ec3 h\u01a1i kh\u00f3 hi\u1ec3u"),
        (re.compile(r"\b\u0111\u00f3 l\u00e0 n\u01a1i t\u00f4i khuy\u00ean b\u1ea1n n\u00ean (?:th\u1ef1c hi\u1ec7n|tri\u1ec3n khai)\b", re.IGNORECASE), "\u0111\u00f3 l\u00e0 l\u00fac t\u00f4i khuy\u00ean b\u1ea1n n\u00ean tham kh\u1ea3o"),
        (re.compile(r"\bth\u1ef1c hi\u1ec7n (?=Takem)", re.IGNORECASE), "tham kh\u1ea3o "),
        (re.compile(r"\b(?:v\u00e0 )?sau \u0111\u00f3 khi b\u1ea1n ti\u1ebfp t\u1ee5c h\u1ecdc, t\u1ea1o ra c\u00e1c c\u00e2u v\u00ed d\u1ee5, nh\u01b0 s\u1eed d\u1ee5ng nh\u1eefng g\u00ec b\u1ea1n \u0111ang h\u1ecdc", re.IGNORECASE), "Khi ti\u1ebfp t\u1ee5c h\u1ecdc, h\u00e3y \u0111\u1eb7t c\u00e2u v\u00ed d\u1ee5 v\u00e0 \u00e1p d\u1ee5ng nh\u1eefng g\u00ec b\u1ea1n \u0111ang h\u1ecdc"),
        (re.compile(r"\b\u0111\u00f3 l\u00e0 ngu\u1ed3n mi\u1ec5n ph\u00ed n\u00e0y\b", re.IGNORECASE), "\u0111\u00e2y l\u00e0 t\u00e0i li\u1ec7u mi\u1ec5n ph\u00ed"),
        (re.compile(r"\bngu\u1ed3n mi\u1ec5n ph\u00ed n\u00e0y\b", re.IGNORECASE), "t\u00e0i li\u1ec7u mi\u1ec5n ph\u00ed n\u00e0y"),
        (re.compile(r"[.;,]?\s+\bv\u00e0 s\u1eed d\u1ee5ng n\u00f3, h\u1ecdc\b", re.IGNORECASE), ". H\u00e3y d\u00f9ng t\u00e0i li\u1ec7u \u0111\u00f3 \u0111\u1ec3 h\u1ecdc"),
        (re.compile(r"\bn\u00ean h\u00e3y h\u1ecdc r\u1eb1ng khi b\u1ea1n s\u1eed d\u1ee5ng (?:m\u1ed9t )?\u1ee9ng d\u1ee5ng\b", re.IGNORECASE), "h\u00e3y h\u1ecdc d\u1ea7n qua \u1ee9ng d\u1ee5ng"),
        (re.compile(r"\bngu\u1ed3n l\u1ef1c mi\u1ec5n ph\u00ed\b", re.IGNORECASE), "t\u00e0i li\u1ec7u mi\u1ec5n ph\u00ed"),
        (re.compile(r"[.;,]?\s+\bv\u00e0 s\u1eed d\u1ee5ng \u0111\u00f3, h\u1ecdc\b", re.IGNORECASE), ". H\u00e3y d\u00f9ng t\u00e0i li\u1ec7u \u0111\u00f3 \u0111\u1ec3 h\u1ecdc"),
    )
    _unsafe_patterns = (
        (re.compile(r"\bblog\s+v\u00e0\s+l\u1ed7i\s+t\u1ed1t\b", re.IGNORECASE), "Ambiguous phrase 'blog v\u00e0 l\u1ed7i t\u1ed1t' was left literal for review."),
        (re.compile(r"\bt\u00f4i \u0111\u00e3 l\u00e0m m\u1ed9t s\u1ed1 ng\u01b0\u1eddi \u1edf nh\u1eadt b\u1ea3n\b", re.IGNORECASE), "Known malformed proper-name rendering was left literal for review."),
        (re.compile(r"\b\u0111\u00e3 l\u00e0m Cuban Japanese\b", re.IGNORECASE), "The phrase 'did Cuban Japanese' may be an ASR or proper-name error; preserve it for bilingual review."),
        (re.compile(r"\bth\u1ef1c hi\u1ec7n (?:h\u01b0\u1edbng d\u1eabn|takem)\b", re.IGNORECASE), "The guide recommendation remains an unnatural literal construction."),
        (re.compile(r"\b\u0111\u1ec3 l\u00e0 m\u1ed9t ch\u00fat nh\u1ea7m l\u1eabn\b", re.IGNORECASE), "The confusion clause could not be safely rewritten."),
    )

    def naturalize(self, text: str, *, language: str, source_context: str) -> str:
        del source_context
        if language.split("-", 1)[0].lower() != "vi":
            return text
        result = text
        for pattern, replacement in self._rules:
            result = pattern.sub(
                lambda match: replacement[:1].upper() + replacement[1:]
                if match.group(0)[:1].isupper()
                else replacement,
                result,
            )
        if result:
            result = result[0].upper() + result[1:]
        return result

    def review_reason(self, text: str, *, naturalized_text: str, language: str) -> str | None:
        if language.split("-", 1)[0].lower() != "vi":
            return "The deterministic local naturalizer only supports Vietnamese."
        for pattern, reason in self._unsafe_patterns:
            if pattern.search(naturalized_text):
                return reason
        if naturalized_text == text:
            return "No safe deterministic rewrite matched; preserve the literal translation for human review."
        return None
