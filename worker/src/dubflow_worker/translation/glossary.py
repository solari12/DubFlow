from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TERMS = (
    "Hiragana",
    "Katakana",
    "Kanji",
    "JLPT",
    "App Store",
    "tofugood.com",
    "Cuban Japanese",
    "Juissancei",
    "Bumpal",
    "Takem's Guide to Learning Japanese",
)


@dataclass(frozen=True, slots=True)
class Glossary:
    terms: tuple[str, ...] = DEFAULT_TERMS
    review_required_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(term.strip() for term in self.terms if term.strip()))
        object.__setattr__(self, "terms", normalized)
        review_terms = tuple(dict.fromkeys(term.strip() for term in self.review_required_terms if term.strip()))
        object.__setattr__(self, "review_required_terms", review_terms)

    @classmethod
    def from_json(cls, path: Path | None) -> Glossary:
        if path is None:
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            review_required = payload.get("review_required_terms", [])
            terms = payload.get("terms")
        else:
            review_required = []
            terms = payload
        if (
            not isinstance(terms, list)
            or any(not isinstance(term, str) for term in terms)
            or not isinstance(review_required, list)
            or any(not isinstance(term, str) for term in review_required)
        ):
            raise ValueError("Glossary JSON must be a list of strings or an object with a 'terms' list")
        return cls(tuple(terms), tuple(review_required))

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        replacements: dict[str, str] = {}
        terms = sorted(self.terms, key=lambda value: (-len(value), value.casefold()))
        masked = text
        for index, term in enumerate(terms):
            token = f"ZXQTERM{index:04d}ZXQ"
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            masked, count = pattern.subn(token, masked)
            if count:
                replacements[token] = term
        return masked, replacements

    @staticmethod
    def restore(text: str, replacements: dict[str, str]) -> str:
        missing = [token for token in replacements if token.casefold() not in text.casefold()]
        if missing:
            terms = [replacements[token] for token in missing]
            raise ValueError("Translation provider dropped protected glossary marker(s): " + ", ".join(terms))
        restored = text
        for token, term in replacements.items():
            restored = re.sub(re.escape(token), lambda _: term, restored, flags=re.IGNORECASE)
        return restored
