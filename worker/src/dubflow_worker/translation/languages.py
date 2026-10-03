"""Language codes shared by translation providers."""

NLLB_LANGUAGE_CODES = {
    "ja": "jpn_Jpan",
    "vi": "vie_Latn",
    "en": "eng_Latn",
}


def nllb_language_code(language: str) -> str:
    code = language.strip().lower().replace("_", "-").split("-", 1)[0]
    try:
        return NLLB_LANGUAGE_CODES[code]
    except KeyError as exc:
        supported = ", ".join(sorted(NLLB_LANGUAGE_CODES))
        raise ValueError(f"NLLB language {code!r} is not mapped; supported codes: {supported}") from exc
