from __future__ import annotations

from typing import Any

from dubflow_worker.translation.base import TranslationEngine, UnsupportedLanguageError
from dubflow_worker.translation.languages import nllb_language_code


class NllbTranslationEngine(TranslationEngine):
    """Lazy Transformers adapter for NLLB-200. Model weights are never downloaded at import."""

    name = "facebook/nllb-200-distilled-600M"

    def __init__(self, model_name: str = name, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._tokenizer: Any = None
        self._model: Any = None

    def validate_language_pair(self, source_language: str, target_language: str) -> None:
        try:
            nllb_language_code(source_language)
            nllb_language_code(target_language)
        except ValueError as exc:
            raise UnsupportedLanguageError(str(exc)) from exc

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLLB requires the worker's nllb extra (transformers, sentencepiece, and torch)."
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"NLLB device {self.device!r} requested but CUDA is unavailable")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang="jpn_Jpan")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not text.strip():
            return ""
        self.validate_language_pair(source_language, target_language)
        self.load()
        source_code = nllb_language_code(source_language)
        target_code = nllb_language_code(target_language)
        self._tokenizer.src_lang = source_code
        encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        target_id = self._tokenizer.convert_tokens_to_ids(target_code)
        with self._torch.inference_mode():
            generated = self._model.generate(
                **encoded, forced_bos_token_id=target_id, max_new_tokens=256
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
