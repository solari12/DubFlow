from __future__ import annotations

import sys
from types import ModuleType

import pytest

from dubflow_worker.translation.factory import create_translation_engine
from dubflow_worker.translation.languages import nllb_language_code
from dubflow_worker.translation.nllb import NllbTranslationEngine


def test_language_codes_cover_fixture_route() -> None:
    assert nllb_language_code("ja") == "jpn_Jpan"
    assert nllb_language_code("vi") == "vie_Latn"
    assert nllb_language_code("en-US") == "eng_Latn"


def test_unknown_language_code_fails_clearly() -> None:
    with pytest.raises(ValueError, match="not mapped"):
        nllb_language_code("xx")


def test_provider_factory_selects_nllb_without_loading_model() -> None:
    engine = create_translation_engine("nllb")
    assert isinstance(engine, NllbTranslationEngine)
    assert engine.model_name == "facebook/nllb-200-distilled-600M"
    assert engine.device == "cpu"
    assert engine._model is None


def test_provider_factory_reads_nllb_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUBFLOW_NLLB_MODEL", "local/nllb-checkpoint")
    monkeypatch.setenv("DUBFLOW_NLLB_DEVICE", "cpu:0")
    engine = create_translation_engine("nllb")
    assert isinstance(engine, NllbTranslationEngine)
    assert engine.model_name == "local/nllb-checkpoint"
    assert engine.device == "cpu:0"


def test_nllb_lazy_load_and_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTensor:
        def to(self, device: str):
            assert device == "cpu"
            return self

    class FakeTokenizer:
        src_lang = ""

        def __call__(self, text: str, **kwargs):
            assert text == "こんにちは"
            assert kwargs["truncation"] is True
            return {"input_ids": FakeTensor()}

        def convert_tokens_to_ids(self, value: str) -> int:
            assert value == "vie_Latn"
            return 17

        def batch_decode(self, values, skip_special_tokens: bool):
            assert skip_special_tokens
            return ["Xin chào"]

    class FakeModel:
        def to(self, device: str):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["forced_bos_token_id"] == 17
            return [object()]

    torch_module = ModuleType("torch")
    torch_module.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: False)})()
    torch_module.inference_mode = lambda: type("Context", (), {
        "__enter__": lambda self: None, "__exit__": lambda self, *args: None
    })()
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **k: FakeTokenizer())}
    )
    transformers_module.AutoModelForSeq2SeqLM = type(
        "AutoModel", (), {"from_pretrained": staticmethod(lambda *a, **k: FakeModel())}
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    engine = NllbTranslationEngine()
    assert engine._model is None
    assert engine.translate("こんにちは", "ja", "vi") == "Xin chào"
    assert engine._tokenizer.src_lang == "jpn_Jpan"


def test_nllb_validates_language_pair_before_loading() -> None:
    engine = NllbTranslationEngine()
    with pytest.raises(ValueError):
        engine.translate("hello", "xx", "vi")
    assert engine._model is None
