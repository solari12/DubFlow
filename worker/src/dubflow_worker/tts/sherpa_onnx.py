from __future__ import annotations

import importlib.util
import os
import sys
import time
import wave
from pathlib import Path

from dubflow_worker.models.tts import TTSResult
from dubflow_worker.tts.base import TTSEngine


_DLL_DIRECTORY_HANDLES: list[object] = []


def _prepare_windows_dll_search() -> None:
    """Prefer this venv's ONNX Runtime over incompatible system DLLs on Windows."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    for package, subdirs in (
        ("onnxruntime", ("capi", "")),
        ("sherpa_onnx", ("lib", "")),
    ):
        spec = importlib.util.find_spec(package)
        if not spec or not spec.origin:
            continue
        package_dir = Path(spec.origin).parent
        for subdir in subdirs:
            dll_dir = package_dir / subdir
            if dll_dir.is_dir():
                # Keep the returned handles alive: Windows removes a directory when closed.
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir.resolve())))


class SherpaOnnxPiperEngine(TTSEngine):
    """CPU Sherpa-ONNX adapter for the Vietnamese Piper VIVOS voice."""

    name = "sherpa-onnx-piper-vivos-x_low"
    device = "cpu"
    model_name = "vi_VN-vivos-x_low"

    def __init__(self, model_dir: Path | None = None, *, num_threads: int = 2) -> None:
        self.model_dir = model_dir or (
            Path(__file__).resolve().parents[3]
            / ".model-cache"
            / "tts"
            / "vits-piper-vi_VN-vivos-x_low"
        )
        model_path = self.model_dir / "vi_VN-vivos-x_low.onnx"
        token_path = self.model_dir / "tokens.txt"
        data_dir = self.model_dir / "espeak-ng-data"
        for path in (model_path, token_path, data_dir):
            if not path.exists():
                raise FileNotFoundError(
                    f"TTS model files are missing under {self.model_dir}. "
                    "Download vits-piper-vi_VN-vais1000-medium and extract it there."
                )

        _prepare_windows_dll_search()
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "Sherpa-ONNX is not installed. Install the worker 'tts' extra."
            ) from exc

        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(model_path),
            tokens=str(token_path),
            data_dir=str(data_dir),
        )
        model = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits,
            num_threads=num_threads,
            provider="cpu",
        )
        config = sherpa_onnx.OfflineTtsConfig(model=model, max_num_sentences=1)
        if not config.validate():
            raise ValueError(f"Invalid Sherpa-ONNX TTS configuration for {self.model_dir}")

        self._sherpa_onnx = sherpa_onnx
        self._tts = sherpa_onnx.OfflineTts(config)

    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Path,
        speaker: str | None = None,
    ) -> TTSResult:
        if language.strip().lower().replace("-", "_") not in {"vi", "vi_vn"}:
            raise ValueError(f"The installed Piper voice supports Vietnamese, not {language!r}")
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        audio = self._tts.generate(text=text, sid=0, speed=1.0)
        wrote = self._sherpa_onnx.write_wave(
            str(output_path), audio.samples, audio.sample_rate
        )
        runtime = time.perf_counter() - started
        if not wrote or not output_path.is_file() or output_path.stat().st_size <= 44:
            raise RuntimeError(f"Sherpa-ONNX did not create a valid WAV file at {output_path}")

        # Read the artifact's metadata rather than deriving duration from the input text.
        with wave.open(str(output_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        if sample_rate <= 0 or frame_count <= 0:
            raise RuntimeError(f"Generated WAV has invalid metadata: {output_path}")
        duration = frame_count / sample_rate
        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=sample_rate,
            engine=self.name,
            language=language,
            speaker=speaker,
            runtime=runtime,
        )
