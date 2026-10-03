from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

from dubflow_worker.tts.sherpa_onnx import SherpaOnnxPiperEngine


class PiperBenchmarkProvider:
    provider = "Piper / Sherpa-ONNX"
    model = "vi_VN-vivos-x_low"
    language = "vi"
    device = "cpu"

    def __init__(self, model_dir: Path, num_threads: int = 2):
        self.model_dir = Path(model_dir)
        self.model_size_bytes = sum(p.stat().st_size for p in self.model_dir.rglob("*") if p.is_file())
        self.model_size_measurement_method = "Sum of all regular files in the local Piper model directory."
        self.benchmark_configuration = {"num_threads": num_threads, "sid": 0, "speed": 1.0}
        self.engine = SherpaOnnxPiperEngine(self.model_dir, num_threads=num_threads)

    def synthesize(self, text: str, output_path: Path) -> None:
        result = self.engine.synthesize(text, "vi", output_path)
        if not result.audio_path.exists():
            raise RuntimeError("Piper returned without creating its WAV")


class KorvaBenchmarkProvider:
    provider = "KorvaTTS"
    model = "dogenthq/KorvaTTS"
    language = "vi"
    device = "cpu"

    def __init__(self, voice: str = "khanh_vy", steps: int = 32):
        from korvatts import TTS

        self.tts = TTS()
        self.voice = voice
        self.steps = steps
        self.benchmark_configuration = {"voice": voice, "total_steps": steps, "speed": 1.05}
        module = importlib.util.find_spec("korvatts")
        package_dir = Path(module.origin).parent if module and module.origin else None
        # The runtime/model may use the Hugging Face cache; this measures the
        # installed package tree only and is labeled as a lower-bound estimate.
        self.model_size_bytes = (
            sum(p.stat().st_size for p in package_dir.rglob("*") if p.is_file())
            if package_dir and package_dir.exists() else None
        )
        cache_root = Path.home() / ".cache/huggingface/hub/models--dogenthq--KorvaTTS"
        if cache_root.exists():
            cache_size = sum(p.stat().st_size for p in cache_root.rglob("*") if p.is_file())
            self.model_size_bytes = (self.model_size_bytes or 0) + cache_size
        self.model_size_measurement_method = "Installed korvatts package plus local Hugging Face snapshot regular files, when present; hard-link/cache accounting may vary."

    def synthesize(self, text: str, output_path: Path) -> None:
        wav, _duration = self.tts.synthesize(text, voice=self.voice, lang="vi", total_steps=self.steps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tts.save_audio(wav, str(output_path))


PROVIDER_REGISTRY = {
    "piper": PiperBenchmarkProvider,
    "korva": KorvaBenchmarkProvider,
}
