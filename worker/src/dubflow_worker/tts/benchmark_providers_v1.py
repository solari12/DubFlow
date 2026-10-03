from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

from dubflow_worker.tts.sherpa_onnx import SherpaOnnxPiperEngine


def _ort_device_for_request(device: str) -> str:
    value = device.strip().lower()
    if value == "cpu":
        return "cpu"
    if value in {"cuda", "gpu"}:
        return "gpu"
    raise ValueError(f"Unsupported Korva benchmark device: {device!r}")


def create_cuda_only_session_options(ort):
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options


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

    def __init__(self, voice: str = "khanh_vy", steps: int = 32, *, device: str = "cpu"):
        ort_device = _ort_device_for_request(device)
        from korvatts import TTS

        import onnxruntime as ort

        self.requested_device = "cuda" if ort_device == "gpu" else "cpu"
        if self.requested_device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable in this environment; refusing CPU fallback. "
                    f"Available providers: {ort.get_available_providers()}"
                )
            preload = getattr(ort, "preload_dlls", None)
            if preload:
                preload()
            self.tts = _load_korva_cuda(TTS)
            self.device = "cuda"
        else:
            # Keep the existing CPU benchmark on CPU even after ORT-GPU is installed.
            self.tts = TTS(device="cpu")
            self.device = "cpu"
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
        if self.requested_device == "cuda":
            self.execution_providers = _session_execution_providers(self.tts)
            if not self.execution_providers or any(
                providers[0] != "CUDAExecutionProvider" for providers in self.execution_providers.values()
            ):
                raise RuntimeError(
                    "Korva CUDA session did not select CUDAExecutionProvider for every graph: "
                    f"{self.execution_providers}"
                )

    def synthesize(self, text: str, output_path: Path) -> None:
        wav, _duration = self.tts.synthesize(text, voice=self.voice, lang="vi", total_steps=self.steps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.tts.save_audio(wav, str(output_path))


PROVIDER_REGISTRY = {
    "piper": PiperBenchmarkProvider,
    "korva": KorvaBenchmarkProvider,
}


def _load_korva_cuda(tts_class):
    """Construct Korva with CUDA-only ONNX sessions and CPU EP fallback disabled."""
    import onnxruntime as ort
    import korvatts.tts as korva_tts_module
    from korvatts.assets import resolve_assets_dir
    from korvatts.session import ModelSessions

    class CudaOnlySessions(ModelSessions):
        @classmethod
        def load(cls, onnx_dir, device="auto", num_threads=None):
            if device != "gpu":
                raise RuntimeError(f"CUDA-only benchmark received unexpected device {device!r}")
            options = create_cuda_only_session_options(ort)
            if num_threads:
                options.intra_op_num_threads = num_threads

            def open_session(filename):
                try:
                    return ort.InferenceSession(
                        str(onnx_dir / filename),
                        sess_options=options,
                        providers=["CUDAExecutionProvider"],
                    )
                except Exception as exc:
                    raise RuntimeError(f"CUDA-only ONNX session initialization failed for {filename}: {exc}") from exc

            return cls(
                duration_predictor=open_session("duration_predictor.onnx"),
                text_encoder=open_session("text_encoder.onnx"),
                vector_estimator=open_session("vector_estimator.onnx"),
                vocoder=open_session("vocoder.onnx"),
            )

    original_sessions_class = korva_tts_module.ModelSessions
    korva_tts_module.ModelSessions = CudaOnlySessions
    try:
        try:
            assets = resolve_assets_dir(auto_download=False)
        except FileNotFoundError:
            from huggingface_hub import snapshot_download

            assets = snapshot_download(
                repo_id="dogenthq/KorvaTTS",
                allow_patterns=["onnx/*", "voice_styles/*"],
                local_files_only=True,
            )
        return tts_class(assets_dir=assets, device="gpu", auto_download=False)
    finally:
        korva_tts_module.ModelSessions = original_sessions_class


def _session_execution_providers(tts) -> dict[str, list[str]]:
    sessions = tts.sessions
    return {
        name: list(getattr(sessions, name).get_providers())
        for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
    }
