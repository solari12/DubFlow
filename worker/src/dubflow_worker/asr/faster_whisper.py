from __future__ import annotations

import gc
import logging
from pathlib import Path

from dubflow_worker.asr.base import ASRError, ASREngine
from dubflow_worker.config.settings import Settings
from dubflow_worker.models.transcript import Transcript, TranscriptSegment, WordTimestamp
from dubflow_worker.runtime.cuda import configure_cuda_dll_search


logger = logging.getLogger(__name__)


class FasterWhisperASR(ASREngine):
    name = "faster-whisper"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def _check_device(self) -> None:
        if self.settings.asr_device != "cuda":
            return
        try:
            configure_cuda_dll_search()
            import ctranslate2

            count = ctranslate2.get_cuda_device_count()
        except Exception as exc:
            raise ASRError(
                "CUDA was requested but CTranslate2 could not initialize CUDA. "
                "Check the NVIDIA driver and the CUDA/cuDNN runtime supported by CTranslate2. "
                f"Details: {exc}"
            ) from exc
        if count < 1:
            raise ASRError(
                "CUDA was requested, but CTranslate2 reports no CUDA devices. "
                "Use --device cpu to select CPU execution explicitly."
            )
        logger.info(
            "CTranslate2 detected %s CUDA device(s); ASR configured with device=%s compute_type=%s",
            count,
            self.settings.asr_device,
            self.settings.asr_compute_type,
        )

    def _cuda_runtime_error(self, exc: Exception) -> ASRError | None:
        message = str(exc)
        lower = message.lower()
        if self.settings.asr_device == "cuda" and any(
            token in lower for token in ("cublas", "cudnn", "cudart", "cuda driver")
        ):
            return ASRError(
                "CUDA was requested and an NVIDIA GPU is visible, but CTranslate2 could not "
                f"load a required CUDA runtime library ({message}). Install "
                "`nvidia-cublas-cu12==12.6.4.1` into the Python environment running this worker "
                "and keep the CUDA/cuDNN runtime available."
            )
        return None

    def _load_model(self):
        if self._model is not None:
            return self._model
        self._check_device()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRError(
                "faster-whisper is not installed in this environment. Install worker dependencies "
                "with `uv sync --project worker`."
            ) from exc
        try:
            self._model = WhisperModel(
                self.settings.asr_model,
                device=self.settings.asr_device,
                compute_type=self.settings.asr_compute_type,
            )
            logger.info(
                "Loaded faster-whisper model=%s device=%s compute_type=%s",
                self.settings.asr_model,
                self.settings.asr_device,
                self.settings.asr_compute_type,
            )
            return self._model
        except Exception as exc:
            runtime_error = self._cuda_runtime_error(exc)
            if runtime_error:
                raise runtime_error from exc
            if "out of memory" in str(exc).lower() or "cuda_error_out_of_memory" in str(exc).lower():
                raise ASRError(
                    f"Loading ASR model '{self.settings.asr_model}' exhausted CUDA memory. "
                    "Try a smaller model or --device cpu; no model is guaranteed to fit 4 GB VRAM."
                ) from exc
            raise ASRError(
                f"Could not load faster-whisper model '{self.settings.asr_model}' on "
                f"{self.settings.asr_device} ({self.settings.asr_compute_type}): {exc}"
            ) from exc

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"ASR audio file does not exist: {path}")
        model = self._load_model()
        try:
            segments_iter, info = model.transcribe(
                str(path),
                language=language,
                word_timestamps=True,
                vad_filter=True,
            )
            segments: list[TranscriptSegment] = []
            for index, raw in enumerate(segments_iter):
                text = (raw.text or "").strip()
                if not text:
                    continue
                words: list[WordTimestamp] = []
                for word in getattr(raw, "words", None) or []:
                    word_text = (getattr(word, "word", "") or "").strip()
                    start, end = getattr(word, "start", None), getattr(word, "end", None)
                    if word_text and start is not None and end is not None:
                        words.append(WordTimestamp(word=word_text, start=float(start), end=float(end)))
                segments.append(
                    TranscriptSegment(
                        id=len(segments),
                        start=float(raw.start),
                        end=float(raw.end),
                        text=text,
                        words=words,
                    )
                )
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            return Transcript(
                language=getattr(info, "language", None), duration=duration, segments=segments
            )
        except ASRError:
            raise
        except Exception as exc:
            runtime_error = self._cuda_runtime_error(exc)
            if runtime_error:
                raise runtime_error from exc
            if "out of memory" in str(exc).lower() or "cuda_error_out_of_memory" in str(exc).lower():
                raise ASRError(
                    f"Transcription with model '{self.settings.asr_model}' exhausted CUDA memory. "
                    "Try a smaller model or --device cpu."
                ) from exc
            raise ASRError(f"faster-whisper transcription failed: {exc}") from exc

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
