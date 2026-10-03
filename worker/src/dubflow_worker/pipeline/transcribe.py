from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dubflow_worker.asr.base import ASREngine
from dubflow_worker.audio.extractor import AudioExtractor
from dubflow_worker.models.transcript import Transcript


ProgressCallback = Callable[[str], None]


class TranscriptionPipeline:
    def __init__(self, asr_engine: ASREngine, audio_extractor: AudioExtractor) -> None:
        self.asr_engine = asr_engine
        self.audio_extractor = audio_extractor

    def run(
        self,
        input_path: Path,
        language: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> Transcript:
        source = self.audio_extractor.validate_input(Path(input_path))
        if progress:
            progress("Extracting audio...")
        with self.audio_extractor.extract(source) as audio:
            if progress:
                progress("Running ASR...")
            transcript = self.asr_engine.transcribe(audio.path, language=language)
            # Use the decoded WAV duration as the stable timeline boundary. ASR
            # model-reported duration can differ slightly across engines.
            return Transcript(
                language=transcript.language,
                duration=audio.duration,
                segments=transcript.segments,
            )
