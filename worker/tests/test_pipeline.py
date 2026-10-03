from contextlib import AbstractContextManager
from pathlib import Path

from dubflow_worker.models.transcript import Transcript, TranscriptSegment
from dubflow_worker.pipeline.transcribe import TranscriptionPipeline


class FakeAudio(AbstractContextManager):
    path = Path("audio.wav")
    duration = 2.0

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeExtractor:
    def validate_input(self, input_path: Path) -> Path:
        return input_path

    def extract(self, input_path: Path) -> FakeAudio:
        return FakeAudio()


class FakeASR:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        self.calls.append((audio_path, language))
        return Transcript(
            language=language or "en",
            duration=1.9,
            segments=[TranscriptSegment(id=0, start=0.2, end=1.2, text="Hello there.")],
        )


def test_pipeline_extracts_then_calls_asr_and_uses_audio_duration(tmp_path: Path) -> None:
    engine = FakeASR()
    pipeline = TranscriptionPipeline(engine, FakeExtractor())  # type: ignore[arg-type]
    progress: list[str] = []

    transcript = pipeline.run(tmp_path / "video.mp4", language="vi", progress=progress.append)

    assert engine.calls == [(Path("audio.wav"), "vi")]
    assert transcript.language == "vi"
    assert transcript.duration == 2.0
    assert transcript.segments[0].text == "Hello there."
    assert progress == ["Extracting audio...", "Running ASR..."]
