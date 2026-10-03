from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Self


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac"}


class AudioExtractionError(RuntimeError):
    """An input media file could not be converted to ASR audio."""


class FFmpegNotFoundError(AudioExtractionError):
    """FFmpeg is missing."""


@dataclass(slots=True)
class ExtractedAudio:
    path: Path
    duration: float
    _temporary_directory: tempfile.TemporaryDirectory[str]

    def close(self) -> None:
        self._temporary_directory.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class AudioExtractor:
    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self.ffmpeg_path = ffmpeg_path

    @staticmethod
    def validate_input(input_path: Path) -> Path:
        path = Path(input_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if not path.is_file():
            raise AudioExtractionError(f"Input path is not a file: {path}")
        if path.stat().st_size == 0:
            raise AudioExtractionError(f"Input media file is empty: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise AudioExtractionError(
                f"Unsupported media format '{path.suffix or '(no extension)'}'. Supported: {allowed}"
            )
        return path.resolve()

    @staticmethod
    def build_command(ffmpeg: str, input_path: Path, output_path: Path) -> list[str]:
        return [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

    def _resolve_executable(self, value: str) -> str:
        resolved = shutil.which(value)
        if resolved:
            return resolved
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        if value == "ffmpeg":
            try:
                import imageio_ffmpeg

                bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
                if bundled.is_file():
                    return str(bundled.resolve())
            except (ImportError, RuntimeError, OSError):
                pass
        raise FFmpegNotFoundError(
            f"FFmpeg executable '{value}' was not found. Install FFmpeg and add it to PATH, "
            "or install worker dependencies (which include an imageio-ffmpeg binary), "
            "or set DUBFLOW_FFMPEG."
        )

    @staticmethod
    def _wav_duration(path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as audio:
                if audio.getnframes() == 0:
                    raise AudioExtractionError("No audio samples were extracted from the input.")
                return audio.getnframes() / float(audio.getframerate())
        except (wave.Error, OSError) as exc:
            raise AudioExtractionError(f"Extracted WAV is invalid or empty: {exc}") from exc

    def extract(self, input_path: Path) -> ExtractedAudio:
        source = self.validate_input(input_path)
        ffmpeg = self._resolve_executable(self.ffmpeg_path)

        temporary = tempfile.TemporaryDirectory(prefix="dubflow-audio-")
        output_path = Path(temporary.name) / "audio.wav"
        command = self.build_command(ffmpeg, source, output_path)
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as exc:
            temporary.cleanup()
            raise AudioExtractionError(f"Could not start FFmpeg: {exc}") from exc
        if completed.returncode != 0:
            temporary.cleanup()
            detail = completed.stderr.strip()[-1600:]
            if "matches no streams" in detail.lower() or "does not contain any stream" in detail.lower():
                raise AudioExtractionError(f"No audio stream could be extracted from '{source.name}'.")
            raise AudioExtractionError(
                f"FFmpeg failed to extract audio from '{source.name}' (exit {completed.returncode}): "
                f"{detail or 'no diagnostic output'}"
            )
        try:
            duration = self._wav_duration(output_path)
        except Exception:
            temporary.cleanup()
            raise
        return ExtractedAudio(path=output_path, duration=duration, _temporary_directory=temporary)
