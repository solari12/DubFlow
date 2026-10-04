from __future__ import annotations

import json
import re
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from dubflow_worker.audio.extractor import AudioExtractor


class VideoRenderError(RuntimeError):
    """A source could not be rendered as a dubbed MP4."""


@dataclass(slots=True)
class MediaInfo:
    duration: float | None
    video_codec: str | None
    audio_codec: str | None
    has_video: bool
    has_audio: bool
    format_name: str | None = None


@dataclass(slots=True)
class RenderSettings:
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    video_mode: str = "copy"
    expected_audio_sample_rate: int | None = 16000
    expected_audio_channels: int | None = 1
    ffmpeg_path: str = "ffmpeg"

    def __post_init__(self) -> None:
        if self.audio_codec != "aac":
            raise ValueError("MP4 rendering currently supports AAC audio only")
        if not re.fullmatch(r"\d{2,4}k", self.audio_bitrate):
            raise ValueError("Audio bitrate must be a value such as 128k or 192k")
        if self.video_mode not in {"copy", "h264"}:
            raise ValueError("video_mode must be 'copy' or 'h264'")
        if self.expected_audio_sample_rate is not None and self.expected_audio_sample_rate < 1:
            raise ValueError("Expected audio sample rate must be positive")
        if self.expected_audio_channels is not None and self.expected_audio_channels not in {1, 2}:
            raise ValueError("Expected audio channels must be 1 or 2")


@dataclass(slots=True)
class RenderResult:
    input_video: str
    input_audio: str
    output_video: str
    source_video_duration: float | None
    source_audio_duration: float | None
    dubbed_audio_duration: float | None
    output_duration: float | None
    video_codec: str | None
    audio_codec: str | None
    video_stream_copied: bool
    audio_reencoded: bool
    duration_mismatch: float | None
    duration_policy: str
    runtime_seconds: float
    status: str
    error: str | None = None
    output_size_bytes: int | None = None
    output_has_video: bool = False
    output_has_audio: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VideoRenderer(Protocol):
    def render(self, video_path: Path, dubbed_audio_path: Path, output_path: Path) -> RenderResult:
        """Replace the source audio with dubbed audio and write an MP4."""


class FFmpegVideoRenderer:
    def __init__(self, settings: RenderSettings | None = None) -> None:
        self.settings = settings or RenderSettings()
        self._audio_extractor = AudioExtractor(self.settings.ffmpeg_path)

    @staticmethod
    def _wav_info(path: Path, settings: RenderSettings) -> tuple[float, int, int]:
        try:
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                compression = wav_file.getcomptype()
        except (wave.Error, OSError) as exc:
            raise VideoRenderError(f"Dubbed audio is not a readable WAV: {exc}") from exc
        if frames <= 0 or sample_rate <= 0 or sample_width not in {1, 2, 3, 4} or compression != "NONE":
            raise VideoRenderError("Dubbed WAV has invalid or empty audio data")
        if settings.expected_audio_sample_rate and sample_rate != settings.expected_audio_sample_rate:
            raise VideoRenderError(
                f"Dubbed WAV sample rate is {sample_rate} Hz; expected "
                f"{settings.expected_audio_sample_rate} Hz"
            )
        if settings.expected_audio_channels and channels != settings.expected_audio_channels:
            raise VideoRenderError(
                f"Dubbed WAV has {channels} channels; expected {settings.expected_audio_channels}"
            )
        return frames / sample_rate, sample_rate, channels

    def _resolve_ffmpeg(self) -> str:
        return self._audio_extractor._resolve_executable(self.settings.ffmpeg_path)

    @staticmethod
    def _probe(ffmpeg: str, path: Path) -> MediaInfo:
        # FFmpeg prints container and stream metadata while opening an input, even
        # without an output. This also works with imageio-ffmpeg's bundled binary.
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        diagnostic = result.stderr or ""
        if not diagnostic or "Invalid data found" in diagnostic or "No such file" in diagnostic:
            raise VideoRenderError(f"FFmpeg could not inspect media '{path}': {diagnostic[-1200:]}")
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", diagnostic)
        duration = None
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        streams = re.findall(
            r"Stream #\d+:\d+(?:\[[^\]]+\])?(?:\([^)]*\))?:\s*(Video|Audio):\s*([^,\s(]+)",
            diagnostic,
        )
        video_codecs = [codec for kind, codec in streams if kind == "Video" and codec.lower() != "attached" ]
        audio_codecs = [codec for kind, codec in streams if kind == "Audio"]
        format_match = re.search(r"Input #\d+,\s*([^,\s]+)", diagnostic)
        if result.returncode == 1 and "At least one output file must be specified" not in diagnostic:
            # Some FFmpeg builds return a different message for metadata-only open.
            if not streams:
                raise VideoRenderError(f"FFmpeg could not inspect media '{path}': {diagnostic[-1200:]}")
        return MediaInfo(
            duration=duration,
            video_codec=video_codecs[0] if video_codecs else None,
            audio_codec=audio_codecs[0] if audio_codecs else None,
            has_video=bool(video_codecs),
            has_audio=bool(audio_codecs),
            format_name=format_match.group(1) if format_match else None,
        )

    @staticmethod
    def _decode_validate(ffmpeg: str, output_path: Path) -> None:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-v",
                "error",
                "-i",
                str(output_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise VideoRenderError(f"Rendered MP4 failed stream decode: {result.stderr[-1600:]}")

    def _command(
        self,
        ffmpeg: str,
        video: Path,
        audio: Path,
        output: Path,
        duration: float,
        *,
        extend_video: bool = False,
        extension_duration: float = 0.0,
    ) -> list[str]:
        if self.settings.video_mode == "copy" and not extend_video:
            video_options = ["-c:v", "copy"]
        else:
            video_options = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        video_filter = (
            ["-vf", f"tpad=stop_mode=clone:stop_duration={extension_duration:.9f}"]
            if extend_video
            else []
        )
        return [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            *video_filter,
            *video_options,
            "-c:a",
            self.settings.audio_codec,
            "-b:a",
            self.settings.audio_bitrate,
            "-af",
            "apad",
            "-t",
            f"{duration:.9f}",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(output),
        ]

    def render(self, video_path: Path, dubbed_audio_path: Path, output_path: Path) -> RenderResult:
        video_path = Path(video_path).expanduser()
        dubbed_audio_path = Path(dubbed_audio_path).expanduser()
        output_path = Path(output_path).expanduser()
        started = time.perf_counter()
        result = RenderResult(
            input_video=str(video_path),
            input_audio=str(dubbed_audio_path),
            output_video=str(output_path),
            source_video_duration=None,
            source_audio_duration=None,
            dubbed_audio_duration=None,
            output_duration=None,
            video_codec=None,
            audio_codec=None,
            video_stream_copied=self.settings.video_mode == "copy",
            audio_reencoded=True,
            duration_mismatch=None,
            duration_policy="preserve_speech_audio; extend_video_with_final_frame_when_needed",
            runtime_seconds=0,
            status="failed",
        )
        temporary_output = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
        try:
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise VideoRenderError(f"Input video does not exist or is empty: {video_path}")
            if not dubbed_audio_path.is_file() or dubbed_audio_path.stat().st_size == 0:
                raise VideoRenderError(f"Dubbed audio does not exist or is empty: {dubbed_audio_path}")
            if output_path.suffix.lower() != ".mp4":
                raise VideoRenderError("Output path must use the .mp4 extension")
            audio_duration, _, _ = self._wav_info(dubbed_audio_path, self.settings)
            result.dubbed_audio_duration = audio_duration
            ffmpeg = self._resolve_ffmpeg()
            source_info = self._probe(ffmpeg, video_path)
            if not source_info.has_video:
                raise VideoRenderError("Input media has no video stream")
            if source_info.duration is None or source_info.duration <= 0:
                raise VideoRenderError("Input video duration is missing or invalid")
            result.source_video_duration = source_info.duration
            if source_info.has_audio:
                with self._audio_extractor.extract(video_path) as source_audio:
                    result.source_audio_duration = source_audio.duration
            result.duration_mismatch = round(audio_duration - source_info.duration, 6)
            if audio_duration > source_info.duration + 0.001:
                extension_duration = audio_duration - source_info.duration
                output_duration = audio_duration
                extend_video = True
                result.duration_policy = "extend_video_with_frozen_last_frame_to_preserve_audio"
                result.video_stream_copied = False
            elif audio_duration < source_info.duration - 0.001:
                extension_duration = 0.0
                output_duration = source_info.duration
                extend_video = False
                result.duration_policy = "pad_audio_with_trailing_silence_to_video_duration"
            else:
                extension_duration = 0.0
                output_duration = source_info.duration
                extend_video = False
                result.duration_policy = "audio_matches_video_duration"

            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_output.unlink(missing_ok=True)
            command = self._command(
                ffmpeg,
                video_path.resolve(),
                dubbed_audio_path.resolve(),
                temporary_output,
                output_duration,
                extend_video=extend_video,
                extension_duration=extension_duration,
            )
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode:
                raise VideoRenderError(
                    f"FFmpeg render failed (exit {completed.returncode}): {completed.stderr[-2000:]}"
                )
            if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
                raise VideoRenderError("FFmpeg completed without creating a non-empty MP4")
            output_info = self._probe(ffmpeg, temporary_output)
            if not output_info.has_video or not output_info.has_audio:
                raise VideoRenderError("Rendered MP4 must contain both video and dubbed audio streams")
            self._decode_validate(ffmpeg, temporary_output)
            temporary_output.replace(output_path)
            result.status = "success"
            result.output_duration = output_info.duration
            result.video_codec = output_info.video_codec
            result.audio_codec = output_info.audio_codec
            result.output_size_bytes = output_path.stat().st_size
            result.output_has_video = output_info.has_video
            result.output_has_audio = output_info.has_audio
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            temporary_output.unlink(missing_ok=True)
        finally:
            result.runtime_seconds = round(time.perf_counter() - started, 6)
        return result


def write_render_result(path: Path, result: RenderResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
