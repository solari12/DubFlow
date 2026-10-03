from pathlib import Path

import pytest

from dubflow_worker.audio.extractor import AudioExtractionError, AudioExtractor


def test_missing_input_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        AudioExtractor.validate_input(tmp_path / "missing.mp4")


def test_unsupported_extension_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "audio.ogg"
    source.write_bytes(b"not inspected")
    with pytest.raises(AudioExtractionError, match="Unsupported media format"):
        AudioExtractor.validate_input(source)


def test_empty_media_file_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "empty.wav"
    source.touch()
    with pytest.raises(AudioExtractionError, match="file is empty"):
        AudioExtractor.validate_input(source)


def test_supported_extensions_are_case_insensitive(tmp_path: Path) -> None:
    source = tmp_path / "clip.MP4"
    source.write_bytes(b"placeholder")
    assert AudioExtractor.validate_input(source) == source.resolve()


def test_command_converts_first_audio_stream_to_mono_16khz_pcm(tmp_path: Path) -> None:
    command = AudioExtractor.build_command("ffmpeg.exe", tmp_path / "in.mp4", tmp_path / "out.wav")
    assert command[0] == "ffmpeg.exe"
    assert command[command.index("-map") + 1] == "0:a:0"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"


def test_missing_ffmpeg_is_actionable(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"placeholder")
    extractor = AudioExtractor("definitely-not-ffmpeg")
    with pytest.raises(AudioExtractionError, match="Install FFmpeg"):
        extractor._resolve_executable(extractor.ffmpeg_path)
