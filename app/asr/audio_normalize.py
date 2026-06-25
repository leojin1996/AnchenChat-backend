from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePath

logger = logging.getLogger(__name__)

MIN_NORMALIZED_BYTES = 1024
MIN_AUDIO_DURATION_SECONDS = 0.3


class AudioNormalizeError(Exception):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def normalize_audio_for_asr(content: bytes, filename: str) -> tuple[bytes, str]:
    """Convert arbitrary upload bytes to 16 kHz mono WAV for Tencent ASR."""
    if len(content) < MIN_NORMALIZED_BYTES:
        raise AudioNormalizeError("录音过短或未采集到声音，请重新录制。")

    suffix = PurePath(filename or "audio.bin").suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="anchen-asr-") as tmp_dir:
        tmp = Path(tmp_dir)
        input_path = tmp / f"input{suffix}"
        output_path = tmp / "normalized.wav"
        input_path.write_bytes(content)

        duration = _probe_duration_seconds(input_path)
        if duration is not None and duration < MIN_AUDIO_DURATION_SECONDS:
            raise AudioNormalizeError(
                "录音时长过短或未采集到声音。开发者工具模拟器麦克风常不可用，请改用真机预览测试。"
            )

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "ffmpeg normalize failed: filename=%s stderr=%s",
                filename,
                (result.stderr or "").strip()[:300],
            )
            raise AudioNormalizeError(
                "无法解析录音文件。开发者工具模拟器常录不到有效声音，请改用真机预览，并确认已授权麦克风。"
            )

        wav_bytes = output_path.read_bytes()
        if len(wav_bytes) < MIN_NORMALIZED_BYTES:
            raise AudioNormalizeError("录音过短或未采集到声音，请重新录制。")

        normalized_duration = _probe_duration_seconds(output_path)
        if normalized_duration is not None and normalized_duration < MIN_AUDIO_DURATION_SECONDS:
            raise AudioNormalizeError(
                "录音时长过短或未采集到声音。开发者工具模拟器麦克风常不可用，请改用真机预览测试。"
            )

        logger.info(
            "Audio normalized for ASR: filename=%s in_bytes=%s out_bytes=%s duration=%s",
            filename,
            len(content),
            len(wav_bytes),
            normalized_duration,
        )
        return wav_bytes, "wav"


def _probe_duration_seconds(path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    raw = (result.stdout or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
