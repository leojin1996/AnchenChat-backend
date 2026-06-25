import pytest

from app.asr.audio_normalize import AudioNormalizeError, normalize_audio_for_asr


def test_normalize_rejects_too_short_payload() -> None:
    with pytest.raises(AudioNormalizeError, match="录音过短"):
        normalize_audio_for_asr(b"x" * 100, "voice.mp3")


def test_normalize_rejects_invalid_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.asr.audio_normalize._probe_duration_seconds",
        lambda _path: None,
    )

    class Result:
        returncode = 1
        stderr = "Invalid data found when processing input"

    monkeypatch.setattr("app.asr.audio_normalize.subprocess.run", lambda *args, **kwargs: Result())

    with pytest.raises(AudioNormalizeError, match="无法解析录音文件"):
        normalize_audio_for_asr(b"ID3" + b"\x00" * 2045, "voice.mp3")
