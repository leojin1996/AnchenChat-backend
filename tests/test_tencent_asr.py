import pytest

from app.asr.factory import build_audio_transcriber
from app.asr.tencent import TencentAsrTranscriber, resolve_voice_format
from app.config import Settings
from app.openai_client import UpstreamServiceError


def test_resolve_voice_format_prefers_content_sniffing_over_filename() -> None:
    mp3 = b"ID3" + b"\x00" * 9
    assert resolve_voice_format(mp3, "voice.m4a", "application/octet-stream") == "mp3"


def test_resolve_voice_format_falls_back_to_filename_extension() -> None:
    assert resolve_voice_format(b"audio", "voice.mp3", "audio/mp4") == "mp3"
    assert resolve_voice_format(b"audio", "voice.m4a", "application/octet-stream") == "m4a"


def test_build_audio_transcriber_uses_tencent_provider() -> None:
    settings = Settings(
        asr_provider="tencent",
        tencent_secret_id="id",
        tencent_secret_key="key",
    )
    transcriber = build_audio_transcriber(settings)
    assert isinstance(transcriber, TencentAsrTranscriber)


@pytest.mark.asyncio
async def test_tencent_transcriber_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        asr_provider="tencent",
        tencent_secret_id="id",
        tencent_secret_key="key",
    )
    transcriber = TencentAsrTranscriber(settings)
    monkeypatch.setattr("app.asr.tencent.ffmpeg_available", lambda: False)

    def fake_recognize(*args: object) -> str:
        return "你好"

    monkeypatch.setattr(transcriber, "_recognize_sync", fake_recognize)
    text = await transcriber.transcribe_audio(b"x" * 1024, "voice.mp3", "audio/mpeg")
    assert text == "你好"


@pytest.mark.asyncio
async def test_tencent_transcriber_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        asr_provider="tencent",
        tencent_secret_id="",
        tencent_secret_key="",
    )
    transcriber = TencentAsrTranscriber(settings)
    monkeypatch.setattr("app.asr.tencent.ffmpeg_available", lambda: False)
    with pytest.raises(UpstreamServiceError) as exc_info:
        await transcriber.transcribe_audio(b"x" * 1024, "voice.mp3", "audio/mpeg")
    assert exc_info.value.code == "tencent_asr_not_configured"
