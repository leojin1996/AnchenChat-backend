from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.asr.tencent import TencentAsrTranscriber
from app.config import Settings
from app.openai_client import OpenAIClient


@runtime_checkable
class AudioTranscriber(Protocol):
    async def transcribe_audio(self, content: bytes, filename: str, content_type: str) -> str: ...


def build_audio_transcriber(
    settings: Settings,
    *,
    openai_client: OpenAIClient | None = None,
) -> AudioTranscriber:
    provider = settings.asr_provider.strip().lower()
    if provider == "tencent":
        return TencentAsrTranscriber(settings)
    return openai_client or OpenAIClient(settings=settings)
