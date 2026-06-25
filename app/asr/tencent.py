from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import PurePath

from tencentcloud.asr.v20190614 import asr_client, models
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

from app.asr.audio_normalize import AudioNormalizeError, ffmpeg_available, normalize_audio_for_asr
from app.config import Settings
from app.openai_client import UpstreamServiceError

logger = logging.getLogger(__name__)

TENCENT_ASR_MAX_BYTES = 3 * 1024 * 1024
MIN_AUDIO_BYTES = 1024
SUPPORTED_VOICE_FORMATS = frozenset(
    {"mp3", "m4a", "wav", "pcm", "ogg-opus", "speex", "silk", "aac", "amr"}
)


class TencentAsrTranscriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def transcribe_audio(self, content: bytes, filename: str, content_type: str) -> str:
        if len(content) < MIN_AUDIO_BYTES:
            raise UpstreamServiceError(
                code="audio_too_short",
                message="录音过短或未采集到声音，请重新录制。",
            )
        if len(content) > TENCENT_ASR_MAX_BYTES:
            raise UpstreamServiceError(
                code="audio_too_large",
                message="腾讯云一句话识别要求音频不超过 3MB。",
            )

        secret_id = self.settings.tencent_secret_id.strip()
        secret_key = self.settings.tencent_secret_key.strip()
        if not secret_id or not secret_key:
            raise UpstreamServiceError(
                code="tencent_asr_not_configured",
                message="腾讯云 ASR 未配置 SecretId/SecretKey。",
            )

        payload, voice_format = self._prepare_audio_payload(content, filename, content_type)
        engine = self.settings.tencent_asr_engine.strip() or "16k_zh"
        logger.info(
            "Tencent ASR request prepared: filename=%s content_type=%s bytes=%s voice_format=%s engine=%s",
            filename,
            content_type,
            len(payload),
            voice_format,
            engine,
        )

        try:
            text = await asyncio.to_thread(
                self._recognize_sync,
                secret_id,
                secret_key,
                payload,
                voice_format,
                engine,
            )
        except TencentCloudSDKException as exc:
            logger.warning("Tencent ASR request failed: %s", exc)
            message = exc.get_message()
            if "audio data empty" in message.lower():
                raise UpstreamServiceError(
                    code="audio_not_decodable",
                    message=(
                        "录音无法识别。开发者工具模拟器常录不到有效声音，请改用真机预览；"
                        "并确认已授权麦克风后再试。"
                    ),
                ) from exc
            raise UpstreamServiceError(
                code="tencent_asr_request_failed",
                message=f"腾讯云语音识别失败：{message}",
            ) from exc

        normalized = text.strip()
        if not normalized:
            raise UpstreamServiceError(
                code="tencent_asr_empty_transcript",
                message="腾讯云语音识别结果为空，请重新录制。",
            )
        return normalized

    def _prepare_audio_payload(
        self,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> tuple[bytes, str]:
        if ffmpeg_available():
            try:
                return normalize_audio_for_asr(content, filename)
            except AudioNormalizeError as exc:
                raise UpstreamServiceError(
                    code="audio_not_decodable",
                    message=str(exc),
                ) from exc
        return content, resolve_voice_format(content, filename, content_type)

    def _recognize_sync(
        self,
        secret_id: str,
        secret_key: str,
        content: bytes,
        voice_format: str,
        engine: str,
    ) -> str:
        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = self.settings.tencent_asr_endpoint.strip() or "asr.tencentcloudapi.com"
        http_profile.reqTimeout = int(self.settings.tencent_asr_timeout)
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client = asr_client.AsrClient(cred, "", client_profile)

        request = models.SentenceRecognitionRequest()
        request.EngSerViceType = engine
        request.SourceType = 1
        request.VoiceFormat = voice_format
        request.Data = base64.b64encode(content).decode("ascii")
        request.DataLen = len(content)

        response = client.SentenceRecognition(request)
        return str(response.Result or "")


def sniff_voice_format(content: bytes) -> str | None:
    if len(content) >= 3 and content[:3] == b"ID3":
        return "mp3"
    if len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0:
        return "mp3"
    if len(content) >= 12 and content[4:8] == b"ftyp":
        return "m4a"
    if len(content) >= 4 and content[:4] == b"RIFF":
        return "wav"
    if len(content) >= 4 and content[:4] == b"OggS":
        return "ogg-opus"
    if len(content) >= 4 and content[:4] == b"#!AMR":
        return "amr"
    return None


def resolve_voice_format(content: bytes, filename: str, content_type: str) -> str:
    sniffed = sniff_voice_format(content)
    if sniffed is not None:
        return sniffed

    extension = PurePath(filename).suffix.lower().lstrip(".")
    if extension in SUPPORTED_VOICE_FORMATS:
        return extension

    normalized_type = content_type.split(";", 1)[0].strip().lower()
    content_type_map = {
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "audio/m4a": "m4a",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/aac": "aac",
        "audio/amr": "amr",
        "audio/ogg": "ogg-opus",
    }
    mapped = content_type_map.get(normalized_type)
    if mapped is not None:
        return mapped

    return "mp3"
