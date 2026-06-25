import json

import httpx
import pytest

from app.config import Settings
from app.openai_client import OpenAIClient, SpeechResult, StreamEvent, UpstreamServiceError


def build_settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
    )


@pytest.mark.asyncio
async def test_stream_chat_parses_text_and_web_citations() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.5"
        assert {"type": "web_search_preview"} in payload["tools"]
        body = "\n".join(
            [
                'data: {"type":"response.output_text.delta","delta":"你"}',
                'data: {"type":"response.output_text.delta","delta":"好"}',
                (
                    'data: {"type":"response.output_text.annotation.added",'
                    '"annotation":{"type":"url_citation","url":"https://example.com",'
                    '"title":"Example"}}'
                ),
                'data: {"type":"response.completed"}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    settings = build_settings()
    async with httpx.AsyncClient(transport=transport, base_url=settings.openai_base_url) as client:
        openai = OpenAIClient(settings=settings, http_client=client)
        events = [
            event
            async for event in openai.stream_chat(
                assistant_id="general",
                messages=[{"role": "user", "content": "你好"}],
            )
        ]

    assert events == [
        StreamEvent(type="delta", text="你"),
        StreamEvent(type="delta", text="好"),
        StreamEvent(
            type="citation",
            citation={"url": "https://example.com", "title": "Example"},
        ),
        StreamEvent(type="done"),
    ]


@pytest.mark.asyncio
async def test_stream_chat_retries_without_web_search_when_gateway_rejects_tool() -> None:
    request_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        request_payloads.append(payload)
        if len(request_payloads) == 1:
            assert payload["tools"] == [{"type": "web_search_preview"}]
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": '{"detail":"Unsupported tool type: web_search_preview"}',
                        "type": "invalid_request_error",
                    }
                },
            )

        assert "tools" not in payload
        body = "\n".join(
            [
                'data: {"type":"response.output_text.delta","delta":"兼"}',
                'data: {"type":"response.output_text.delta","delta":"容"}',
                'data: {"type":"response.completed"}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    settings = build_settings()
    async with httpx.AsyncClient(transport=transport, base_url=settings.openai_base_url) as client:
        openai = OpenAIClient(settings=settings, http_client=client)
        events = [
            event
            async for event in openai.stream_chat(
                assistant_id="general",
                messages=[{"role": "user", "content": "你好"}],
            )
        ]

    assert len(request_payloads) == 2
    assert events == [
        StreamEvent(type="delta", text="兼"),
        StreamEvent(type="delta", text="容"),
        StreamEvent(type="done"),
    ]


@pytest.mark.asyncio
async def test_stream_chat_can_disable_builtin_web_search() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "tools" not in payload
        body = "\n".join(
            [
                'data: {"type":"response.output_text.delta","delta":"摘要"}',
                'data: {"type":"response.completed"}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    settings = build_settings()
    async with httpx.AsyncClient(transport=transport, base_url=settings.openai_base_url) as client:
        openai = OpenAIClient(settings=settings, http_client=client)
        events = [
            event
            async for event in openai.stream_chat(
                assistant_id="search",
                messages=[{"role": "user", "content": "总结来源"}],
                use_builtin_web_search=False,
            )
        ]

    assert events == [
        StreamEvent(type="delta", text="摘要"),
        StreamEvent(type="done"),
    ]


@pytest.mark.asyncio
async def test_transcribe_audio_posts_multipart_file() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        body = request.content
        assert b"whisper-1" in body
        assert b"language" in body
        assert b"zh" in body
        assert b"voice.m4a" in body
        return httpx.Response(200, json={"text": "你好"})

    transport = httpx.MockTransport(handler)
    settings = build_settings()
    async with httpx.AsyncClient(transport=transport, base_url=settings.openai_base_url) as client:
        openai = OpenAIClient(settings=settings, http_client=client)
        text = await openai.transcribe_audio(b"fake-audio", "voice.m4a", "audio/mp4")

    assert text == "你好"


@pytest.mark.asyncio
async def test_create_speech_returns_audio_bytes_and_content_type() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-4o-mini-tts"
        assert payload["voice"] == "alloy"
        assert payload["input"] == "你好"
        return httpx.Response(200, content=b"mp3-bytes", headers={"content-type": "audio/mpeg"})

    transport = httpx.MockTransport(handler)
    settings = build_settings()
    async with httpx.AsyncClient(transport=transport, base_url=settings.openai_base_url) as client:
        openai = OpenAIClient(settings=settings, http_client=client)
        result = await openai.create_speech("你好")

    assert result == SpeechResult(content=b"mp3-bytes", media_type="audio/mpeg")


@pytest.mark.asyncio
async def test_transcribe_audio_rejects_non_json_gateway_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        return httpx.Response(
            200,
            text="<html>gateway</html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)
    settings = build_settings()
    async with httpx.AsyncClient(transport=transport, base_url=settings.openai_base_url) as client:
        openai = OpenAIClient(settings=settings, http_client=client)
        with pytest.raises(UpstreamServiceError) as exc_info:
            await openai.transcribe_audio(b"fake-audio", "voice.m4a", "audio/mp4")

    assert exc_info.value.code == "upstream_transcription_invalid_response"


@pytest.mark.asyncio
async def test_transcribe_audio_falls_back_when_primary_model_fails() -> None:
    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("latin-1")
        if "whisper-1" in body:
            attempts.append("whisper-1")
            return httpx.Response(
                429,
                json={"error": {"message": "upstream saturated"}},
            )
        if "gpt-4o-mini-transcribe" in body:
            attempts.append("gpt-4o-mini-transcribe")
            return httpx.Response(200, json={"text": "备用模型成功"})
        raise AssertionError(f"unexpected request body: {body[:200]}")

    transport = httpx.MockTransport(handler)
    settings = build_settings()
    settings = settings.model_copy(update={"openai_max_retries": 0})
    async with httpx.AsyncClient(transport=transport, base_url=settings.openai_base_url) as client:
        openai = OpenAIClient(settings=settings, http_client=client)
        text = await openai.transcribe_audio(b"fake-audio", "voice.m4a", "audio/mp4")

    assert text == "备用模型成功"
    assert attempts == ["whisper-1", "gpt-4o-mini-transcribe"]


@pytest.mark.asyncio
async def test_create_speech_rejects_non_audio_gateway_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/speech"
        return httpx.Response(
            200,
            text="<html>gateway</html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    transport = httpx.MockTransport(handler)
    settings = build_settings()
    async with httpx.AsyncClient(transport=transport, base_url=settings.openai_base_url) as client:
        openai = OpenAIClient(settings=settings, http_client=client)
        with pytest.raises(UpstreamServiceError) as exc_info:
            await openai.create_speech("你好")

    assert exc_info.value.code == "upstream_speech_invalid_response"
