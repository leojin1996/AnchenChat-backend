import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

import httpx

from app.assistants import get_assistant_instructions
from app.config import Settings

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class StreamEvent:
    type: str
    text: str | None = None
    citation: dict[str, str] | None = None


@dataclass(frozen=True)
class SpeechResult:
    content: bytes
    media_type: str


class UpstreamServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OpenAIClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._http_client = http_client

    async def stream_chat(
        self,
        assistant_id: str,
        messages: Sequence[Mapping[str, str]],
        use_builtin_web_search: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        async with self._client() as client:
            search_options = (True, False) if use_builtin_web_search else (False,)
            for use_web_search in search_options:
                payload = self._build_chat_payload(
                    assistant_id,
                    messages,
                    use_web_search=use_web_search,
                )
                fell_back_to_no_tools = False
                max_attempts = max(1, self.settings.openai_max_retries + 1)
                for attempt in range(max_attempts):
                    yielded_any = False
                    try:
                        async with client.stream(
                            "POST",
                            "/responses",
                            json=payload,
                            headers=self._headers(),
                            timeout=self.settings.openai_request_timeout,
                        ) as response:
                            if (
                                use_web_search
                                and response.status_code == 400
                                and await self._should_retry_without_web_search(response)
                            ):
                                fell_back_to_no_tools = True
                                break

                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                event = self._parse_stream_line(line)
                                if event is not None:
                                    yielded_any = True
                                    yield event
                            return
                    except httpx.HTTPStatusError as exc:
                        status_code = exc.response.status_code
                        if (
                            not yielded_any
                            and status_code in TRANSIENT_STATUS_CODES
                            and attempt + 1 < max_attempts
                        ):
                            await self._sleep_before_retry(attempt)
                            continue
                        raise UpstreamServiceError(
                            code="upstream_chat_request_failed",
                            message=(
                                f"Upstream chat request failed with status {status_code}. "
                                "Please verify the configured gateway supports the Responses API."
                            ),
                        ) from exc
                    except httpx.HTTPError as exc:
                        if not yielded_any and attempt + 1 < max_attempts:
                            await self._sleep_before_retry(attempt)
                            continue
                        raise UpstreamServiceError(
                            code="upstream_chat_request_failed",
                            message=(
                                "Upstream chat request failed. "
                                "Please check network and gateway settings."
                            ),
                        ) from exc
                if fell_back_to_no_tools:
                    continue

    async def transcribe_audio(self, content: bytes, filename: str, content_type: str) -> str:
        models = self._transcribe_model_candidates()
        last_error: UpstreamServiceError | None = None

        async with self._client() as client:
            for index, model in enumerate(models):
                max_attempts = None
                if index < len(models) - 1:
                    max_attempts = max(1, self.settings.openai_max_retries + 1)
                try:
                    return await self._transcribe_with_model(
                        client,
                        content,
                        filename,
                        content_type,
                        model,
                        max_attempts=max_attempts,
                    )
                except UpstreamServiceError as exc:
                    last_error = exc

        if last_error is not None:
            raise last_error
        raise UpstreamServiceError(
            code="upstream_transcription_invalid_response",
            message="No transcription model configured.",
        )

    def _transcribe_model_candidates(self) -> list[str]:
        candidates: list[str] = []
        for model in (
            self.settings.openai_transcribe_model.strip(),
            self.settings.openai_transcribe_fallback_model.strip(),
        ):
            if model and model not in candidates:
                candidates.append(model)
        return candidates

    async def _transcribe_with_model(
        self,
        client: httpx.AsyncClient,
        content: bytes,
        filename: str,
        content_type: str,
        model: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        files = {
            "file": (filename, content, content_type),
        }
        data = {
            "model": model,
            "response_format": "json",
        }
        language = self.settings.openai_transcribe_language.strip()
        if language:
            data["language"] = language

        response = await self._post_audio_request(
            client,
            "/audio/transcriptions",
            data=data,
            files=files,
            max_attempts=max_attempts,
        )
        content_type_header = response.headers.get("content-type", "")
        if "json" not in content_type_header.lower():
            raise UpstreamServiceError(
                code="upstream_transcription_invalid_response",
                message=(
                    "Upstream audio transcription did not return JSON. "
                    "Please verify OPENAI_BASE_URL supports audio APIs."
                ),
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamServiceError(
                code="upstream_transcription_invalid_response",
                message="Upstream audio transcription returned malformed JSON.",
            ) from exc

        text = str(payload.get("text", "")).strip()
        if not text:
            raise UpstreamServiceError(
                code="upstream_transcription_invalid_response",
                message="Upstream audio transcription returned an empty transcript.",
            )
        return text

    async def create_speech(self, text: str) -> SpeechResult:
        payload = {
            "model": self.settings.openai_tts_model,
            "voice": self.settings.openai_tts_voice,
            "input": text,
        }

        async with self._client() as client:
            response = await self._post_audio_request(client, "/audio/speech", json=payload)
            media_type = response.headers.get("content-type", "audio/mpeg")
            if not media_type.lower().startswith("audio/"):
                raise UpstreamServiceError(
                    code="upstream_speech_invalid_response",
                    message=(
                        "Upstream speech endpoint did not return audio data. "
                        "Please verify OPENAI_BASE_URL supports audio APIs."
                    ),
                )
            if not response.content:
                raise UpstreamServiceError(
                    code="upstream_speech_invalid_response",
                    message="Upstream speech endpoint returned an empty audio payload.",
                )
            return SpeechResult(content=response.content, media_type=media_type)

    async def _sleep_before_retry(self, attempt: int, *, rate_limited: bool = False) -> None:
        base = self.settings.openai_retry_backoff_seconds
        if rate_limited:
            base = max(base, 2.0)
        backoff = base * (2**attempt)
        if backoff > 0:
            await asyncio.sleep(backoff)

    def _client(self) -> "_ClientContext":
        if self._http_client is not None:
            return _ClientContext(self._http_client, should_close=False)
        client = httpx.AsyncClient(base_url=self.settings.openai_base_url)
        return _ClientContext(client, should_close=True)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.openai_api_key}"}

    def _build_chat_payload(
        self,
        assistant_id: str,
        messages: Sequence[Mapping[str, str]],
        use_web_search: bool,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.settings.openai_chat_model,
            "instructions": get_assistant_instructions(assistant_id),
            "input": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
            "stream": True,
        }
        if use_web_search:
            payload["tools"] = [{"type": "web_search_preview"}]
        return payload

    async def _should_retry_without_web_search(self, response: httpx.Response) -> bool:
        await response.aread()
        try:
            payload = response.json()
        except ValueError:
            return False

        error = payload.get("error") or {}
        message = str(error.get("message", ""))
        return "Unsupported tool type: web_search_preview" in message

    async def _post_audio_request(
        self,
        client: httpx.AsyncClient,
        path: str,
        *,
        max_attempts: int | None = None,
        **kwargs: object,
    ) -> httpx.Response:
        if max_attempts is None:
            max_attempts = max(max(1, self.settings.openai_max_retries + 1), 4)
        else:
            max_attempts = max(1, max_attempts)
        for attempt in range(max_attempts):
            try:
                response = await client.post(
                    path,
                    headers=self._headers(),
                    timeout=self.settings.openai_request_timeout,
                    **kwargs,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in TRANSIENT_STATUS_CODES and attempt + 1 < max_attempts:
                    await self._sleep_before_retry(
                        attempt,
                        rate_limited=status_code == 429,
                    )
                    continue
                upstream_message = extract_upstream_error_message(exc.response)
                base_message = (
                    f"Upstream audio request failed with status {status_code}. "
                    "Please verify the configured audio-capable gateway."
                )
                message = (
                    f"{base_message} {upstream_message}"
                    if upstream_message
                    else base_message
                )
                raise UpstreamServiceError(
                    code="upstream_audio_request_failed",
                    message=message,
                ) from exc
            except httpx.HTTPError as exc:
                if attempt + 1 < max_attempts:
                    await self._sleep_before_retry(attempt)
                    continue
                raise UpstreamServiceError(
                    code="upstream_audio_request_failed",
                    message=(
                        "Upstream audio request failed. "
                        "Please check network and gateway settings."
                    ),
                ) from exc
            return response
        raise UpstreamServiceError(
            code="upstream_audio_request_failed",
            message="Upstream audio request failed after retries.",
        )

    def _parse_stream_line(self, line: str) -> StreamEvent | None:
        if not line.startswith("data: "):
            return None

        raw_payload = line.removeprefix("data: ").strip()
        if not raw_payload or raw_payload == "[DONE]":
            return None

        payload = json.loads(raw_payload)
        event_type = payload.get("type")
        if event_type == "response.output_text.delta":
            return StreamEvent(type="delta", text=str(payload.get("delta", "")))
        if event_type == "response.output_text.annotation.added":
            annotation = payload.get("annotation") or {}
            if annotation.get("type") == "url_citation":
                citation = {
                    "url": str(annotation.get("url", "")),
                    "title": str(annotation.get("title", annotation.get("url", ""))),
                }
                return StreamEvent(type="citation", citation=citation)
        if event_type == "response.completed":
            return StreamEvent(type="done")
        if event_type == "response.failed":
            error = payload.get("error") or {}
            message = str(error.get("message", "OpenAI request failed"))
            return StreamEvent(type="error", text=message)
        return None


def extract_upstream_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:200] if text else ""

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    detail = payload.get("detail")
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    if isinstance(detail, str) and detail.strip():
        return detail.strip()

    return ""


class _ClientContext:
    def __init__(self, client: httpx.AsyncClient, should_close: bool) -> None:
        self.client = client
        self.should_close = should_close

    async def __aenter__(self) -> httpx.AsyncClient:
        return self.client

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.should_close:
            await self.client.aclose()
