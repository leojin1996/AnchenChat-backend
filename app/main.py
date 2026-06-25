import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from app.agent_graph import run_agent_graph, stream_agent_graph_events
from app.asr.factory import build_audio_transcriber
from app.assistants import SUGGESTED_PROMPTS
from app.auth.allowlist import Allowlist
from app.auth.context import AuthContext, build_auth_context
from app.auth.dependencies import CurrentUser, build_current_user_dependency
from app.auth.errors import AuthConfigError
from app.auth.routes import build_auth_router
from app.auth.sms import SmsSender
from app.config import Settings, get_settings
from app.models import (
    ChatCompleteResponse,
    ChatRequest,
    Citation,
    ErrorDetail,
    SalesAskRequest,
    SalesAskResponse,
    SpeechRequest,
)
from app.openai_client import OpenAIClient, SpeechResult, StreamEvent, UpstreamServiceError
from app.rate_limit import InMemoryRateLimiter
from app.sales_db import SalesDBError
from app.sales_db import ping as sales_ping
from app.sales_tools import answer_sales_question

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    openai_client: object | None = None,
    audio_transcriber: object | None = None,
    auth_context: AuthContext | None = None,
    auth_allowlist: Allowlist | None = None,
    auth_sms_sender: SmsSender | None = None,
) -> FastAPI:
    active_settings = settings or get_settings()
    limiter = InMemoryRateLimiter(limit=active_settings.requests_per_minute, window_seconds=60)
    app = FastAPI(title="Doubao-style AI Chat Backend")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    auth_ctx = auth_context or build_auth_context(
        settings=active_settings,
        allowlist=auth_allowlist,
        sms_sender=auth_sms_sender,
    )
    app.state.auth_context = auth_ctx
    require_user = build_current_user_dependency(auth_ctx)
    app.include_router(build_auth_router(auth_ctx))

    async def get_openai_client() -> object:
        if openai_client is not None:
            return openai_client
        return OpenAIClient(settings=active_settings)

    async def get_audio_transcriber() -> object:
        if audio_transcriber is not None:
            return audio_transcriber
        if active_settings.asr_provider.strip().lower() == "tencent":
            return build_audio_transcriber(active_settings)
        return await get_openai_client()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat/stream")
    async def chat_stream(
        request: ChatRequest,
        client: object = Depends(get_openai_client),
        user: CurrentUser = Depends(require_user),
    ) -> StreamingResponse:
        ensure_allowed(limiter, _rate_key(user, request.device_id))
        return StreamingResponse(
            stream_chat_events(request, client, format_sse, active_settings, user.phone),
            media_type="text/event-stream",
        )

    @app.post("/chat/stream/chunks")
    async def chat_stream_chunks(
        request: ChatRequest,
        client: object = Depends(get_openai_client),
        user: CurrentUser = Depends(require_user),
    ) -> StreamingResponse:
        ensure_allowed(limiter, _rate_key(user, request.device_id))
        return StreamingResponse(
            stream_chat_events(request, client, format_ndjson, active_settings, user.phone),
            media_type="application/x-ndjson",
        )

    @app.post("/chat/complete")
    async def chat_complete(
        request: ChatRequest,
        client: object = Depends(get_openai_client),
        user: CurrentUser = Depends(require_user),
    ) -> ChatCompleteResponse:
        ensure_allowed(limiter, _rate_key(user, request.device_id))
        return await collect_chat_response(request, client, active_settings, user.phone)

    @app.post("/audio/transcribe")
    async def transcribe_audio(
        device_id: Annotated[str, Form()],
        file: Annotated[UploadFile, File()],
        transcriber: object = Depends(get_audio_transcriber),
        user: CurrentUser = Depends(require_user),
    ) -> dict[str, str]:
        ensure_allowed(limiter, _rate_key(user, device_id))
        content = await file.read()
        logger.info(
            "Audio upload received: filename=%s content_type=%s bytes=%s head=%s",
            file.filename,
            file.content_type,
            len(content),
            content[:12].hex(),
        )
        if not content:
            raise coded_http_error(
                status_code=400,
                code="audio_empty",
                message="录音文件为空，请重新录制。",
            )
        if len(content) > active_settings.max_audio_bytes:
            raise coded_http_error(
                status_code=413,
                code="audio_too_large",
                message="Audio file exceeds the configured upload limit.",
            )
        try:
            text = await transcriber.transcribe_audio(
                content=content,
                filename=file.filename or "audio.mp3",
                content_type=file.content_type or "application/octet-stream",
            )
        except UpstreamServiceError as exc:
            if exc.code in {"audio_too_large"}:
                status_code = 413
            elif exc.code in {"audio_empty", "audio_too_short", "audio_not_decodable"}:
                status_code = 400
            else:
                status_code = 502
            raise coded_http_error(status_code, exc.code, exc.message) from exc
        return {"text": text}

    @app.post("/audio/speech")
    async def create_speech(
        request: SpeechRequest,
        client: object = Depends(get_openai_client),
        user: CurrentUser = Depends(require_user),
    ) -> Response:
        ensure_allowed(limiter, _rate_key(user, request.device_id))
        try:
            result: SpeechResult = await client.create_speech(request.text)
        except UpstreamServiceError as exc:
            raise coded_http_error(502, exc.code, exc.message) from exc
        return Response(content=result.content, media_type=result.media_type)

    @app.get("/assistants/prompts")
    async def list_suggested_prompts(
        user: CurrentUser = Depends(require_user),
    ) -> dict[str, list[str]]:
        return dict(SUGGESTED_PROMPTS)

    @app.get("/sales/health")
    async def sales_health(
        user: CurrentUser = Depends(require_user),
    ) -> dict[str, object]:
        try:
            return await sales_ping(active_settings)
        except SalesDBError as exc:
            raise coded_http_error(503, exc.code, exc.message) from exc

    @app.post("/sales/ask")
    async def sales_ask(
        request: SalesAskRequest,
        user: CurrentUser = Depends(require_user),
    ) -> SalesAskResponse:
        ensure_allowed(limiter, _rate_key(user, request.device_id))
        answer = await answer_sales_question(request.question, settings=active_settings)
        return SalesAskResponse(**answer.to_dict())

    return app


def _rate_key(user: CurrentUser, device_id: str) -> str:
    return f"{user.phone}:{device_id}" if user.phone != "anonymous" else device_id


def ensure_allowed(limiter: InMemoryRateLimiter, device_id: str) -> None:
    if not limiter.allow(device_id):
        raise coded_http_error(
            status_code=429,
            code="rate_limited",
            message="Too many requests from this device. Please wait and try again.",
        )


def coded_http_error(status_code: int, code: str, message: str) -> HTTPException:
    detail = ErrorDetail(code=code, message=message)
    return HTTPException(status_code=status_code, detail=detail.model_dump())


def build_chat_messages(request: ChatRequest) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in request.messages]


async def stream_chat_events(
    request: ChatRequest,
    client: object,
    formatter: callable,
    settings: Settings | None = None,
    user_phone: str = "anonymous",
) -> AsyncIterator[str]:
    try:
        async for event in stream_agent_graph_events(
            request,
            client,
            settings=settings,
            user_phone=user_phone,
        ):
            yield formatter(event)
    except Exception as exc:  # pragma: no cover - exercised in real OpenAI failures.
        yield formatter(StreamEvent(type="error", text=str(exc)))


async def collect_chat_response(
    request: ChatRequest,
    client: object,
    settings: Settings | None = None,
    user_phone: str = "anonymous",
) -> ChatCompleteResponse:
    try:
        result = await run_agent_graph(request, client, settings=settings, user_phone=user_phone)
    except UpstreamServiceError as exc:
        raise coded_http_error(502, exc.code, exc.message) from exc
    except Exception as exc:  # pragma: no cover - exercised in real OpenAI failures.
        raise coded_http_error(
            status_code=502,
            code="upstream_chat_failed",
            message=str(exc),
        ) from exc

    return ChatCompleteResponse(
        text=result.text,
        citations=[Citation(**citation) for citation in result.citations],
        used_search=result.used_search,
        route=result.route,
        intent=result.intent,
    )


def format_sse(event: StreamEvent) -> str:
    payload = {"type": event.type}
    if event.text is not None:
        payload["text"] = event.text
    if event.citation is not None:
        payload["citation"] = event.citation
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def format_ndjson(event: StreamEvent) -> str:
    payload = {"type": event.type}
    if event.text is not None:
        payload["text"] = event.text
    if event.citation is not None:
        payload["citation"] = event.citation
    return f"{json.dumps(payload, ensure_ascii=False)}\n"


def _bootstrap_default_app() -> FastAPI:
    try:
        return create_app()
    except AuthConfigError as exc:
        reason = str(exc)
        logger.error(
            "Auth subsystem not configured (%s). The app will only expose /health "
            "until configuration is fixed.",
            reason,
        )
        fallback = FastAPI(title="Doubao-style AI Chat Backend (auth misconfigured)")

        @fallback.get("/health")
        async def _health() -> dict[str, str]:
            return {"status": "auth_misconfigured", "message": reason}

        @fallback.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def _blocked(full_path: str) -> Response:
            raise HTTPException(
                status_code=503,
                detail={"code": "auth_config_error", "message": reason},
            )

        return fallback


app = _bootstrap_default_app()
