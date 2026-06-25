from __future__ import annotations

import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from app.auth.errors import AuthError
from app.config import Settings


@dataclass(frozen=True)
class WechatBindSession:
    id: str
    openid: str
    expires_at: float


class WechatCodeSessionClient(Protocol):
    async def code_to_openid(self, code: str) -> str: ...


class WechatSessionClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._http_client = http_client

    async def code_to_openid(self, code: str) -> str:
        if not self._settings.wechat_app_id.strip() or not self._settings.wechat_app_secret.strip():
            raise AuthError("auth_wechat_not_configured", "微信登录服务未配置。")

        params = {
            "appid": self._settings.wechat_app_id,
            "secret": self._settings.wechat_app_secret,
            "js_code": code,
            "grant_type": "authorization_code",
        }
        try:
            if self._http_client is not None:
                response = await self._http_client.get(
                    self._settings.wechat_code2session_url,
                    params=params,
                    timeout=10,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        self._settings.wechat_code2session_url,
                        params=params,
                        timeout=10,
                    )
        except httpx.HTTPError as exc:
            raise AuthError("auth_wechat_request_failed", "微信登录请求失败，请稍后重试。") from exc

        if response.status_code >= 500 or response.status_code == 429:
            raise AuthError("auth_wechat_request_failed", "微信登录服务暂不可用，请稍后重试。")

        try:
            payload = response.json()
        except ValueError as exc:
            raise AuthError("auth_wechat_invalid_response", "微信登录返回内容无法解析。") from exc

        errcode = payload.get("errcode")
        if errcode == 40029:
            raise AuthError("auth_wechat_invalid_code", "微信登录凭证无效，请重试。")
        if response.status_code >= 400 or errcode:
            raise AuthError("auth_wechat_request_failed", "微信登录服务暂不可用，请稍后重试。")

        openid = str(payload.get("openid") or "")
        if not openid:
            raise AuthError("auth_wechat_invalid_response", "微信登录未返回 openid。")
        return openid


class WechatBindingStore:
    def __init__(self, path: Path | None = None, initial: dict[str, str] | None = None) -> None:
        self._path = path
        self._bindings = dict(initial or {})
        if path is not None and path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8") or "{}")
            except (OSError, json.JSONDecodeError) as exc:
                raise AuthError(
                    "auth_wechat_binding_store_invalid",
                    "微信绑定存储文件无法读取或解析。",
                ) from exc
            if not isinstance(raw, dict):
                raise AuthError(
                    "auth_wechat_binding_store_invalid",
                    "微信绑定存储文件格式不正确。",
                )
            self._bindings.update({str(openid): str(phone) for openid, phone in raw.items()})

    @classmethod
    def in_memory(cls, initial: dict[str, str] | None = None) -> WechatBindingStore:
        return cls(path=None, initial=initial)

    def get_phone(self, openid: str) -> str | None:
        return self._bindings.get(openid)

    def bind(self, openid: str, phone: str) -> None:
        previous = self._bindings.get(openid)
        self._bindings[openid] = phone
        try:
            self._persist()
        except AuthError:
            if previous is None:
                self._bindings.pop(openid, None)
            else:
                self._bindings[openid] = previous
            raise

    def _persist(self) -> None:
        if self._path is None:
            return

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._bindings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except OSError as exc:
            raise AuthError(
                "auth_wechat_binding_store_failed",
                "微信绑定关系保存失败，请稍后重试。",
            ) from exc


class WechatBindSessionStore:
    def __init__(self, ttl_seconds: int, now: Callable[[], float] | None = None) -> None:
        self._ttl_seconds = ttl_seconds
        self._now = now or time.time
        self._sessions: dict[str, WechatBindSession] = {}

    def create(self, openid: str) -> WechatBindSession:
        session = WechatBindSession(
            id=secrets.token_urlsafe(24),
            openid=openid,
            expires_at=self._now() + self._ttl_seconds,
        )
        self._sessions[session.id] = session
        return session

    def pop(self, session_id: str) -> WechatBindSession:
        session = self.get(session_id)
        self._sessions.pop(session_id, None)
        return session

    def get(self, session_id: str) -> WechatBindSession:
        session = self._sessions.get(session_id)
        if session is None or session.expires_at <= self._now():
            self._sessions.pop(session_id, None)
            raise AuthError("auth_wechat_bind_expired", "微信绑定会话已过期，请重新登录。")
        return session

    def consume(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class WechatAuthService:
    def __init__(
        self,
        settings: Settings,
        binding_store: WechatBindingStore,
        bind_sessions: WechatBindSessionStore,
        session_client: WechatCodeSessionClient,
    ) -> None:
        self.settings = settings
        self.bindings = binding_store
        self.bind_sessions = bind_sessions
        self.session_client = session_client

    async def resolve_openid(self, code: str) -> str:
        if not self.settings.wechat_app_id.strip() or not self.settings.wechat_app_secret.strip():
            raise AuthError("auth_wechat_not_configured", "微信登录服务未配置。")
        return await self.session_client.code_to_openid(code)


def build_wechat_auth_service(settings: Settings) -> WechatAuthService:
    return WechatAuthService(
        settings=settings,
        binding_store=WechatBindingStore(Path(settings.auth_wechat_bindings_path)),
        bind_sessions=WechatBindSessionStore(settings.auth_wechat_bind_ttl_seconds),
        session_client=WechatSessionClient(settings),
    )
