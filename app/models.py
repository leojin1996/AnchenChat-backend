from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1, max_length=20_000)

    @field_validator("content")
    @classmethod
    def trim_content(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message content cannot be blank")
        return trimmed


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1, max_length=128)
    assistant_id: str = Field(min_length=1, max_length=64)
    messages: list[ChatMessage] = Field(min_length=1, max_length=80)
    voice_mode: bool = False


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=4_000)

    @field_validator("text")
    @classmethod
    def trim_text(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("speech text cannot be blank")
        return trimmed


class Citation(BaseModel):
    url: str
    title: str


class ChatCompleteResponse(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    used_search: bool = False
    route: str | None = None
    intent: dict | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str


class SalesAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("question cannot be blank")
        return trimmed


class SalesAskResponse(BaseModel):
    text: str
    intent: dict | None = None
    rows: list[dict] = Field(default_factory=list)
    error: str | None = None


class SmsSendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phone: str = Field(min_length=1, max_length=32)


class SmsSendResponse(BaseModel):
    cooldown_seconds: int
    expires_in: int


class SmsVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phone: str = Field(min_length=1, max_length=32)
    code: str = Field(min_length=4, max_length=8)


class AuthUserInfo(BaseModel):
    phone: str
    name: str
    role: str


class SmsVerifyResponse(BaseModel):
    token: str
    expires_at: int
    user: AuthUserInfo
