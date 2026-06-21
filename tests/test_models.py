import pytest
from pydantic import ValidationError

from app.models import ChatMessage, ChatRequest


def test_chat_request_requires_non_empty_messages() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(device_id="device-1", assistant_id="general", messages=[])


def test_chat_request_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            device_id="device-1",
            assistant_id="general",
            messages=[ChatMessage(role="user", content="   ")],
        )


def test_chat_request_trims_content_and_defaults_voice_mode() -> None:
    request = ChatRequest(
        device_id="device-1",
        assistant_id="general",
        messages=[ChatMessage(role="user", content="  hello  ")],
    )

    assert request.messages[0].content == "hello"
    assert request.voice_mode is False
