from app.config import Settings


def test_settings_use_secure_defaults() -> None:
    settings = Settings(openai_api_key="test-key")

    assert settings.openai_chat_model == "gpt-5.5"
    assert settings.openai_transcribe_model == "whisper-1"
    assert settings.openai_transcribe_language == "zh"
    assert settings.openai_tts_model == "gpt-4o-mini-tts"
    assert settings.openai_tts_voice == "alloy"
    assert settings.max_audio_bytes == 15 * 1024 * 1024
