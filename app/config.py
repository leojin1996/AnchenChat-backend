from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="OPENAI_BASE_URL",
    )
    openai_chat_model: str = Field(default="gpt-5.5", validation_alias="OPENAI_CHAT_MODEL")
    openai_transcribe_model: str = Field(
        default="whisper-1",
        validation_alias="OPENAI_TRANSCRIBE_MODEL",
    )
    openai_transcribe_language: str = Field(
        default="zh",
        validation_alias="OPENAI_TRANSCRIBE_LANGUAGE",
    )
    openai_tts_model: str = Field(default="gpt-4o-mini-tts", validation_alias="OPENAI_TTS_MODEL")
    openai_tts_voice: str = Field(default="alloy", validation_alias="OPENAI_TTS_VOICE")
    openai_intent_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_INTENT_MODEL",
    )
    openai_intent_timeout: float = Field(
        default=10.0,
        validation_alias="OPENAI_INTENT_TIMEOUT",
    )
    openai_router_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_ROUTER_MODEL",
    )
    openai_router_timeout: float = Field(
        default=10.0,
        validation_alias="OPENAI_ROUTER_TIMEOUT",
    )
    openai_request_timeout: float = Field(
        default=60.0,
        validation_alias="OPENAI_REQUEST_TIMEOUT",
    )
    openai_max_retries: int = Field(
        default=2,
        validation_alias="OPENAI_MAX_RETRIES",
    )
    openai_retry_backoff_seconds: float = Field(
        default=0.8,
        validation_alias="OPENAI_RETRY_BACKOFF_SECONDS",
    )
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    tavily_base_url: str = Field(
        default="https://api.tavily.com",
        validation_alias="TAVILY_BASE_URL",
    )
    tavily_search_depth: str = Field(default="basic", validation_alias="TAVILY_SEARCH_DEPTH")
    tavily_max_results: int = Field(default=5, validation_alias="TAVILY_MAX_RESULTS")
    tavily_timeout: float = Field(default=10.0, validation_alias="TAVILY_TIMEOUT")
    requests_per_minute: int = Field(default=30, validation_alias="REQUESTS_PER_MINUTE")
    max_audio_bytes: int = Field(default=15 * 1024 * 1024, validation_alias="MAX_AUDIO_BYTES")
    backend_cors_origins: str = Field(default="*", validation_alias="BACKEND_CORS_ORIGINS")

    sql_server_host_name: str = Field(default="", validation_alias="SQL_SERVER_HOST_NAME")
    sql_server_user_name: str = Field(default="", validation_alias="SQL_SERVER_USER_NAME")
    sql_server_user_password: str = Field(default="", validation_alias="SQL_SERVER_USER_PASSWORD")
    sql_server_database: str = Field(default="", validation_alias="SQL_SERVER_DATABASE")
    sql_server_schema: str = Field(default="dbo", validation_alias="SQL_SERVER_SCHEMA")
    sql_server_include_freeze: bool = Field(
        default=False, validation_alias="SQL_SERVER_INCLUDE_FREEZE"
    )
    sql_server_query_timeout: int = Field(default=30, validation_alias="SQL_SERVER_QUERY_TIMEOUT")

    auth_enabled: bool = Field(default=True, validation_alias="AUTH_ENABLED")
    auth_jwt_secret: str = Field(default="", validation_alias="AUTH_JWT_SECRET")
    auth_jwt_ttl_seconds: int = Field(
        default=30 * 24 * 3600,
        validation_alias="AUTH_JWT_TTL_SECONDS",
    )
    auth_allowlist_path: str = Field(
        default="auth/allowlist.yaml",
        validation_alias="AUTH_ALLOWLIST_PATH",
    )
    auth_code_ttl_seconds: int = Field(default=300, validation_alias="AUTH_CODE_TTL_SECONDS")
    auth_code_resend_cooldown: int = Field(
        default=60,
        validation_alias="AUTH_CODE_RESEND_COOLDOWN",
    )
    auth_code_max_attempts: int = Field(default=5, validation_alias="AUTH_CODE_MAX_ATTEMPTS")
    auth_sms_hourly_limit: int = Field(default=5, validation_alias="AUTH_SMS_HOURLY_LIMIT")
    auth_dev_bypass_code: str = Field(default="", validation_alias="AUTH_DEV_BYPASS_CODE")

    aliyun_sms_access_key_id: str = Field(default="", validation_alias="ALIYUN_SMS_ACCESS_KEY_ID")
    aliyun_sms_access_key_secret: str = Field(
        default="",
        validation_alias="ALIYUN_SMS_ACCESS_KEY_SECRET",
    )
    aliyun_sms_sign_name: str = Field(default="", validation_alias="ALIYUN_SMS_SIGN_NAME")
    aliyun_sms_template_code: str = Field(default="", validation_alias="ALIYUN_SMS_TEMPLATE_CODE")
    aliyun_sms_endpoint: str = Field(
        default="https://dypnsapi.aliyuncs.com",
        validation_alias="ALIYUN_SMS_ENDPOINT",
    )
    aliyun_sms_scheme_name: str = Field(default="", validation_alias="ALIYUN_SMS_SCHEME_NAME")
    aliyun_sms_country_code: str = Field(default="86", validation_alias="ALIYUN_SMS_COUNTRY_CODE")
    aliyun_sms_code_length: int = Field(default=6, validation_alias="ALIYUN_SMS_CODE_LENGTH")
    aliyun_sms_code_type: int = Field(default=1, validation_alias="ALIYUN_SMS_CODE_TYPE")
    aliyun_sms_case_auth_policy: int = Field(
        default=1,
        validation_alias="ALIYUN_SMS_CASE_AUTH_POLICY",
    )
    aliyun_sms_timeout: float = Field(default=10.0, validation_alias="ALIYUN_SMS_TIMEOUT")

    @property
    def cors_origins(self) -> list[str]:
        if self.backend_cors_origins.strip() == "*":
            return ["*"]
        return [
            origin.strip()
            for origin in self.backend_cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
