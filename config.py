"""Environment-driven settings for the jev-verify-retry pipeline."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated access to everything the pipeline needs at runtime."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = "ollama"  # Ollama ignores this value but the SDK requires a non-empty string
    openai_base_url: str | None = None

    default_code_model: str = "gpt-4.1"
    default_test_model: str = "gpt-4.1"

    # jev_api_key: str  # no default -- fails loudly at startup if missing, rather than running with an empty key
    # jev_base_url: str = "https://api.typesafe.ai/v1"  # placeholder, confirm the real endpoint
    # max_lint_attempts: int = 3
    # max_execution_attempts: int = 3
    # confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)


settings = Settings()