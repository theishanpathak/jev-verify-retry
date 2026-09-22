"""Environment-driven settings for the jev-verify-retry pipeline."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Defaults run everything on local Ollama; .env overrides switch to OpenAI."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

   
    openai_api_key: str = "ollama"
    openai_base_url: str = "http://localhost:11434/v1"

    default_code_model: str = "qwen2.5-coder:7b"
    default_test_model: str = "qwen2.5-coder:7b"
    default_judge_model: str = "qwen2.5-coder:7b"

    jev_api_key: str

    max_lint_attempts: int = 3
    max_semantic_attempts: int = 3
    max_execution_attempts: int = 3


settings = Settings()