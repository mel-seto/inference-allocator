from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_")

    gpu_count: int = 4
    queue_max_size: int = 100
    request_timeout_seconds: float = 30.0
    inference_min_ms: int = 100
    inference_max_ms: int = 500


settings = Settings()
