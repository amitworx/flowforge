from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Security & Debug
    SECRET_KEY: str = "insecure-dev-secret-key-change-in-production-only"
    DEBUG: bool = True
    ALLOWED_HOSTS: list[str] | str = ["localhost", "127.0.0.1", "[::1]", "web"]

    # Database
    DATABASE_URL: str | None = None
    DB_ENGINE: str = "django.db.backends.postgresql"
    DB_NAME: str = "flowforge"
    DB_USER: str = "flowforge"
    DB_PASSWORD: str = "flowforge"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    # CORS & Security Defaults
    CORS_ALLOWED_ORIGINS: list[str] | str = []

    @field_validator("ALLOWED_HOSTS", "CORS_ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v]
        return []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings_cfg = AppSettings()
