from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_DEV_SECRET_KEY = "insecure-dev-secret-key-change-in-production-only"
DEFAULT_DEV_DB_PASSWORD = "flowforge"


class AppSettings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Development configurations provide convenient defaults for rapid local prototyping.
    Non-development configurations (e.g., 'production', 'staging') enforce strict security:
    they explicitly reject sample/insecure keys, require DEBUG=False, and require non-default credentials.
    """

    # Environment
    ENVIRONMENT: Literal["development", "production", "staging", "test"] = "development"

    # Security & Debug
    SECRET_KEY: str = INSECURE_DEV_SECRET_KEY
    DEBUG: bool = True
    ALLOWED_HOSTS: list[str] | str = ["localhost", "127.0.0.1", "[::1]", "web"]

    # Database
    DATABASE_URL: str | None = None
    DB_ENGINE: str = "django.db.backends.postgresql"
    DB_NAME: str = "flowforge"
    DB_USER: str = "flowforge"
    DB_PASSWORD: str = DEFAULT_DEV_DB_PASSWORD
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    # Celery & Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"

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

    @model_validator(mode="after")
    def validate_environment_security(self) -> "AppSettings":
        if self.ENVIRONMENT in ("production", "staging"):
            if self.SECRET_KEY == INSECURE_DEV_SECRET_KEY:
                raise ValueError(
                    f"Insecure default SECRET_KEY is not permitted in {self.ENVIRONMENT}. "
                    "Set an explicit, cryptographically secure SECRET_KEY in your environment."
                )
            if self.DEBUG:
                raise ValueError(f"DEBUG cannot be True in {self.ENVIRONMENT}. Set DEBUG=False.")
            if self.DB_PASSWORD == DEFAULT_DEV_DB_PASSWORD:
                raise ValueError(
                    f"Default DB_PASSWORD '{DEFAULT_DEV_DB_PASSWORD}' is not permitted in {self.ENVIRONMENT}. "
                    "Configure explicit secure database credentials."
                )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings_cfg = AppSettings()
