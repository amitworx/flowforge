import pytest
from pydantic import ValidationError

from flowforge.config import AppSettings


def test_app_settings_dev_defaults():
    """Development settings allow insecure default secret key when DEBUG=True."""
    settings = AppSettings(
        ENVIRONMENT="development",
        DEBUG=True,
        SECRET_KEY="insecure-dev-secret-key-change-in-production-only",
    )
    assert settings.DEBUG is True
    assert settings.ENVIRONMENT == "development"


def test_app_settings_non_dev_rejects_insecure_default_secret_key():
    """Non-development environments (e.g. production) must reject the sample/insecure secret key."""
    with pytest.raises(ValidationError) as excinfo:
        AppSettings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="insecure-dev-secret-key-change-in-production-only",
        )
    assert "Insecure default SECRET_KEY is not permitted in production" in str(excinfo.value)


def test_app_settings_non_dev_rejects_debug_true():
    """Non-development environments must not allow DEBUG=True."""
    with pytest.raises(ValidationError) as excinfo:
        AppSettings(
            ENVIRONMENT="production",
            DEBUG=True,
            SECRET_KEY="a-sufficiently-long-and-secure-custom-secret-key-12345",
        )
    assert "DEBUG cannot be True in production" in str(excinfo.value)


def test_app_settings_non_dev_rejects_default_database_password():
    """Non-development environments must reject default sample database password."""
    with pytest.raises(ValidationError) as excinfo:
        AppSettings(
            ENVIRONMENT="production",
            DEBUG=False,
            SECRET_KEY="a-sufficiently-long-and-secure-custom-secret-key-12345",
            DB_PASSWORD="flowforge",
        )
    assert "Default DB_PASSWORD 'flowforge' is not permitted in production" in str(excinfo.value)


def test_app_settings_non_dev_valid_configuration():
    """Non-development environment passes with explicit secure settings."""
    settings = AppSettings(
        ENVIRONMENT="production",
        DEBUG=False,
        SECRET_KEY="a-sufficiently-long-and-secure-custom-secret-key-12345",
        DB_PASSWORD="secure-prod-db-password-98765",
        ALLOWED_HOSTS="flowforge.example.com",
    )
    assert settings.ENVIRONMENT == "production"
    assert settings.DEBUG is False
    assert settings.SECRET_KEY == "a-sufficiently-long-and-secure-custom-secret-key-12345"
    assert settings.ALLOWED_HOSTS == ["flowforge.example.com"]
