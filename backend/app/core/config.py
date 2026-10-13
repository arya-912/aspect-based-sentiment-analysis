"""Backend settings, read from the environment with sane defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root: this file is <root>/backend/app/core/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be overridden by an environment variable of the same name
    (or a ``.env`` file), which is how the container and the test suite point
    the service at different artifacts.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ARTIFACT_DIR: str = "training/artifacts/absa-v1"
    DEVICE: str = "cpu"

    API_PREFIX: str = "/api/v1"
    TITLE: str = "ABSA API"
    VERSION: str = "0.1.0"

    #: Browser origins allowed to call the API (the Vite and CRA dev servers).
    CORS_ORIGINS: tuple[str, ...] = (
        "http://localhost:5173",
        "http://localhost:3000",
    )

    #: When false, a failure to load the model at startup is fatal. The default
    #: lets the service boot and report "degraded" on /health instead, so an
    #: orchestrator can surface the problem rather than crash-looping.
    ALLOW_DEGRADED_START: bool = True

    @property
    def artifact_path(self) -> Path:
        """``ARTIFACT_DIR`` resolved against the repository root.

        The default is written repo-relative so it works regardless of the
        directory uvicorn is started from.
        """
        path = Path(self.ARTIFACT_DIR)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings, built once."""
    return Settings()
