"""Backend settings, read from the environment with sane defaults."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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

    #: The pooled laptop+restaurant model: it beats each specialist on that
    #: specialist's own gold test set, so one artifact covers both domains
    #: and the UI reports "Laptop and restaurant reviews". See the README's
    #: "Two domains, one model".
    ARTIFACT_DIR: str = "training/artifacts/absa-joint-v1"
    DEVICE: str = "cpu"

    API_PREFIX: str = "/api/v1"
    TITLE: str = "ABSA API"
    VERSION: str = "0.1.0"

    #: Browser origins allowed to call the API: the Vite and CRA dev servers,
    #: plus the deployed frontend.
    #:
    #: The Vercel origin is a default rather than something the operator must
    #: remember to set, because the deployed page is served over HTTPS and
    #: talks to this service on localhost -- browsers permit that particular
    #: mixed-content case, so `uvicorn app.main:app` with no environment at
    #: all is enough to drive the live site from a laptop. Override the whole
    #: tuple from the environment to allow a different frontend.
    #:
    #: ``NoDecode`` so the raw environment string reaches the validator below:
    #: pydantic-settings JSON-decodes a complex field *before* validation, and
    #: would reject a comma-separated value without ever calling it.
    CORS_ORIGINS: Annotated[tuple[str, ...], NoDecode] = (
        "http://localhost:5173",
        "http://localhost:3000",
        "https://absa-sentiment.vercel.app",
    )

    #: Refuse a /predict whose aspect does not occur in the review text.
    #: The model has no "not mentioned" class and cannot abstain, so an absent
    #: aspect gets a confident polarity that means nothing. Turn this off only
    #: to reproduce that behaviour deliberately -- the off-distribution figures
    #: quoted in the READMEs were measured with it off.
    REQUIRE_ASPECT_IN_TEXT: bool = True

    #: When false, a failure to load the model at startup is fatal. The default
    #: lets the service boot and report "degraded" on /health instead, so an
    #: orchestrator can surface the problem rather than crash-looping.
    ALLOW_DEGRADED_START: bool = True

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Parse the raw ``CORS_ORIGINS`` environment string.

        Both forms are accepted::

            CORS_ORIGINS=https://absa.vercel.app,http://localhost:5173
            CORS_ORIGINS=["https://absa.vercel.app"]

        The comma form is what someone pasting two URLs into a hosting
        dashboard will actually write; the JSON form is what plain
        pydantic-settings would have required, kept so an existing
        deployment's value kept working. Blank entries are dropped, so a
        trailing comma is harmless.

        Because the field is annotated ``NoDecode``, this receives the
        untouched string and is the only thing that parses it.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            return json.loads(text)
        return tuple(part.strip() for part in text.split(",") if part.strip())

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
