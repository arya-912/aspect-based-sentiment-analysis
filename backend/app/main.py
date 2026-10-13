"""FastAPI application factory and ASGI entry point.

Run locally with::

    cd backend && uvicorn app.main:app --reload --port 8000

Then: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import Settings, get_settings
from .core.errors import ModelNotReadyError, register_error_handlers
from .routes.sentiment import router as sentiment_router
from .services.model_registry import registry

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, before the first request is served."""
    settings: Settings = get_settings()
    configure_logging()
    LOGGER.info("loading artifact from %s on %s", settings.artifact_path, settings.DEVICE)

    try:
        registry.load(settings.artifact_path, settings.DEVICE)
    except ModelNotReadyError:
        if not settings.ALLOW_DEGRADED_START:
            raise
        # Boot anyway: /health then reports "degraded" and /predict answers
        # 503, which is far easier to diagnose than a crash-looping container.
        LOGGER.error(
            "starting without a model; /health will report degraded. Train one with "
            "`cd training && python train.py train`"
        )

    yield

    LOGGER.info("shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. A factory keeps tests from sharing global state."""
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.TITLE,
        version=settings.VERSION,
        description=(
            "Aspect-based sentiment analysis over laptop reviews. Send a "
            "sentence and an aspect term inside it; get the sentiment "
            "expressed about that aspect."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)
    app.include_router(sentiment_router, prefix=settings.API_PREFIX)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Point a browser hitting the bare host at something useful."""
        return {
            "service": settings.TITLE,
            "version": settings.VERSION,
            "docs": "/docs",
            "health": f"{settings.API_PREFIX}/health",
        }

    return app


app = create_app()
