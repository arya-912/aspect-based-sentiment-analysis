"""HTTP routes."""

from .sentiment import router as sentiment_router

__all__ = ["sentiment_router"]
