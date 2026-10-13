"""Uniform error responses. No traceback or internal detail ever reaches a client."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Starlette's HTTPException, not FastAPI's. FastAPI's subclasses it, so a
# handler registered for this base catches both -- including the 404 and 405
# that the router raises for an unmatched path or method, which a handler
# registered for the subclass alone would miss.
from starlette.exceptions import HTTPException

LOGGER = logging.getLogger(__name__)

#: Starlette renamed this constant (ENTITY -> CONTENT) and deprecated the old
#: name; the number is stable, so resolve whichever the installed version has.
HTTP_422 = getattr(
    status, "HTTP_422_UNPROCESSABLE_CONTENT", None
) or getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)


class ModelNotReadyError(RuntimeError):
    """The model has not finished loading, or failed to load.

    Maps to 503, which tells a caller (and a load balancer) that retrying
    later is reasonable -- unlike a 500, which suggests the request itself is
    hopeless.
    """


def error_response(message: str, code: int) -> JSONResponse:
    """The single response shape every handler returns."""
    return JSONResponse(status_code=code, content={"error": message, "status": code})


def _format_validation_errors(exc: RequestValidationError) -> str:
    """Condense pydantic's error list into one readable line.

    Pydantic's raw output nests ``loc`` tuples and echoes the input back; this
    keeps the part a client can act on -- which field, and what was wrong.
    """
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(piece) for piece in error.get("loc", ()) if piece != "body")
        message = error.get("msg", "invalid value")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "request validation failed"


def register_error_handlers(app: FastAPI) -> None:
    """Attach the handlers that give every failure the same JSON shape."""

    @app.exception_handler(ModelNotReadyError)
    async def _model_not_ready(request: Request, exc: ModelNotReadyError) -> JSONResponse:
        LOGGER.warning("model not ready for %s %s: %s", request.method, request.url.path, exc)
        return error_response(str(exc) or "model is not ready", status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(_format_validation_errors(exc), HTTP_422)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        # Without this, the default handler returns {"detail": ...}, so neither
        # the 400 the predictor raises nor a router 404/405 would match the
        # documented {"error", "status"} shape.
        return error_response(str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Logged in full with the stack trace; the client is told nothing about
        # internals, because an exception message can carry file paths or data.
        LOGGER.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response("internal server error", status.HTTP_500_INTERNAL_SERVER_ERROR)
