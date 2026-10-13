"""Configuration and error handling."""

from .config import PROJECT_ROOT, Settings, get_settings
from .errors import ModelNotReadyError, error_response, register_error_handlers

__all__ = [
    "ModelNotReadyError",
    "PROJECT_ROOT",
    "Settings",
    "error_response",
    "get_settings",
    "register_error_handlers",
]
