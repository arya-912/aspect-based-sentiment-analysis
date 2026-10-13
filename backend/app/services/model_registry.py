"""Process-wide holder for the loaded model, encoder and label map.

The artifact is a few hundred megabytes, so it is loaded once at startup and
shared by every request. Accessors raise :class:`ModelNotReadyError` rather
than returning ``None``, so a half-initialised service fails with a clean 503
instead of an ``AttributeError`` deep inside a request.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

from ..core.config import PROJECT_ROOT, get_settings
from ..core.errors import ModelNotReadyError

LOGGER = logging.getLogger(__name__)

METRICS_FILE = "metrics.json"


def _ensure_repo_on_path() -> None:
    """Put the repository root on ``sys.path``.

    The training package is not installed; it is imported as
    ``training.src.artifact`` (``training/`` resolves as a namespace package).
    Prepending the root makes that import work regardless of where uvicorn was
    started from.
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class ModelRegistry:
    """Singleton registry. Use :data:`registry`, or ``ModelRegistry()``."""

    _instance: "ModelRegistry | None" = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_state()
        return cls._instance

    def _init_state(self) -> None:
        self._model: Any = None
        self._encoder: Any = None
        self._label_map: Any = None
        self._config: dict[str, Any] | None = None
        self._artifact_dir: Path | None = None
        self._device: str = "cpu"
        self._load_error: str | None = None
        self._load_lock = threading.Lock()

    # ---- loading ------------------------------------------------------ #

    def load(
        self, artifact_dir: str | Path | None = None, device: str | None = None
    ) -> None:
        """Load an artifact into the registry.

        Args:
            artifact_dir: the artifact directory; defaults to
                ``Settings.artifact_path``.
            device: ``cpu`` or ``cuda``; defaults to ``Settings.DEVICE``.

        Raises:
            ModelNotReadyError: if the artifact is missing or cannot be loaded.
                The underlying cause is logged and kept in
                :meth:`load_error` for ``/health`` to report.
        """
        settings = get_settings()
        directory = Path(artifact_dir) if artifact_dir else settings.artifact_path
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        device = device or settings.DEVICE

        with self._load_lock:
            _ensure_repo_on_path()
            try:
                # Imported here, not at module scope: it pulls in torch and
                # transformers, which would otherwise make simply importing
                # this module cost seconds.
                from training.src.artifact import load_artifact

                model, encoder, label_map, config = load_artifact(
                    directory, device=device
                )
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                LOGGER.error("failed to load artifact from %s: %s", directory, exc)
                raise ModelNotReadyError(
                    f"could not load the model from {directory}"
                ) from exc

            self._model = model
            self._encoder = encoder
            self._label_map = label_map
            self._config = config
            self._artifact_dir = directory
            self._device = str(device)
            self._load_error = None

        LOGGER.info(
            "model ready: %s on %s (labels=%s)",
            self.model_version,
            device,
            getattr(label_map, "labels", None),
        )

    def reset(self) -> None:
        """Drop the loaded model. Used by tests and by a reload endpoint."""
        self._init_state()

    # ---- accessors ---------------------------------------------------- #

    def _require(self, value: Any, what: str) -> Any:
        """Return ``value`` or fail with a 503-mapped error.

        The message stays free of the underlying cause on purpose: it can
        carry filesystem paths, and this string reaches the client. The cause
        is logged at ERROR when the load fails and is available in-process via
        :meth:`load_error`.
        """
        if value is None:
            raise ModelNotReadyError(f"{what} is not loaded")
        return value

    def get_model(self) -> Any:
        return self._require(self._model, "model")

    def get_encoder(self) -> Any:
        return self._require(self._encoder, "encoder")

    def get_label_map(self) -> Any:
        return self._require(self._label_map, "label map")

    def get_config(self) -> dict[str, Any]:
        return self._require(self._config, "model config")

    def is_loaded(self) -> bool:
        """True when every piece needed to serve a prediction is present."""
        return None not in (self._model, self._encoder, self._label_map, self._config)

    # ---- metadata ----------------------------------------------------- #

    @property
    def device(self) -> str:
        return self._device

    @property
    def artifact_dir(self) -> Path | None:
        return self._artifact_dir

    def load_error(self) -> str | None:
        """Why the last load failed, if it did."""
        return self._load_error

    @property
    def model_version(self) -> str:
        """The artifact's version string.

        ``build_artifact`` writes ``version`` (and nests the configured value
        under ``artifact.version``); ``model_version`` is accepted too so a
        hand-written config.json works. Falls back to the directory name,
        which is the version by construction.
        """
        config = self._config or {}
        for candidate in (
            config.get("model_version"),
            config.get("version"),
            (config.get("artifact") or {}).get("version"),
            config.get("run_id"),
        ):
            if candidate:
                return str(candidate)
        return self._artifact_dir.name if self._artifact_dir else "unknown"

    @property
    def base_model(self) -> str:
        """The backbone the artifact was fine-tuned from."""
        config = self._config or {}
        return str((config.get("model") or {}).get("name") or "unknown")

    @property
    def labels(self) -> list[str]:
        """Label names in class-id order."""
        label_map = self._label_map
        if label_map is not None and getattr(label_map, "labels", None):
            return list(label_map.labels)
        config = self._config or {}
        return list((config.get("data") or {}).get("labels") or [])

    def read_metrics(self) -> dict[str, Any] | None:
        """The artifact's ``metrics.json``, or ``None`` if it is absent.

        Missing or malformed metrics are not fatal -- a model can serve
        predictions perfectly well without its evaluation report -- so this
        logs and returns ``None`` rather than raising.
        """
        if self._artifact_dir is None:
            return None
        path = self._artifact_dir / METRICS_FILE
        if not path.is_file():
            LOGGER.warning("no %s in %s", METRICS_FILE, self._artifact_dir)
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("could not read %s: %s", path, exc)
            return None


#: The shared instance the routes and services use.
registry = ModelRegistry()
