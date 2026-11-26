"""ASGI entry point for the Vercel deployment of the API.

Vercel's Python runtime looks for a top-level ``app`` in one of a handful of
filenames at the project's root directory -- ``app.py``, ``index.py``,
``server.py``, ``main.py``, ``wsgi.py``, ``asgi.py`` -- and builds the whole
thing into a single function. This is the file it finds; the application
itself still lives in ``backend/app``.

A thin shim rather than a second application factory, for two reasons:

* ``backend/app/main.py`` uses relative imports and the repository root has to
  be on ``sys.path`` for ``backend.app.main`` (and, later,
  ``training.src.artifact``) to import. Vercel imports this file with the root
  as the working directory, but the insert below makes that explicit rather
  than assumed.
* The two ``HF_HOME`` / ``HF_HUB_OFFLINE`` defaults have to be set *before*
  transformers is imported, and importing the app does not import it
  (``model_registry.load`` pulls torch and transformers in lazily). A Vercel
  function's filesystem is read-only apart from ``/tmp``, so a hub cache
  pointed at the default ``$HOME/.cache`` would fail at import; and the
  artifact is already on local disk, so nothing should reach the network.

Run locally the same way Vercel does::

    uvicorn server:app --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from backend.app.main import app  # noqa: E402  (after the environment above)

__all__ = ["app"]
