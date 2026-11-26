"""Pull the served artifact from the Hub into the deployment's file tree.

This is the Build Command of the `absa-api` Vercel project (wired up in the
repository root's `vercel.json`). Vercel runs it after installing
`requirements.txt` and before bundling the function, so whatever it writes
under the project root is part of the deployed function.

Why a build step at all: `model.pt` is ~450MB and gitignored, so it cannot
travel with the code, and a Vercel function has no persistent disk to cache
it on between cold starts. Downloading it at build time puts it *inside* the
bundle -- paid for once per deployment instead of once per cold start, which
would otherwise add ~450MB of transfer to a user's first request.

Both inputs are overridable from the project's environment variables, so a
rebuild can roll the weights forward or back without editing this file:

    MODEL_REPO     Hub model repo to pull           (arya912/absa-joint-v1)
    ARTIFACT_DIR   where to put it, repo-relative   (training/artifacts/absa-joint-v1)

ARTIFACT_DIR is deliberately the same variable the service reads at runtime
(`backend/app/core/config.py`), so the two cannot drift apart: setting it
once points both the download and the loader at the same place.

Run it by hand to populate a working tree the same way a build would::

    python deploy/vercel/fetch_artifact.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Repository root: this file is <root>/deploy/vercel/fetch_artifact.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_REPO = "arya912/absa-joint-v1"
DEFAULT_ARTIFACT_DIR = "training/artifacts/absa-joint-v1"

#: What `training.src.artifact.load_artifact` insists on, plus the tokenizer
#: directory it needs a moment later. Checked after the download so a build
#: fails here, with a readable message, rather than at the first cold start.
REQUIRED_ENTRIES = (
    "config.json",
    "label_map.json",
    "model.pt",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
)


def main() -> int:
    repo_id = os.environ.get("MODEL_REPO") or DEFAULT_MODEL_REPO
    artifact_dir = os.environ.get("ARTIFACT_DIR") or DEFAULT_ARTIFACT_DIR

    destination = Path(artifact_dir)
    if not destination.is_absolute():
        destination = REPO_ROOT / destination

    # The Hub cache defaults to $HOME/.cache/huggingface. `local_dir` already
    # writes real files rather than symlinks into that cache, but pointing
    # HOME-independent state at /tmp keeps the build from depending on
    # whatever HOME the builder happens to set.
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")

    from huggingface_hub import snapshot_download

    print(f"fetching {repo_id} -> {destination}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(destination))

    missing = [name for name in REQUIRED_ENTRIES if not (destination / name).exists()]
    if missing:
        print(
            f"error: {repo_id} is missing {', '.join(missing)}; "
            f"the function would boot without a model",
            file=sys.stderr,
        )
        return 1

    weights = (destination / "model.pt").stat().st_size
    print(f"ok: model.pt is {weights / 1e6:.0f}MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
