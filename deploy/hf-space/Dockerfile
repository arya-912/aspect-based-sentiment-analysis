# Backend container for the ABSA API, shaped for a Hugging Face Space
# (Docker SDK). One-time setup is in the README next to this file.
#
# The image carries no application code of its own: it clones the public
# GitHub repo and pulls the trained artifact from the Hub at build time. That
# keeps the Space repository down to the three files in this directory, and
# makes a rebuild pick up whatever those two upstream sources currently hold.

FROM python:3.12-slim

# GIT_REF pins the code, MODEL_REPO the weights. Both are overridable from the
# Space's Settings -> Variables, so a rebuild can roll either forward or back
# without editing this file.
ARG GIT_REPO=https://github.com/arya-912/aspect-based-sentiment-analysis.git
ARG GIT_REF=main
ARG MODEL_REPO=arya912/absa-joint-v1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# A Space's container runs as uid 1000. HOME has to be writable by that uid or
# the Hub cache cannot be created and the build fails on the download below.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user

# CPU wheels only. The default index serves the CUDA build, which drags in
# ~2.5GB of nvidia-* packages that a CPU Space can never use and which push
# the image past the storage a free Space is given.
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch==2.14.1"

COPY --chown=user requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --depth 1 wants a branch or tag name; a raw commit SHA will not clone this
# way, so pin GIT_REF to a tag if you need a specific commit.
RUN git clone --depth 1 --branch "${GIT_REF}" "${GIT_REPO}" app
WORKDIR /home/user/app

# The weights are gitignored, so they come from the Hub, not from the clone.
# Baked into the image rather than fetched at boot on purpose: a free Space
# sleeps when idle and cold-starts often, and pulling 450MB on every wake
# would time out the frontend's first request.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODEL_REPO}', local_dir='training/artifacts/absa-joint-v1')"

ENV ARTIFACT_DIR=training/artifacts/absa-joint-v1 \
    DEVICE=cpu
# The weights are in the image, so a load failure means the image itself is
# broken. Crash on boot and say so in the Space logs rather than serving a
# "degraded" /health that looks like a transient problem.
ENV ALLOW_DEGRADED_START=false

# CORS_ORIGINS is deliberately not set here: it carries the Vercel URL, which
# this file should not have to know. Set it in Settings -> Variables.

WORKDIR /home/user/app/backend
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
