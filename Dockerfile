# Serving image for the booking-curve API. Also used for batch jobs
# (train/predict) by overriding CMD — see the two `docker run` examples
# below — rather than maintaining a second, near-duplicate image.
#
# Build:   docker build -t booking-curve-service .
# Serve:   docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/artifacts:/app/artifacts" booking-curve-service
# Train:   docker run --rm -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/artifacts:/app/artifacts" booking-curve-service python -m src.train
# Predict: docker run --rm -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/artifacts:/app/artifacts" -v "$(pwd)/evaluation:/app/evaluation" booking-curve-service python -m src.predict --generate-eval
#
# `data/` and `artifacts/` are mounted volumes, not baked into the image:
# the proprietary Ampliphi extract has no business inside a container
# image that might end up in a registry, and a model artifact should be
# swappable (a version rollback, a canary) without a rebuild.

# Pinned by digest, not just the `3.11-slim` tag: a floating tag means a
# rebuild next month can silently pull a different underlying image (and a
# different set of OS-level CVEs) than the one this Dockerfile was actually
# built and verified against. Re-resolve deliberately (`docker pull
# python:3.11-slim` + update this digest) rather than let it drift.
FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6

WORKDIR /app

# libgomp1 provides libgomp.so.1 — LightGBM's compiled core is linked
# against OpenMP and segfaults-on-import without it. `python:*-slim`
# strips it (and most other system libraries) to stay small; this is the
# single most common "works on my machine, breaks in the container"
# LightGBM issue. Found by actually running the built image, not by
# reading the Dockerfile — pip installing successfully says nothing about
# whether the compiled extension it just installed can load.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Only the dependency manifest first, so the (slow) pip install layer is
# cached across rebuilds that only touch application code.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# uvicorn/fastapi/pydantic are needed for the serving CMD below even
# though they're a "dev" concern in requirements-dev.txt locally — the
# runtime image needs them unconditionally to serve anything.
RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.52.4 pydantic==2.13.5

COPY src/ src/
COPY evaluation/evaluate.py evaluation/evaluate.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz', timeout=3).status==200 else 1)"

# JSON-array CMD (no shell wrapping) pointing at a script that `exec`s
# uvicorn, not shell-form CMD directly — see docker-entrypoint.sh for why:
# shell-form here would swallow SIGTERM on `docker stop` instead of
# forwarding it, breaking graceful shutdown. A single worker by default
# (correct for the scale this repo actually runs at), overridable per
# deployment without a rebuild: `docker run -e WEB_CONCURRENCY=4 ...`.
# Multiple uvicorn workers each load their own copy of the model (no
# shared memory) — fine at this model's size (~200KB), worth knowing if
# that ever changes.
CMD ["/usr/local/bin/docker-entrypoint.sh"]
