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

FROM python:3.11-slim

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

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz', timeout=3).status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
