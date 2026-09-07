"""
FastAPI wrapper around predict_booking_curve, matching Ampliphi's actual
stack (FastAPI/Pydantic).

This is a real serving skeleton, not a toy: the model loads once at
startup (so a broken artifact fails the readiness probe at deploy time,
not on a customer's first request), `/healthz` actually reflects load
state, every request gets a traceable request id, errors are split into
"your input was bad" (400) vs. "something broke on our side" (500), the
business endpoint is versioned (`/v1/...`), it exposes Prometheus metrics,
and it rate-limits. None of this is the full production picture — see
DESIGN.md §6.5 for the batch-precompute path this is meant to sit
alongside, and the caveats on each mechanism below (rate limiting is
per-process, not shared across replicas; the API-key check is a
placeholder for the platform's real identity provider).

Run:    uvicorn src.api:app --host 0.0.0.0 --port 8000
Try:    curl "http://localhost:8000/v1/booking-curve?hotel_id=hotel_C&room_type_code=rt_ea30c05c4c&stay_date=2025-08-15"
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .logging_config import get_logger
from .model import BookingCurveModel
from .predict import _context, _default_paths, predict_booking_curve
from .pricing import recommend_price
from .registry import resolve_model_dir

log = get_logger(__name__)

# Requests/minute per client IP, in-process. Deliberately simple — the
# honest limitation: with more than one uvicorn worker (WEB_CONCURRENCY>1,
# see Dockerfile) or more than one replica, each process counts
# independently, so the *effective* limit is (this number) x (worker
# count) x (replica count), not a hard global ceiling. A real multi-replica
# deployment needs a shared store (Redis) for this to mean what it says;
# this is still strictly better than no limit at all, and correctly sized
# for the single-process case this repo actually runs at.
RATE_LIMIT = os.environ.get("BOOKING_CURVE_RATE_LIMIT", "60/minute")

limiter = Limiter(key_func=get_remote_address)

# --- Prometheus metrics -----------------------------------------------
# Deliberately few, chosen for what you'd actually page on: request
# volume/latency/errors by route, plus which model version is live (so a
# "silent regression on hotel_F six weeks from now" — DESIGN.md §6.5's own
# question — has *something* to correlate against a deploy timestamp with).
REQUEST_COUNT = Counter(
    "booking_curve_requests_total", "Total requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "booking_curve_request_duration_seconds", "Request duration", ["method", "path"]
)
MODEL_INFO = Gauge(
    "booking_curve_model_loaded", "1 if a model is loaded and ready, 0 otherwise", ["version"]
)


class ServiceState:
    model: BookingCurveModel | None = None
    model_version: str | None = None
    load_error: str | None = None


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reset on every startup, not just the first — otherwise a second
    # lifespan run in the same process (a restart in a long-lived worker,
    # or a test harness spinning up a fresh TestClient) would see stale
    # state left over from a previous run instead of a clean load.
    state.model = None
    state.model_version = None
    state.load_error = None
    data_dir, model_base = _default_paths()
    try:
        resolved = resolve_model_dir(Path(model_base))
        state.model_version = resolved.name
        # Load through _context — the exact same loader predict_booking_curve
        # itself uses — rather than a second load path here, and without
        # hardcoding a real dataset's hotel/room-type id just to warm the
        # cache (this service has no business assuming which hotels exist).
        _context(data_dir, model_base, None)
        state.model = True  # sentinel: loaded OK (the real object is cached in predict.py)
        MODEL_INFO.labels(version=state.model_version).set(1)
        log.info(f"Startup: model version {state.model_version} loaded and warmed.")
    except Exception as exc:
        state.load_error = str(exc)
        log.error(f"Startup: model failed to load — {exc!r}")
    yield


app = FastAPI(title="Ampliphi Booking Curve Service", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    response.headers["X-Request-ID"] = request_id
    # request.url.path, not the templated route — fine at this route
    # count (four routes total); would want the templated path specifically
    # to avoid a cardinality blowup if this ever grew to path-parameterized
    # routes with high-cardinality segments.
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
    log.info(
        f"request_id={request_id} method={request.method} path={request.url.path} "
        f"status={response.status_code} duration_ms={duration * 1000:.1f}"
    )
    return response


def require_api_key(x_api_key: str | None = Header(default=None)):
    # Read fresh per request, not once at import time: an env-var-based
    # secret rotation (a re-mounted k8s secret, a test) takes effect
    # immediately, without requiring a full process restart to notice it.
    # Unset means no auth — fine for local dev, not for anything reachable
    # off localhost. Swap this whole mechanism for the platform's real
    # identity provider before this ever serves a real request; it exists
    # to show the right instinct, not to be the final answer.
    api_key = os.environ.get("BOOKING_CURVE_API_KEY")
    if api_key and x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


class BookingCurveResponse(BaseModel):
    hotel_id: str
    room_type_code: str
    stay_date: str
    predictions: dict[str, float]
    intervals: dict[str, dict[str, float]]
    diagnostics: dict


class PriceRecommendationResponse(BaseModel):
    recommended_price: float
    base_rate: float
    adjustment_pct: float
    pace_ratio: float | None
    confidence: float | None
    reason: str
    diagnostics: dict


@app.get("/healthz")
def healthz():
    """Liveness: process is up. Always 200 once the process can respond."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness: model actually loaded. A load balancer / ECS task
    definition should point health checks here, not at /healthz — a
    process that's up but has no model should not receive traffic."""
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {state.load_error or 'still starting'}",
        )
    return {"status": "ready", "model_version": state.model_version}


@app.get("/metrics")
def metrics():
    """Prometheus scrape target. Unauthenticated on purpose — that's the
    norm for in-cluster scraping (the scraper is trusted-network, not a
    public caller) and keeping it separate from `require_api_key` means a
    metrics-scraper outage never depends on API-key rotation."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


v1 = APIRouter(prefix="/v1")


@v1.get("/booking-curve", response_model=BookingCurveResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
def get_booking_curve(
    request: Request,  # required positionally by slowapi's decorator, unused otherwise
    hotel_id: str,
    room_type_code: str,
    stay_date: date,
    as_of_date: date | None = None,
):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predict_booking_curve(
            hotel_id,
            room_type_code,
            stay_date.isoformat(),
            as_of_date=as_of_date.isoformat() if as_of_date else None,
        )
    except (KeyError, ValueError) as exc:
        # Bad/unrecognized input we understand — safe to explain.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        # Anything else is our bug, not the caller's. Log the real error,
        # don't leak internals to the client.
        log.exception(f"Unhandled error predicting {hotel_id}/{room_type_code}/{stay_date}")
        raise HTTPException(status_code=500, detail="Internal error generating prediction")


@v1.get(
    "/price-recommendation", response_model=PriceRecommendationResponse, dependencies=[Depends(require_api_key)]
)
@limiter.limit(RATE_LIMIT)
def get_price_recommendation(
    request: Request,  # required positionally by slowapi's decorator, unused otherwise
    hotel_id: str,
    room_type_code: str,
    stay_date: date,
    as_of_date: date,
    base_rate: float,
):
    """Pace-based yield adjustment — see src/pricing.py's module docstring
    for exactly what this is (and, importantly, isn't: no learned price
    elasticity, see DESIGN.md SS6.10/SS6.12). Unlike /booking-curve,
    as_of_date is required here: without a live pickup signal there's
    nothing for this endpoint to react to, by design, not by omission."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return recommend_price(
            hotel_id,
            room_type_code,
            stay_date.isoformat(),
            as_of_date.isoformat(),
            base_rate,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        log.exception(f"Unhandled error pricing {hotel_id}/{room_type_code}/{stay_date}")
        raise HTTPException(status_code=500, detail="Internal error generating price recommendation")


app.include_router(v1)
