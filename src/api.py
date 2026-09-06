"""
FastAPI wrapper around predict_booking_curve, matching Ampliphi's actual
stack (FastAPI/Pydantic).

This is a real serving skeleton, not a toy: the model loads once at
startup (so a broken artifact fails the readiness probe at deploy time,
not on a customer's first request), `/healthz` actually reflects load
state, every request gets a traceable request id, and errors are split
into "your input was bad" (400, message safe to show) vs. "something
broke on our side" (500, logged with a stack trace, generic message to
the client). None of this is the full production picture — see
DESIGN.md §6.5 for the batch-precompute path this is meant to sit
alongside, and for what's still missing (real auth via the platform's
identity provider, rate limiting, tracing) rather than the placeholder
API-key check below.

Run:    uvicorn src.api:app --host 0.0.0.0 --port 8000
Try:    curl "http://localhost:8000/booking-curve?hotel_id=hotel_C&room_type_code=rt_ea30c05c4c&stay_date=2025-08-15"
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from .logging_config import get_logger
from .model import BookingCurveModel
from .predict import _context, _default_paths, predict_booking_curve
from .registry import resolve_model_dir

log = get_logger(__name__)

# Set to require an API key on every request (checked in `require_api_key`
# below). Unset (the default) means no auth — fine for local dev, not for
# anything reachable off localhost. Swap this whole mechanism for the
# platform's real identity provider before this ever serves a real request;
# it exists to show the right instinct, not to be the final answer.
API_KEY = os.environ.get("BOOKING_CURVE_API_KEY")


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
        log.info(f"Startup: model version {state.model_version} loaded and warmed.")
    except Exception as exc:
        state.load_error = str(exc)
        log.error(f"Startup: model failed to load — {exc!r}")
    yield


app = FastAPI(title="Ampliphi Booking Curve Service", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.monotonic()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    duration_ms = (time.monotonic() - start) * 1000
    log.info(
        f"request_id={request_id} method={request.method} path={request.url.path} "
        f"status={response.status_code} duration_ms={duration_ms:.1f}"
    )
    return response


def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


class BookingCurveResponse(BaseModel):
    hotel_id: str
    room_type_code: str
    stay_date: str
    predictions: dict[str, float]
    intervals: dict[str, dict[str, float]]
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


@app.get("/booking-curve", response_model=BookingCurveResponse, dependencies=[Depends(require_api_key)])
def get_booking_curve(
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
