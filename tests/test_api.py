"""
Tests for the FastAPI service — previously verified only by hand with
`curl` against a running container, which is real verification but not
one that runs again on the next change. These make it automatic.

Every test here points the app at the synthetic fixture via
`BOOKING_CURVE_DATA_DIR` / `BOOKING_CURVE_MODEL_BASE_DIR` (see
`src/predict.py::_default_paths`) rather than the real data — same
principle as the rest of the suite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client_with_model(synthetic_data_dir, trained_model_dir, monkeypatch):
    monkeypatch.setenv("BOOKING_CURVE_DATA_DIR", str(synthetic_data_dir))
    monkeypatch.setenv("BOOKING_CURVE_MODEL_BASE_DIR", str(trained_model_dir))
    monkeypatch.delenv("BOOKING_CURVE_API_KEY", raising=False)

    import src.api as api_module

    # The rate limiter's storage is a module-level singleton keyed by
    # client IP, and TestClient always presents the same source IP — so
    # every test sharing this fixture would otherwise share one bucket
    # (a test making 65 requests would leave the next test's first
    # request looking rate-limited). Reset per test for isolation; this
    # is a test-only concern; real deployments correctly have one bucket
    # per client IP per process, which is the behavior being tested.
    api_module.limiter.reset()

    with TestClient(api_module.app) as c:
        yield c


@pytest.fixture()
def client_without_model(tmp_path, monkeypatch):
    # An empty base dir: no current.json, so resolve_model_dir raises and
    # the lifespan's except-branch is what should leave state unready.
    monkeypatch.setenv("BOOKING_CURVE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOOKING_CURVE_MODEL_BASE_DIR", str(tmp_path / "no_model_here"))
    monkeypatch.delenv("BOOKING_CURVE_API_KEY", raising=False)

    import src.api as api_module

    api_module.limiter.reset()  # same global limiter as client_with_model — see its comment

    with TestClient(api_module.app) as c:
        yield c


def test_healthz_always_ok_even_without_a_model(client_without_model):
    r = client_without_model.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_503_without_a_model(client_without_model):
    r = client_without_model.get("/readyz")
    assert r.status_code == 503
    assert "not loaded" in r.json()["detail"].lower() or "no current.json" in r.json()["detail"].lower()


def test_booking_curve_503_without_a_model(client_without_model):
    r = client_without_model.get(
        "/v1/booking-curve", params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "2025-08-15"}
    )
    assert r.status_code == 503


def test_readyz_ready_with_a_model(client_with_model):
    r = client_with_model.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model_version"] == "test-fixture"


def test_booking_curve_happy_path(client_with_model):
    r = client_with_model.get(
        "/v1/booking-curve", params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "2025-08-15"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["hotel_id"] == "hotel_X"
    assert body["stay_date"] == "2025-08-15"
    values = [body["predictions"][str(cp)] for cp in (90, 60, 45, 30, 21, 14, 7, 3, 1, 0)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))


def test_booking_curve_with_as_of_date(client_with_model):
    r = client_with_model.get(
        "/v1/booking-curve",
        params={
            "hotel_id": "hotel_X", "room_type_code": "rt_x1",
            "stay_date": "2025-08-15", "as_of_date": "2025-08-01",
        },
    )
    assert r.status_code == 200
    assert r.json()["diagnostics"]["mode"] in ("anchored", "blind", "blind_no_ground_truth")


def test_booking_curve_unseen_hotel_is_200_not_a_crash(client_with_model):
    # hotel_Z is nowhere in the fixture at all — the real cold-start case.
    r = client_with_model.get(
        "/v1/booking-curve", params={"hotel_id": "hotel_Z", "room_type_code": "rt_z", "stay_date": "2025-08-15"}
    )
    assert r.status_code == 200
    assert r.json()["diagnostics"]["known_hotel"] is False


def test_booking_curve_invalid_date_is_422(client_with_model):
    r = client_with_model.get(
        "/v1/booking-curve", params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "not-a-date"}
    )
    assert r.status_code == 422


def test_value_error_from_prediction_becomes_400(client_with_model, monkeypatch):
    # FastAPI's own `date` type validation means a malformed date never
    # reaches predict_booking_curve — so the 400 branch is otherwise
    # unreachable through the real endpoint today. Testing it directly by
    # substitution rather than leaving the wiring unverified.
    import src.api as api_module

    def boom(*a, **k):
        raise ValueError("simulated bad input")

    monkeypatch.setattr(api_module, "predict_booking_curve", boom)
    r = client_with_model.get(
        "/v1/booking-curve", params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "2025-08-15"}
    )
    assert r.status_code == 400
    assert "simulated bad input" in r.json()["detail"]


def test_unexpected_error_becomes_500_without_leaking_internals(client_with_model, monkeypatch):
    import src.api as api_module

    def boom(*a, **k):
        raise RuntimeError("some internal secret detail")

    monkeypatch.setattr(api_module, "predict_booking_curve", boom)
    r = client_with_model.get(
        "/v1/booking-curve", params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "2025-08-15"}
    )
    assert r.status_code == 500
    assert "some internal secret detail" not in r.text
    assert r.json()["detail"] == "Internal error generating prediction"


def test_metrics_endpoint_exposes_prometheus_format(client_with_model):
    client_with_model.get("/healthz")  # generate at least one counted request first
    r = client_with_model.get("/metrics")
    assert r.status_code == 200
    assert "booking_curve_requests_total" in r.text
    assert "booking_curve_model_loaded" in r.text


def test_rate_limit_actually_trips(client_with_model):
    # BOOKING_CURVE_RATE_LIMIT is read once, at import time, by the
    # @limiter.limit decorator (a slowapi/limits constraint — unlike
    # require_api_key's per-call env read, this one can't be reconfigured
    # via monkeypatch after the module's already loaded). So this exercises
    # the real default (60/minute) directly rather than pretending to
    # reconfigure it: fire more than 60 requests and confirm at least one
    # actually gets rate-limited, not just that the decorator is present.
    # (client_with_model resets the limiter on setup — see its own comment
    # — so this starts from a clean bucket regardless of test order.)
    from src.api import limiter

    try:
        statuses = [
            client_with_model.get(
                "/v1/booking-curve",
                params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "2025-08-15"},
            ).status_code
            for _ in range(65)
        ]
        assert 429 in statuses
        assert statuses.count(200) >= 55  # the limit is 60/minute, not near-zero
    finally:
        limiter.reset()  # don't leave the bucket exhausted for client_without_model's tests


def test_every_response_carries_a_request_id(client_with_model):
    r = client_with_model.get("/healthz")
    assert "x-request-id" in r.headers
    # a real uuid4, not a placeholder
    assert len(r.headers["x-request-id"]) == 36


def test_api_key_enforced_when_configured(client_with_model, monkeypatch):
    # require_api_key reads BOOKING_CURVE_API_KEY fresh on every call (see
    # src/api.py) specifically so this doesn't need a module reload —
    # reloading a module that registers Prometheus metrics at import time
    # crashes on the second import (DuplicateTimeseries in the default
    # registry), which is exactly what an earlier version of this test did.
    monkeypatch.setenv("BOOKING_CURVE_API_KEY", "secret-test-key")

    no_key = client_with_model.get(
        "/v1/booking-curve", params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "2025-08-15"}
    )
    assert no_key.status_code == 401

    wrong_key = client_with_model.get(
        "/v1/booking-curve",
        params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "2025-08-15"},
        headers={"X-API-Key": "wrong"},
    )
    assert wrong_key.status_code == 401

    right_key = client_with_model.get(
        "/v1/booking-curve",
        params={"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": "2025-08-15"},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert right_key.status_code == 200
