"""
Shared test fixtures.

Deliberately synthetic data, not a sample of Ampliphi's extract: the real
dataset is proprietary and is never committed to source control (it isn't
in this repo — see README's "place the provided data/ folder" note), so
CI and local `pytest` runs need a fixture with the same *schema* that
doesn't depend on it existing. `hotel_X` has enough synthetic reservation
history to actually train on; `hotel_Y` has almost none (a cold-start
stand-in for hotel_H); `hotel_Z` has zero reservations and isn't in
`room_types.csv` at all — a stand-in for a hotel that hasn't onboarded yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

STAY_DATES = pd.date_range("2025-03-01", "2025-06-30", freq="7D")
# ^ spans train.py's real TRAIN_FLOOR/TRAIN_END/VAL_START constants
# (2025-01-01 / 2025-06-30 / 2025-06-09) so the internal validation split
# actually has rows in it — an earlier draft of this fixture only used
# Feb-May dates, which left `is_val` all-False and crashed `_best_rounds`
# on an empty validation set. Kept as a comment because it's a real,
# non-obvious trap for anyone editing this fixture's date range later.
LEAD_TIMES = [82, 55, 38, 24, 12, 6, 2]  # days-before-stay each synthetic booking lands


def _write_fixture(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)

    hotels = pd.DataFrame(
        [
            {
                "hotel_id": "hotel_X", "hotel_name": "Test Hotel X", "city_label": "Testville",
                "country": "US", "currency": "USD", "timezone": "America/New_York",
                "pms_type": "agora", "primary_rate_mode": "lowest",
                "lat_coarse": 40.7, "lon_coarse": -74.0, "region_type": "urban_core",
            },
            {
                "hotel_id": "hotel_Y", "hotel_name": "Test Hotel Y", "city_label": "Coldstart Falls",
                "country": "CA", "currency": "CAD", "timezone": "America/Toronto",
                "pms_type": "agora", "primary_rate_mode": "lowest",
                "lat_coarse": 45.4, "lon_coarse": -75.7, "region_type": "resort_mountain",
            },
        ]
    )
    hotels.to_csv(base / "hotels.csv", index=False)

    room_types = pd.DataFrame(
        [
            {"hotel_id": "hotel_X", "room_type_code": "rt_x1", "inventory_count": 10,
             "room_type_kind": "optimized", "display_position": 0},
            {"hotel_id": "hotel_Y", "room_type_code": "rt_y1", "inventory_count": 8,
             "room_type_kind": "optimized", "display_position": 0},
        ]
    )
    room_types.to_csv(base / "room_types.csv", index=False)

    meta = {
        "hotel_X": {
            "country": "US", "currency": "USD", "timezone": "America/New_York",
            "pms_type": "agora", "primary_rate_mode": "lowest", "total_rooms": 10,
            "room_types": {"rt_x1": {"inventory_count": 10, "kind": "optimized"}},
        },
        "hotel_Y": {
            "country": "CA", "currency": "CAD", "timezone": "America/Toronto",
            "pms_type": "agora", "primary_rate_mode": "lowest", "total_rooms": 8,
            "room_types": {"rt_y1": {"inventory_count": 8, "kind": "optimized"}},
        },
    }
    (base / "property_metadata.json").write_text(json.dumps(meta, indent=2))

    rows = []
    rid = 0
    for stay in STAY_DATES:
        # hotel_X: full, learnable booking curve — every stay date gets one
        # reservation at each lead time, so the final occupancy pattern and
        # pace are both real signal a model can fit.
        for lead in LEAD_TIMES:
            rid += 1
            booking_date = stay - pd.Timedelta(days=lead)
            rows.append(
                {
                    "reservation_id": f"res_{rid:05d}", "hotel_id": "hotel_X",
                    "room_type_code": "rt_x1", "rate_plan_code": "BAR",
                    "stay_date": stay.strftime("%Y-%m-%d"),
                    "checkout_date": (stay + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                    "booking_date": booking_date.strftime("%Y-%m-%d"),
                    "cancelled_at": "", "num_rooms": 1, "num_adults": 2, "num_children": "",
                    "rate_amount": "", "total_amount": 100.0, "currency_code": "USD",
                    "booking_source": "web", "status": "checked out",
                }
            )
    # hotel_Y: sparse, off-season-only, cold-start stand-in — 3 reservations
    # total, all with a short lead time.
    for i, stay in enumerate(STAY_DATES[:3]):
        rid += 1
        booking_date = stay - pd.Timedelta(days=5)
        rows.append(
            {
                "reservation_id": f"res_{rid:05d}", "hotel_id": "hotel_Y",
                "room_type_code": "rt_y1", "rate_plan_code": "BAR",
                "stay_date": stay.strftime("%Y-%m-%d"),
                "checkout_date": (stay + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "booking_date": booking_date.strftime("%Y-%m-%d"),
                "cancelled_at": "", "num_rooms": 1, "num_adults": 1, "num_children": "",
                "rate_amount": "", "total_amount": 100.0, "currency_code": "CAD",
                "booking_source": "web", "status": "checked out",
            }
        )
    pd.DataFrame(rows).to_csv(base / "reservations.csv", index=False)
    return base


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    return _write_fixture(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, synthetic_data_dir) -> Path:
    """Trains once per test session on the synthetic fixture and returns
    the *base* (versioned-parent) model directory, with current.json
    already pointing at the trained version — exactly what predict.py
    expects."""
    from src.train import run_training

    base = tmp_path_factory.mktemp("model")
    run_training(synthetic_data_dir, base, version="test-fixture")
    return base
