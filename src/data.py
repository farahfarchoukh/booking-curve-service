"""
Data loading and booking-curve construction.

This module is the *single* source of truth for how raw reservations become
booking-curve labels. `train.py` calls `build_actual_curve_table` to build
training targets; `predict.py` calls the same function (via
`realized_fraction_as_of`) to compute the "already observed" portion of a
curve at inference time. Using one implementation for both prevents the
classic train/serve skew where the training-label definition and the
serving-time "what do we already know" logic quietly drift apart.

The construction semantics deliberately match `evaluation/evaluate.py` and
`starter/baseline_model.py` exactly (same "covering" + "booked-by-cutoff"
definition), so our training labels and the fixed grader's ground truth are
built the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_config import get_logger

log = get_logger(__name__)

CHECKPOINTS = [90, 60, 45, 30, 21, 14, 7, 3, 1, 0]

TRAIN_END = "2025-06-30"
TEST_START = "2025-07-01"
TEST_END = "2025-09-30"

# Hotels with usable reservation history (targets can be built).
RESERVATION_HOTELS = ("hotel_C", "hotel_H")

_EPOCH = np.datetime64("2020-01-01", "D")


def _to_day_int(series: pd.Series) -> np.ndarray:
    """Datetime series -> int days since an arbitrary epoch (fast, overflow-safe)."""
    return (series.values.astype("datetime64[D]") - _EPOCH).astype(np.int64)


@dataclass
class StaticContext:
    """Everything about hotels/room types that is known at onboarding time,
    i.e. *before* a single reservation exists. This is deliberately the only
    kind of feature we let the model depend on for "identity" — see
    src/features.py and DESIGN.md §6.1/§6.2 for why raw hotel_id is excluded.
    """

    hotels: pd.DataFrame
    room_types: pd.DataFrame
    room_static: pd.DataFrame  # one row per (hotel_id, room_type_code), joined
    inventory_lookup: dict  # (hotel_id, room_type_code) -> inventory_count
    total_rooms_lookup: dict  # hotel_id -> total_rooms

    @property
    def known_hotels(self) -> list:
        return sorted(self.hotels["hotel_id"])  # see known_room_types() docstring

    def known_room_types(self, hotel_id: str) -> list:
        """Sorted, not a set: a `set`'s iteration order depends on Python's
        per-process hash seed (randomized by default), so iterating one
        directly to build a training table makes the row order — and
        therefore what LightGBM's histogram construction sees — silently
        different across process runs even with every RNG seed pinned.
        This was a real bug: it's what caused retrains on identical data
        to drift by ~0.01 weighted MAE (see train.py's `_DETERMINISM`
        comment, which fixed the RNG half of the problem but not this
        half). Returning a sorted list makes iteration order a property of
        the data, not the process.
        """
        return sorted(
            self.room_types.loc[self.room_types.hotel_id == hotel_id, "room_type_code"]
        )


def _repair_missing_room_types(room_types: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """`reservations.csv` references a handful of room_type_codes that don't
    exist in `room_types.csv` (2 on hotel_C, 2 on hotel_H) — a realistic
    PMS/dimension-table sync-drift artifact, not something to paper over
    silently. We patch the dimension table so these room types get a
    curve like everything else, inferring `inventory_count` as the peak
    number of concurrent reservations ever observed for that code (a
    conservative stand-in for "how many physical rooms must exist for this
    many bookings to have co-occurred"), floored at 1. In production this
    mismatch is exactly the kind of thing that should raise a data-quality
    alert (see DESIGN.md §6.3), not just get silently patched — we log it.
    """
    res = pd.read_csv(
        data_dir / "reservations.csv", parse_dates=["stay_date", "checkout_date"]
    )
    known = set(zip(room_types.hotel_id, room_types.room_type_code))
    missing = sorted(set(zip(res.hotel_id, res.room_type_code)) - known)
    if not missing:
        return room_types

    extra_rows = []
    for hotel_id, room_type_code in missing:
        sub = res[(res.hotel_id == hotel_id) & (res.room_type_code == room_type_code)]
        nights = pd.date_range(sub.stay_date.min(), sub.checkout_date.max() - pd.Timedelta(days=1))
        stay_i = _to_day_int(sub["stay_date"])
        checkout_i = _to_day_int(sub["checkout_date"])
        night_i = (nights.values.astype("datetime64[D]") - _EPOCH).astype(np.int64)
        concurrent = (
            (stay_i[None, :] <= night_i[:, None]) & (checkout_i[None, :] > night_i[:, None])
        ).sum(axis=1)
        inferred_inventory = max(int(concurrent.max()) if len(concurrent) else 1, 1)
        log.warning(
            f"data-quality: {hotel_id}/{room_type_code} not in room_types.csv "
            f"({len(sub)} reservations reference it) — inferring inventory_count="
            f"{inferred_inventory} from peak concurrent bookings. "
            f"In production this should also raise a dimension-table alert, not just log."
        )
        extra_rows.append(
            {
                "hotel_id": hotel_id,
                "room_type_code": room_type_code,
                "inventory_count": inferred_inventory,
                "room_type_kind": "unknown",
                "display_position": -1,
            }
        )
    return pd.concat([room_types, pd.DataFrame(extra_rows)], ignore_index=True)


def load_static_context(data_dir: Path) -> StaticContext:
    hotels = pd.read_csv(data_dir / "hotels.csv")
    room_types = pd.read_csv(data_dir / "room_types.csv")
    room_types = _repair_missing_room_types(room_types, data_dir)

    with open(data_dir / "property_metadata.json") as f:
        meta = json.load(f)
    total_rooms_lookup = {h: v["total_rooms"] for h, v in meta.items()}

    room_static = room_types.merge(hotels, on="hotel_id", how="left")
    room_static["total_rooms"] = room_static["hotel_id"].map(total_rooms_lookup)
    room_static["room_share"] = (
        room_static["inventory_count"] / room_static["total_rooms"]
    )

    inventory_lookup = {
        (r.hotel_id, r.room_type_code): r.inventory_count
        for r in room_static.itertuples()
    }

    return StaticContext(
        hotels=hotels,
        room_types=room_types,
        room_static=room_static,
        inventory_lookup=inventory_lookup,
        total_rooms_lookup=total_rooms_lookup,
    )


def load_reservations(data_dir: Path) -> pd.DataFrame:
    """Load + lightly clean reservations.

    Data-quality notes (see DESIGN.md / README "known data issues"):
      - ~2% of rows have booking_date > stay_date (booking recorded after
        check-in). Anonymization date-jitter, not real late bookings that
        book themselves into the past. We clip booking_date to stay_date so
        these still contribute to the curve as "day-of" bookings instead of
        silently distorting lead-time features.
      - A handful of hotel_H booking_date values land in 2026 (past the
        entire dataset). We leave these as-is: `evaluate.py`'s ground truth
        is built with the same unclipped rule, so "fixing" them would make
        our training target definition diverge from the fixed grader.
    """
    res = pd.read_csv(
        data_dir / "reservations.csv",
        parse_dates=["stay_date", "checkout_date", "booking_date"],
    )
    res = res[res["status"] != "cancelled"].copy()
    res["booking_date"] = res["booking_date"].clip(upper=res["stay_date"])
    return res


def build_actual_curve_table(
    reservations: pd.DataFrame,
    static: StaticContext,
    night_start: str,
    night_end: str,
    hotels: tuple = RESERVATION_HOTELS,
) -> pd.DataFrame:
    """Build long-format actual booking curves: one row per
    (hotel_id, room_type_code, stay_date, cp) with the realized cumulative
    booked fraction. Nights with zero covering reservations are dropped —
    matching `evaluate.py` exactly, so labels and grading ground truth agree.

    Returns columns: hotel_id, room_type_code, stay_date, cp, actual
    """
    nights = pd.date_range(night_start, night_end)
    night_ints = (nights.values.astype("datetime64[D]") - _EPOCH).astype(np.int64)
    cp_arr = np.array(CHECKPOINTS)

    rows = []
    for hotel_id in hotels:
        sub_hotel = reservations[reservations.hotel_id == hotel_id]
        for room_type_code in static.known_room_types(hotel_id):
            sub = sub_hotel[sub_hotel.room_type_code == room_type_code]
            if sub.empty:
                continue
            total_rooms = static.inventory_lookup.get((hotel_id, room_type_code), 1) or 1

            stay_i = _to_day_int(sub["stay_date"])
            checkout_i = _to_day_int(sub["checkout_date"])
            booking_i = _to_day_int(sub["booking_date"])

            # covering[n, r] = reservation r covers night n
            covering = (stay_i[None, :] <= night_ints[:, None]) & (
                checkout_i[None, :] > night_ints[:, None]
            )
            has_coverage = covering.any(axis=1)
            if not has_coverage.any():
                continue

            # booked_by_cutoff[n, c, r] would be 4D-ish; do it per night instead.
            active_nights = np.where(has_coverage)[0]
            for n_idx in active_nights:
                cover_mask = covering[n_idx]
                covering_booking = booking_i[cover_mask]
                night_date = nights[n_idx]
                cutoffs_n = night_ints[n_idx] - cp_arr
                counts = (covering_booking[None, :] <= cutoffs_n[:, None]).sum(axis=1)
                fracs = np.minimum(counts / total_rooms, 1.0)
                for cp, frac in zip(CHECKPOINTS, fracs):
                    rows.append(
                        (hotel_id, room_type_code, night_date, int(cp), float(frac))
                    )

    df = pd.DataFrame(
        rows, columns=["hotel_id", "room_type_code", "stay_date", "cp", "actual"]
    )
    return df


def hotels_with_reservations(reservations: pd.DataFrame) -> list[str]:
    """Which hotels actually have reservation ground truth in *this* data —
    derived from the data, not a hardcoded pair. `RESERVATION_HOTELS`
    (hotel_C/hotel_H) is specific to this one Ampliphi extract; a real
    onboarding pipeline adds hotels continuously; a function that only
    ever knows about two hardcoded ids would need a code change every time
    a new hotel's history became available; that's exactly the kind of
    thing that should be discovered from the data instead.
    `RESERVATION_HOTELS` is kept only as a named constant for the
    take-home-specific analysis script (`compare_production.py`) that is
    deliberately about hotel_C/hotel_H by name — the actual pipeline
    (train.py, predict.py, this module) calls this function instead.
    """
    return sorted(reservations["hotel_id"].unique())


def realized_fraction_as_of(
    reservations: pd.DataFrame,
    hotel_id: str,
    room_type_code: str,
    stay_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
    total_rooms: float,
) -> float | None:
    """Exact fraction booked for one (hotel, room_type, stay_date) as of an
    arbitrary as_of_date, using the same covering/booked-by-cutoff rule as
    `build_actual_curve_table`. Returns None if `reservations` has no rows
    at all for this hotel — callers should fall back to the model in that
    case. (Zero *covering* rows for this specific stay_date is different
    and legitimate — that's a real "nobody's booked this night" and
    returns 0.0 below, not None.)
    """
    if hotel_id not in reservations["hotel_id"].values:
        return None
    sub = reservations[
        (reservations.hotel_id == hotel_id)
        & (reservations.room_type_code == room_type_code)
        & (reservations.stay_date <= stay_date)
        & (reservations.checkout_date > stay_date)
    ]
    if sub.empty:
        return 0.0
    booked = int((sub["booking_date"] <= as_of_date).sum())
    return float(min(booked / max(total_rooms, 1), 1.0))
