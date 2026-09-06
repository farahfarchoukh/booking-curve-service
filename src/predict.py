"""
predict_booking_curve(...) — the inference contract — plus a CLI for single
lookups and for generating evaluation/predictions.json.

as_of_date semantics (see DESIGN.md and README for the full write-up):

  as_of_date=None (default): "blind" forecast. No live pickup signal is
  used; every checkpoint comes straight from the model. This is the mode
  used to generate evaluation/predictions.json, so that our model, the
  heuristic baseline, and Ampliphi's `expected_booking_curves` are compared
  on equal footing (none of them get to see realized pickup for the stay
  being scored).

  as_of_date=<date>: production mode. Let dus_now = (stay_date -
  as_of_date).days.
    - Checkpoints at or before as_of_date (cp >= dus_now) are already
      history: we return the *exact* realized fraction booked by that date
      (computed straight from reservations — no model call, no error).
      This is available in this dataset only for hotel_C/hotel_H; in
      production it would come from the live reservations table for any
      onboarded hotel.
    - Checkpoints still in the future (cp < dus_now) are model-forecast,
      then rescaled so the curve is continuous with the realized anchor at
      dus_now and still lands on the model's forecast final occupancy
      (or higher, if more is already on the books than the model expected).
  This is what makes the same function usable both to seed a brand-new
  stay's curve 90 days out and to refresh it daily as real bookings land.

Every code path funnels through BookingCurveModel.predict_curve (see
model.py), so this file adds *only* as_of_date handling and I/O — it does
not re-implement feature engineering or constraint enforcement.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    CHECKPOINTS,
    TEST_END,
    TEST_START,
    hotels_with_reservations,
    load_reservations,
    load_static_context,
    realized_fraction_as_of,
)
from .logging_config import get_logger
from .model import BookingCurveModel, enforce_curve_constraints
from .registry import resolve_model_dir

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
_CP_ASC = sorted(CHECKPOINTS)  # [0,1,3,...,90] — ascending, for interpolation


@lru_cache(maxsize=1)
def _context(data_dir: str, model_base_dir: str, model_version: str | None):
    """Cached per (data_dir, model_base_dir, model_version) triple. The
    version is part of the cache key on purpose: if a caller pins a
    different version mid-process (e.g. a canary comparison script loading
    two versions back to back), each gets its own cached load rather than
    silently reusing the wrong one.
    """
    resolved = resolve_model_dir(Path(model_base_dir), model_version)
    log.info(f"Loading model version from {resolved}")
    static = load_static_context(Path(data_dir))
    reservations = load_reservations(Path(data_dir))
    model = BookingCurveModel.load(resolved)
    return static, reservations, model


def _default_paths():
    return str(REPO_ROOT / "data"), str(REPO_ROOT / "artifacts" / "model")


def predict_booking_curve(
    hotel_id: str,
    room_type_code: str,
    stay_date: str,
    as_of_date: str | None = None,
    *,
    data_dir: str | None = None,
    model_dir: str | None = None,
    model_version: str | None = None,
) -> dict:
    """`model_dir` is the *base* directory of versioned artifacts
    (default: artifacts/model). Which version actually loads is resolved
    by `model_version` (pin one explicitly — the rollback lever) or, if
    not given, the `BOOKING_CURVE_MODEL_VERSION` env var, or else
    whatever `current.json` under `model_dir` points at. See
    src/registry.py.
    """
    default_data, default_model = _default_paths()
    static, reservations, model = _context(
        data_dir or default_data, model_dir or default_model, model_version
    )

    stay_ts = pd.Timestamp(stay_date)
    room_total = static.inventory_lookup.get((hotel_id, room_type_code))
    if room_total is None:
        # Unknown (hotel, room_type) pair: fall back to hotel-level total or 1.
        room_total = static.total_rooms_lookup.get(hotel_id, 1) or 1

    blind = model.predict_curve(static, hotel_id, room_type_code, stay_ts)
    point = blind["point"]

    if as_of_date is None:
        result_curve = point
        anchor_info = {"mode": "blind"}
    else:
        as_of_ts = pd.Timestamp(as_of_date)
        dus_now = (stay_ts - as_of_ts).days

        if dus_now >= max(CHECKPOINTS):
            result_curve = point
            anchor_info = {"mode": "blind", "dus_now": dus_now}
        else:
            # realized_fraction_as_of itself returns None when `reservations`
            # has no rows at all for this hotel — no separate hardcoded
            # hotel-list check needed here (there used to be one; it silently
            # broke this path for any hotel other than hotel_C/hotel_H).
            anchor_frac = realized_fraction_as_of(
                reservations, hotel_id, room_type_code, stay_ts, as_of_ts, room_total
            )

            if anchor_frac is None:
                result_curve = point
                anchor_info = {"mode": "blind_no_ground_truth", "dus_now": dus_now}
            else:
                y_asc = np.array([point[str(c)] for c in _CP_ASC])
                model_final = point["0"]
                model_at_anchor = float(np.interp(dus_now, _CP_ASC, y_asc))
                effective_final = max(model_final, anchor_frac)
                denom = max(effective_final - model_at_anchor, 1e-6)

                merged = {}
                for cp in CHECKPOINTS:
                    if cp >= dus_now:
                        calendar_date = stay_ts - pd.Timedelta(days=cp)
                        merged[str(cp)] = realized_fraction_as_of(
                            reservations, hotel_id, room_type_code, stay_ts,
                            calendar_date, room_total,
                        )
                    else:
                        remaining_growth = (point[str(cp)] - model_at_anchor) / denom
                        remaining_growth = float(np.clip(remaining_growth, 0.0, 1.0))
                        merged[str(cp)] = anchor_frac + remaining_growth * (
                            effective_final - anchor_frac
                        )
                result_curve = enforce_curve_constraints(
                    CHECKPOINTS, [merged[str(c)] for c in CHECKPOINTS]
                )
                anchor_info = {
                    "mode": "anchored",
                    "dus_now": dus_now,
                    "anchor_frac": anchor_frac,
                    "model_final_hat": model_final,
                }

    return {
        "hotel_id": hotel_id,
        "room_type_code": room_type_code,
        "stay_date": stay_ts.strftime("%Y-%m-%d"),
        "predictions": {k: result_curve[k] for k in map(str, CHECKPOINTS)},
        "intervals": {"p10": blind["p10"], "p50": blind["p50"], "p90": blind["p90"]},
        "diagnostics": {**blind["diagnostics"], **anchor_info},
    }


def generate_eval_predictions(
    data_dir: Path, model_base_dir: Path, out_path: Path, model_version: str | None = None
):
    """Blind (as_of_date=None) predictions for the full test-window grid —
    every (room_type, night) for every hotel that has reservation history
    in `data_dir` (hotel_C and hotel_H, in this dataset — determined from
    the data, not hardcoded). Vectorized via
    BookingCurveModel.predict_curve_batch for throughput (a few thousand
    curves would otherwise mean tens of thousands of individual LightGBM
    calls through the per-row `predict_booking_curve` path).
    """
    static = load_static_context(data_dir)
    reservations = load_reservations(data_dir)
    reservation_hotels = hotels_with_reservations(reservations)
    resolved = resolve_model_dir(model_base_dir, model_version)
    log.info(f"Loading model version from {resolved}")
    log.info(f"Generating predictions for hotels with reservation history: {reservation_hotels}")
    model = BookingCurveModel.load(resolved)
    nights = pd.date_range(TEST_START, TEST_END)

    keys = pd.DataFrame(
        [
            {"hotel_id": h, "room_type_code": rt, "stay_date": night}
            for h in reservation_hotels
            for rt in sorted(static.known_room_types(h))
            for night in nights
        ]
    )
    results = model.predict_curve_batch(static, keys)

    records = []
    for (_, row), res in zip(keys.iterrows(), results):
        records.append(
            {
                "hotel_id": row["hotel_id"],
                "room_type_code": row["room_type_code"],
                "stay_date": row["stay_date"].strftime("%Y-%m-%d"),
                "predictions": res["point"],
                "intervals": {"p10": res["p10"], "p50": res["p50"], "p90": res["p90"]},
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    log.info(f"Wrote {len(records)} predictions to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotel-id")
    ap.add_argument("--room-type-code")
    ap.add_argument("--stay-date")
    ap.add_argument("--as-of-date", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model-dir", default=None, help="Base directory of versioned artifacts")
    ap.add_argument(
        "--model-version",
        default=None,
        help="Pin a specific version instead of resolving current.json (rollback lever).",
    )
    ap.add_argument("--generate-eval", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    default_data, default_model = _default_paths()
    data_dir = Path(args.data_dir) if args.data_dir else Path(default_data)
    model_dir = Path(args.model_dir) if args.model_dir else Path(default_model)

    if args.generate_eval:
        out_path = Path(args.out) if args.out else REPO_ROOT / "evaluation" / "predictions.json"
        generate_eval_predictions(data_dir, model_dir, out_path, model_version=args.model_version)
        return

    if not (args.hotel_id and args.room_type_code and args.stay_date):
        ap.error("--hotel-id, --room-type-code and --stay-date are required unless --generate-eval")

    result = predict_booking_curve(
        args.hotel_id,
        args.room_type_code,
        args.stay_date,
        as_of_date=args.as_of_date,
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        model_version=args.model_version,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
