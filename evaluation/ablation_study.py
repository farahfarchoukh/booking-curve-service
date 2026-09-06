"""
Does each piece of the architecture actually earn its keep, or are we
carrying complexity that doesn't pay for itself? Four variants, same
trained boosters throughout (isolates each layer's serving-time
contribution instead of confounding it with retraining noise across
separate models):

  A. backbone only        — raw booster output, no correction layers at all
  B. + hotel shrink        — the original per-hotel empirical-Bayes correction
  C. + room-type shrink    — adds the nested (hotel, room_type) correction
  D. + inventory quantization (current shipped model) — adds physical
     rounding for tiny-inventory room types

Each variant is scored on the same held-out Jul-Sep test set. This is
reporting, not tuning — none of these variants is a candidate being
selected by its test score; the architecture was already decided from
first-principles reasoning (see DESIGN.md / model.py docstrings) before
this script existed. What this validates is whether that reasoning
actually shows up in the numbers.

Usage: python evaluation/ablation_study.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import model as model_mod  # noqa: E402
from src.data import hotels_with_reservations, load_reservations, load_static_context  # noqa: E402
from src.predict import _default_paths  # noqa: E402
from src.registry import resolve_model_dir  # noqa: E402

sys.path.insert(0, str(ROOT / "evaluation"))
from evaluate import build_actuals, evaluate  # noqa: E402


def make_variant(base, *, hotel_shrink: bool, room_shrink: bool):
    m = copy.copy(base)
    if not hotel_shrink:
        m.level_shrink = {}
        m.shape_shrink = {}
    if not room_shrink:
        m.room_level_shrink = {}
        m.room_shape_shrink = {}
    return m


def predict_all(model, static, keys_df, quantize: bool):
    orig = model_mod.MAX_INVENTORY_FOR_QUANTIZATION
    if not quantize:
        model_mod.MAX_INVENTORY_FOR_QUANTIZATION = -1  # nothing has inventory <= -1: hard off-switch
    try:
        results = model.predict_curve_batch(static, keys_df)
    finally:
        model_mod.MAX_INVENTORY_FOR_QUANTIZATION = orig
    return results


def main():
    data_dir, model_base_dir = _default_paths()
    data_dir, model_base_dir = Path(data_dir), Path(model_base_dir)
    model_dir = resolve_model_dir(model_base_dir, None)
    base_model = model_mod.BookingCurveModel.load(model_dir)

    static = load_static_context(data_dir)
    reservations = load_reservations(data_dir)
    hotels = hotels_with_reservations(reservations)

    import pandas as pd

    test_nights = pd.date_range("2025-07-01", "2025-09-30")
    rows = []
    for h in hotels:
        for rt in static.known_room_types(h):
            for night in test_nights:
                rows.append({"hotel_id": h, "room_type_code": rt, "stay_date": night})
    keys_df = pd.DataFrame(rows)

    actuals = build_actuals(data_dir)

    variants = [
        ("A. backbone only", dict(hotel_shrink=False, room_shrink=False, quantize=False)),
        ("B. + hotel shrink", dict(hotel_shrink=True, room_shrink=False, quantize=False)),
        ("C. + room-type shrink", dict(hotel_shrink=True, room_shrink=True, quantize=False)),
        ("D. + quantization (shipped)", dict(hotel_shrink=True, room_shrink=True, quantize=True)),
    ]

    print("\n" + "=" * 74)
    print("  ABLATION: contribution of each correction layer (same boosters throughout)")
    print("=" * 74)
    print(f"\n  {'variant':32s}  {'overall_mae':>11}  {'weighted_mae':>13}  {'hotel_C':>9}  {'hotel_H':>9}")

    summary = []
    for label, cfg in variants:
        m = make_variant(base_model, hotel_shrink=cfg["hotel_shrink"], room_shrink=cfg["room_shrink"])
        results = predict_all(m, static, keys_df, cfg["quantize"])
        preds = [
            {
                "hotel_id": row["hotel_id"],
                "room_type_code": row["room_type_code"],
                "stay_date": row["stay_date"].strftime("%Y-%m-%d"),
                "predictions": r["point"],
            }
            for row, r in zip(rows, results)
        ]
        res = evaluate(preds, actuals)
        print(
            f"  {label:32s}  {res['overall_mae']:>11.4f}  {res['weighted_mae']:>13.4f}  "
            f"{res['by_property'].get('hotel_C', {}).get('overall_mae', float('nan')):>9.4f}  "
            f"{res['by_property'].get('hotel_H', {}).get('overall_mae', float('nan')):>9.4f}"
        )
        summary.append({"variant": label, "overall_mae": res["overall_mae"], "weighted_mae": res["weighted_mae"],
                         "hotel_C_mae": res["by_property"].get("hotel_C", {}).get("overall_mae"),
                         "hotel_H_mae": res["by_property"].get("hotel_H", {}).get("overall_mae")})

    print()
    out_path = ROOT / "evaluation" / "ablation_study.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
