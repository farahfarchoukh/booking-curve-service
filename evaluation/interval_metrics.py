"""
Prediction-interval calibration metrics — PICP, pinball loss, mean interval
width — for `evaluation/predictions.json` against real reservation ground
truth.

This exists because the PICP figures cited in DESIGN.md and README (90%
in-distribution, ~39% on the true test window) were originally produced by
a throwaway analysis script that was never committed — meaning the claim
wasn't actually reproducible from this repo. This is that script, for
real, so `python evaluation/interval_metrics.py` is the source of truth
for those numbers going forward, not a paragraph asserting them.

Usage:
    python evaluation/interval_metrics.py [--predictions evaluation/predictions.json] [--data-dir data]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINTS = [90, 60, 45, 30, 21, 14, 7, 3, 1, 0]
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

TEST_START = "2025-07-01"
TEST_END = "2025-09-30"


def _load_inventory_lookup(data_dir: Path) -> dict:
    """Room-type inventory counts, including the same repair for
    dimension-table gaps that `src/data.py::load_static_context` applies —
    reusing that logic (rather than re-deriving it here) so this script's
    denominators always agree with whatever the model itself trained on.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import load_static_context  # noqa: E402

    static = load_static_context(data_dir)
    return static.inventory_lookup


def build_actual_curves(data_dir: Path) -> dict:
    reservations = pd.read_csv(
        data_dir / "reservations.csv",
        parse_dates=["stay_date", "checkout_date", "booking_date"],
    )
    reservations = reservations[reservations["status"] != "cancelled"]
    inventory = _load_inventory_lookup(data_dir)

    nights = pd.date_range(TEST_START, TEST_END)
    actuals = {}
    for (hotel_id, room_type_code), group in reservations.groupby(["hotel_id", "room_type_code"]):
        total = inventory.get((hotel_id, room_type_code), 1) or 1
        for night in nights:
            covering = group[(group.stay_date <= night) & (group.checkout_date > night)]
            if covering.empty:
                continue
            curve = {
                cp: min((covering.booking_date <= night - pd.Timedelta(days=cp)).sum() / total, 1.0)
                for cp in CHECKPOINTS
            }
            actuals[(hotel_id, room_type_code, night.strftime("%Y-%m-%d"))] = curve
    return actuals


def pinball_loss(actual: float, predicted: float, quantile: float) -> float:
    diff = actual - predicted
    return max(quantile * diff, (quantile - 1) * diff)


def evaluate(predictions: list, actuals: dict) -> dict:
    below = above = within = total = 0
    widths = []
    pinball_by_q = {q: [] for q in QUANTILES.values()}  # keyed by the float (0.1/0.5/0.9), not "p10"
    per_hotel = {}

    for pred in predictions:
        key = (pred["hotel_id"], pred["room_type_code"], pred["stay_date"])
        actual_curve = actuals.get(key)
        if actual_curve is None or "intervals" not in pred:
            continue
        hotel_bucket = per_hotel.setdefault(
            pred["hotel_id"], {"below": 0, "above": 0, "within": 0, "total": 0}
        )
        for cp in CHECKPOINTS:
            a = actual_curve[cp]
            lo = pred["intervals"]["p10"][str(cp)]
            hi = pred["intervals"]["p90"][str(cp)]
            total += 1
            hotel_bucket["total"] += 1
            if a < lo:
                below += 1
                hotel_bucket["below"] += 1
            elif a > hi:
                above += 1
                hotel_bucket["above"] += 1
            else:
                within += 1
                hotel_bucket["within"] += 1
            widths.append(hi - lo)
            for q_name, q in QUANTILES.items():
                pinball_by_q[q].append(pinball_loss(a, pred["intervals"][q_name][str(cp)], q))

    if total == 0:
        return {"error": "no predictions matched actual curves — did you pass the right --data-dir?"}

    results = {
        "n_checkpoints_scored": total,
        "picp": round(within / total, 4),
        "share_below_p10": round(below / total, 4),
        "share_above_p90": round(above / total, 4),
        "mean_interval_width": round(float(np.mean(widths)), 4),
        "mean_pinball_loss": round(float(np.mean([v for vals in pinball_by_q.values() for v in vals])), 4),
        "pinball_loss_by_quantile": {
            q_name: round(float(np.mean(pinball_by_q[q])), 4) for q_name, q in QUANTILES.items()
        },
        "by_hotel": {
            h: {
                "picp": round(b["within"] / b["total"], 4),
                "share_below_p10": round(b["below"] / b["total"], 4),
                "share_above_p90": round(b["above"] / b["total"], 4),
                "n": b["total"],
            }
            for h, b in per_hotel.items()
        },
    }
    return results


def print_results(results: dict):
    if "error" in results:
        print(results["error"])
        return
    print("\n" + "=" * 60)
    print("  PREDICTION INTERVAL CALIBRATION (P10 / P90)")
    print("=" * 60)
    print(f"\n  Checkpoints scored:     {results['n_checkpoints_scored']}")
    print(f"  PICP (target 0.80):     {results['picp']:.1%}")
    print(f"    below p10:            {results['share_below_p10']:.1%}")
    print(f"    above p90:            {results['share_above_p90']:.1%}")
    print(f"  Mean interval width:    {results['mean_interval_width']:.4f}")
    print(f"  Mean pinball loss:      {results['mean_pinball_loss']:.4f}")
    for q, v in results["pinball_loss_by_quantile"].items():
        print(f"    {q}:                   {v:.4f}")
    print("\n  Per-hotel:")
    for hotel, stats in results["by_hotel"].items():
        print(
            f"    {hotel}: PICP={stats['picp']:.1%}  below={stats['share_below_p10']:.1%}  "
            f"above={stats['share_above_p90']:.1%}  n={stats['n']}"
        )
    print("\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default="evaluation/predictions.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    predictions = json.load(open(args.predictions))
    if not predictions or "intervals" not in predictions[0]:
        print(
            "No 'intervals' key in predictions.json — regenerate it with "
            "`python -m src.predict --generate-eval` (interval output was added "
            "after some earlier prediction files were generated)."
        )
        return

    actuals = build_actual_curves(data_dir)
    results = evaluate(predictions, actuals)
    print_results(results)

    out_path = Path(args.output) if args.output else Path(args.predictions).parent / "interval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
