"""
Head-to-head comparison against Ampliphi's own production parametric curve
(`expected_booking_curves.csv`), restricted to the one (hotel, room_type)
combination where that comparison is actually possible: hotel_C's base
("optimized") room type, rt_ea30c05c4c — the only room type
`expected_booking_curves` covers for hotel_C (README §4 / DATA_DICTIONARY
both note this table covers hotel_B/C/F for a single room type each).

Usage: python evaluation/compare_production.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = [90, 60, 45, 30, 21, 14, 7, 3, 1, 0]
WEIGHTS = {90: 0.5, 60: 0.6, 45: 0.7, 30: 0.8, 21: 0.9, 14: 1.0, 7: 1.2, 3: 1.3, 1: 1.4, 0: 1.5}

HOTEL, ROOM = "hotel_C", "rt_ea30c05c4c"


def build_actuals():
    res = pd.read_csv(ROOT / "data" / "reservations.csv", parse_dates=["stay_date", "checkout_date", "booking_date"])
    with open(ROOT / "data" / "property_metadata.json") as f:
        meta = json.load(f)
    total = meta[HOTEL]["room_types"][ROOM]["inventory_count"]
    sub = res[(res.hotel_id == HOTEL) & (res.room_type_code == ROOM) & (res.status != "cancelled")]
    nights = pd.date_range("2025-07-01", "2025-09-30")
    out = {}
    for night in nights:
        covering = sub[(sub.stay_date <= night) & (sub.checkout_date > night)]
        if covering.empty:
            continue
        curve = {}
        for cp in CHECKPOINTS:
            cutoff = night - pd.Timedelta(days=cp)
            curve[cp] = min((covering.booking_date <= cutoff).sum() / total, 1.0)
        out[night.strftime("%Y-%m-%d")] = curve
    return out, total


def score(pred_by_date, actuals, label):
    errs = {cp: [] for cp in CHECKPOINTS}
    for date, actual in actuals.items():
        if date not in pred_by_date:
            continue
        for cp in CHECKPOINTS:
            errs[cp].append(abs(actual[cp] - pred_by_date[date][cp]))
    overall = np.mean([e for v in errs.values() for e in v])
    weighted = np.mean([e * WEIGHTS[cp] for cp, v in errs.items() for e in v])
    n = len(next(iter(errs.values())))
    print(f"{label:28s}  n={n:4d}  overall_MAE={overall:.4f}  weighted_MAE={weighted:.4f}")
    return overall, weighted


def main():
    actuals, total = build_actuals()

    # our model
    our_preds = json.load(open(ROOT / "evaluation" / "predictions.json"))
    our_by_date = {
        p["stay_date"]: {int(k): v for k, v in p["predictions"].items()}
        for p in our_preds
        if p["hotel_id"] == HOTEL and p["room_type_code"] == ROOM
    }

    # heuristic baseline (starter/baseline_model.py output, if present alongside this data dir)
    baseline_path = ROOT.parent / "ampliphi-ml-takehome-v3" / "evaluation" / "baseline_predictions.json"
    base_by_date = {}
    if baseline_path.exists():
        base_preds = json.load(open(baseline_path))
        base_by_date = {
            p["stay_date"]: {int(k): v for k, v in p["predictions"].items()}
            for p in base_preds
            if p["hotel_id"] == HOTEL and p["room_type_code"] == ROOM
        }

    # Ampliphi production parametric curve
    ebc = pd.read_csv(ROOT / "data" / "expected_booking_curves.csv")
    ebc = ebc[(ebc.hotel_id == HOTEL) & (ebc.room_type_code == ROOM)]
    prod_by_date = {}
    for date, grp in ebc.groupby("stay_date"):
        curve = {int(r.days_until_stay): min(r.expected_occupancy / total, 1.0) for r in grp.itertuples()}
        if set(CHECKPOINTS).issubset(curve.keys()):
            prod_by_date[date] = curve

    print(f"\nHead-to-head on {HOTEL}/{ROOM} (inventory={total}), test window Jul-Sep 2025:\n")
    score(our_preds and our_by_date, actuals, "Our model")
    if base_by_date:
        score(base_by_date, actuals, "Heuristic baseline")
    if prod_by_date:
        score(prod_by_date, actuals, "Ampliphi expected_booking_curves")
    else:
        print("Ampliphi expected_booking_curves      no dates with all 10 checkpoints present")


if __name__ == "__main__":
    main()
