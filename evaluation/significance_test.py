"""
Is "we beat the heuristic baseline" actually a statistically defensible
claim, or could it be noise from a small test set?

Naively bootstrapping the 2,521 individual (curve, checkpoint) errors
would overstate our confidence: those errors are not independent draws.
Every checkpoint within one curve shares that curve's own model residual,
and every curve within one (hotel_id, room_type_code) shares that room
type's own booking pattern and seasonality. The real sample size for "how
many independent looks at the data do we get" is much closer to the ~23
distinct (hotel, room_type) series in the test period than to 2,521.

So this does a cluster bootstrap at the (hotel_id, room_type_code) level:
each resample redraws whole room-type series with replacement (not
individual curves or checkpoints), which is the standard fix for
panel/grouped data and won't manufacture false confidence out of
within-cluster correlation. (The test period actually touches more
distinct room types than the training window does — some appear for the
first time in Jul-Sep — so the cluster count here is somewhat higher than
the training room-type count; the script reports the real number it
found rather than assuming one.)

Usage: python evaluation/significance_test.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = [90, 60, 45, 30, 21, 14, 7, 3, 1, 0]
CHECKPOINT_WEIGHTS = {90: 0.5, 60: 0.6, 45: 0.7, 30: 0.8, 21: 0.9, 14: 1.0, 7: 1.2, 3: 1.3, 1: 1.4, 0: 1.5}
TEST_START, TEST_END = "2025-07-01", "2025-09-30"
N_BOOTSTRAP = 10_000
SEED = 42


def build_actuals(data_dir: Path) -> dict:
    reservations = pd.read_csv(data_dir / "reservations.csv")
    with open(data_dir / "property_metadata.json") as f:
        metadata = json.load(f)
    active = reservations[reservations["status"] != "cancelled"].copy()
    for c in ("booking_date", "stay_date", "checkout_date"):
        active[c] = pd.to_datetime(active[c])

    test_nights = pd.date_range(TEST_START, TEST_END)
    actuals = {}
    for prop_id in active["hotel_id"].unique():
        prop_data = active[active["hotel_id"] == prop_id]
        for room_type_code in prop_data["room_type_code"].unique():
            rt_data = prop_data[prop_data["room_type_code"] == room_type_code]
            total_rooms = metadata[prop_id]["room_types"].get(room_type_code, {}).get("inventory_count", 1)
            for night in test_nights:
                covering = rt_data[(rt_data["stay_date"] <= night) & (rt_data["checkout_date"] > night)]
                if len(covering) == 0:
                    continue
                curve = {}
                for cp in CHECKPOINTS:
                    cutoff = night - pd.Timedelta(days=cp)
                    booked = (covering["booking_date"] <= cutoff).sum()
                    curve[cp] = min(booked / total_rooms, 1.0)
                actuals[(prop_id, room_type_code, night.strftime("%Y-%m-%d"))] = curve
    return actuals


def curve_errors(preds: list, actuals: dict) -> dict:
    """Returns {(hotel, room_type, date): weighted_mae_of_that_curve}."""
    out = {}
    for p in preds:
        key = (p["hotel_id"], p["room_type_code"], p["stay_date"])
        if key not in actuals:
            continue
        actual = actuals[key]
        pred = p["predictions"]
        errs = [abs(actual[cp] - pred.get(str(cp), 0)) * CHECKPOINT_WEIGHTS[cp] for cp in CHECKPOINTS]
        out[key] = float(np.mean(errs))
    return out


def cluster_bootstrap_diff(ours: dict, baseline: dict, n_boot: int, seed: int) -> dict:
    """Both dicts keyed by (hotel, room_type, date) -> curve error, already
    restricted to keys present in both. Resamples whole (hotel, room_type)
    clusters with replacement; within a resampled cluster, ALL its curves
    are included (not itself resampled) — the cluster, not the curve, is
    the unit of resampling, which is what makes this valid for correlated
    within-cluster errors."""
    common = sorted(set(ours) & set(baseline))
    clusters = defaultdict(list)
    for key in common:
        clusters[(key[0], key[1])].append(key)
    cluster_ids = sorted(clusters.keys())
    rng = np.random.default_rng(seed)

    point_ours = np.mean([ours[k] for k in common])
    point_base = np.mean([baseline[k] for k in common])
    point_diff = point_ours - point_base

    boot_diffs = np.empty(n_boot)
    n_clusters = len(cluster_ids)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        o_errs, b_errs = [], []
        for idx in draw:
            for key in clusters[cluster_ids[idx]]:
                o_errs.append(ours[key])
                b_errs.append(baseline[key])
        boot_diffs[b] = np.mean(o_errs) - np.mean(b_errs)

    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    # one-sided: what fraction of bootstrap resamples show us NOT beating
    # baseline (diff >= 0)? That fraction is the p-value for "ours is
    # better," under the null that there's no true difference.
    p_value = float(np.mean(boot_diffs >= 0))

    return {
        "n_clusters": n_clusters,
        "n_curves": len(common),
        "point_estimate_diff": float(point_diff),
        "our_weighted_mae": float(point_ours),
        "baseline_weighted_mae": float(point_base),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "p_value_ours_better": p_value,
    }


def main():
    data_dir = ROOT / "data"
    actuals = build_actuals(data_dir)

    our_preds = json.load(open(ROOT / "evaluation" / "predictions.json"))

    candidates = []
    if os.environ.get("BASELINE_PREDICTIONS_PATH"):
        candidates.append(Path(os.environ["BASELINE_PREDICTIONS_PATH"]))
    candidates += [
        ROOT.parent / "ampliphi-ml-takehome-v3" / "evaluation" / "baseline_predictions.json",
        ROOT.parent / "aspire-takehome" / "ampliphi-ml-takehome-v3" / "evaluation" / "baseline_predictions.json",
    ]
    baseline_path = next((p for p in candidates if p.exists()), None)
    if baseline_path is None:
        raise SystemExit(
            "Couldn't find baseline_predictions.json (set BASELINE_PREDICTIONS_PATH). "
            "Generate it with the take-home starter's baseline_model.py first."
        )
    baseline_preds = json.load(open(baseline_path))

    ours = curve_errors(our_preds, actuals)
    base = curve_errors(baseline_preds, actuals)

    result = cluster_bootstrap_diff(ours, base, N_BOOTSTRAP, SEED)

    print("\n" + "=" * 64)
    print("  CLUSTER BOOTSTRAP: does our model beat the heuristic baseline?")
    print("  (cluster = one (hotel_id, room_type_code) series; N=%d clusters," % result["n_clusters"])
    print("   %d curves, resampled %d times)" % (result["n_curves"], N_BOOTSTRAP))
    print("=" * 64)
    print(f"\n  Our weighted MAE:       {result['our_weighted_mae']:.4f}")
    print(f"  Baseline weighted MAE:  {result['baseline_weighted_mae']:.4f}")
    print(f"  Difference (ours - baseline): {result['point_estimate_diff']:.4f}")
    print(f"  95% bootstrap CI on the difference: [{result['ci95_low']:.4f}, {result['ci95_high']:.4f}]")
    print(f"  p-value (H0: no true difference, one-sided 'ours is better'): {result['p_value_ours_better']:.4f}")
    if result["ci95_high"] < 0:
        print(f"\n  -> The CI excludes zero: at cluster granularity ({result['n_clusters']} room-type")
        print(f"     series, not {result['n_curves']} checkpoints), beating the heuristic baseline")
        print("     is NOT just noise from a small test set.")
    else:
        print("\n  -> The CI does not exclude zero at this cluster count — treat the")
        print("     'beats baseline' claim as suggestive, not statistically settled.")
    print()

    out_path = ROOT / "evaluation" / "significance_test.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
