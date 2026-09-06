"""
Evaluation Script for Booking Curve Predictions
================================================

Compares predicted booking curves against actuals from the test set.

Usage:
    python evaluation/evaluate.py --predictions path/to/your_predictions.json

Your predictions JSON should be a list of objects:
[
    {
        "hotel_id": "hotel_C",
        "room_type_code": "rt_abc1234567",
        "stay_date": "2025-08-15",
        "predictions": {
            "90": 0.05,
            "60": 0.15,
            ...
            "0": 0.90
        }
    },
    ...
]
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

CHECKPOINTS = [90, 60, 45, 30, 21, 14, 7, 3, 1, 0]

# Weights: closer-to-stay predictions matter more for pricing decisions
CHECKPOINT_WEIGHTS = {
    90: 0.5,
    60: 0.6,
    45: 0.7,
    30: 0.8,
    21: 0.9,
    14: 1.0,
    7: 1.2,
    3: 1.3,
    1: 1.4,
    0: 1.5,
}

TEST_START = "2025-07-01"
TEST_END = "2025-09-30"


def build_actuals(data_dir):
    """Build actual booking curves from reservation data for the test period.

    A reservation contributes to every night it covers: a guest checking in
    June 13 and checking out June 16 occupies a room on nights June 13, 14,
    and 15. So for target_night = June 14, that reservation counts if
    stay_date <= June 14 AND checkout_date > June 14.
    """
    reservations = pd.read_csv(data_dir / "reservations.csv")
    with open(data_dir / "property_metadata.json") as f:
        metadata = json.load(f)

    active = reservations[reservations["status"] != "cancelled"].copy()
    active["booking_date"] = pd.to_datetime(active["booking_date"])
    active["stay_date"] = pd.to_datetime(active["stay_date"])
    active["checkout_date"] = pd.to_datetime(active["checkout_date"])

    # Generate all target nights in the test period
    test_nights = pd.date_range(TEST_START, TEST_END)

    actuals = {}
    for prop_id in active["hotel_id"].unique():
        prop_data = active[active["hotel_id"] == prop_id]

        for room_type_code in prop_data["room_type_code"].unique():
            rt_data = prop_data[prop_data["room_type_code"] == room_type_code]
            total_rooms = metadata[prop_id]["room_types"].get(room_type_code, {}).get(
                "inventory_count", 1
            )

            for target_night in test_nights:
                # All reservations covering this night
                covering = rt_data[
                    (rt_data["stay_date"] <= target_night)
                    & (rt_data["checkout_date"] > target_night)
                ]

                if len(covering) == 0:
                    continue

                curve = {}
                for cp in CHECKPOINTS:
                    cutoff = target_night - pd.Timedelta(days=cp)
                    booked = (covering["booking_date"] <= cutoff).sum()
                    curve[str(cp)] = min(booked / total_rooms, 1.0)

                key = (prop_id, room_type_code, target_night.strftime("%Y-%m-%d"))
                actuals[key] = curve

    return actuals


def evaluate(predictions, actuals):
    """Compute evaluation metrics."""
    results = {
        "total_predictions": len(predictions),
        "matched_predictions": 0,
        "unmatched_predictions": 0,
        "checkpoint_mae": {},
        "checkpoint_rmse": {},
        "overall_mae": 0.0,
        "weighted_mae": 0.0,
        "monotonicity_violations": 0,
        "bound_violations": 0,
        "by_property": {},
    }

    checkpoint_errors = defaultdict(list)
    weighted_errors = []
    property_errors = defaultdict(lambda: defaultdict(list))
    total_curves = 0

    for pred in predictions:
        key = (pred["hotel_id"], pred["room_type_code"], pred["stay_date"])

        if key not in actuals:
            results["unmatched_predictions"] += 1
            continue

        results["matched_predictions"] += 1
        total_curves += 1
        actual_curve = actuals[key]
        pred_curve = pred["predictions"]

        # Check monotonicity
        prev_val = -1
        for cp in sorted(CHECKPOINTS, reverse=True):  # 90 → 0
            val = pred_curve.get(str(cp), 0)
            if val < prev_val - 0.001:  # small tolerance
                results["monotonicity_violations"] += 1
                break
            prev_val = val

        # Check bounds
        for cp in CHECKPOINTS:
            val = pred_curve.get(str(cp), 0)
            if val < -0.01 or val > 1.01:
                results["bound_violations"] += 1
                break

        # Compute errors
        for cp in CHECKPOINTS:
            actual_val = actual_curve[str(cp)]
            pred_val = pred_curve.get(str(cp), 0)
            error = abs(actual_val - pred_val)
            checkpoint_errors[cp].append(error)
            weighted_errors.append(error * CHECKPOINT_WEIGHTS[cp])
            property_errors[pred["hotel_id"]][cp].append(error)

    # Aggregate metrics
    all_errors = []
    for cp in CHECKPOINTS:
        if checkpoint_errors[cp]:
            mae = np.mean(checkpoint_errors[cp])
            rmse = np.sqrt(np.mean(np.array(checkpoint_errors[cp]) ** 2))
            results["checkpoint_mae"][str(cp)] = round(mae, 6)
            results["checkpoint_rmse"][str(cp)] = round(rmse, 6)
            all_errors.extend(checkpoint_errors[cp])

    results["overall_mae"] = round(np.mean(all_errors), 6) if all_errors else 0
    results["weighted_mae"] = round(np.mean(weighted_errors), 6) if weighted_errors else 0
    results["monotonicity_violation_rate"] = (
        round(results["monotonicity_violations"] / total_curves, 4) if total_curves else 0
    )
    results["bound_violation_rate"] = (
        round(results["bound_violations"] / total_curves, 4) if total_curves else 0
    )

    # Per-property breakdown
    for prop_id, cp_errors in property_errors.items():
        prop_all = []
        prop_weighted = []
        prop_cp_mae = {}
        for cp in CHECKPOINTS:
            if cp_errors[cp]:
                prop_cp_mae[str(cp)] = round(np.mean(cp_errors[cp]), 6)
                prop_all.extend(cp_errors[cp])
                prop_weighted.extend(
                    [e * CHECKPOINT_WEIGHTS[cp] for e in cp_errors[cp]]
                )
        results["by_property"][prop_id] = {
            "overall_mae": round(np.mean(prop_all), 6) if prop_all else 0,
            "weighted_mae": round(np.mean(prop_weighted), 6) if prop_weighted else 0,
            "checkpoint_mae": prop_cp_mae,
        }

    return results


def print_results(results):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("  BOOKING CURVE PREDICTION EVALUATION")
    print("=" * 60)

    print(f"\n  Predictions evaluated: {results['matched_predictions']}")
    if results["unmatched_predictions"]:
        print(f"  Unmatched (skipped):   {results['unmatched_predictions']}")

    print(f"\n  {'Checkpoint':>12}  {'MAE':>8}  {'RMSE':>8}  {'Weight':>6}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*6}")
    for cp in CHECKPOINTS:
        cp_str = str(cp)
        mae = results["checkpoint_mae"].get(cp_str, 0)
        rmse = results["checkpoint_rmse"].get(cp_str, 0)
        w = CHECKPOINT_WEIGHTS[cp]
        print(f"  {cp:>9}d    {mae:.4f}    {rmse:.4f}    {w:.1f}x")

    print(f"\n  Overall MAE:           {results['overall_mae']:.4f}")
    print(f"  Weighted MAE:          {results['weighted_mae']:.4f}")
    print(f"  Monotonicity issues:   {results['monotonicity_violation_rate']:.1%}")
    print(f"  Bound violations:      {results['bound_violation_rate']:.1%}")

    if results["by_property"]:
        print(f"\n  Per-Property Breakdown:")
        for prop_id, prop_results in results["by_property"].items():
            print(f"    {prop_id}: MAE={prop_results['overall_mae']:.4f}  Weighted={prop_results['weighted_mae']:.4f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate booking curve predictions")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent.parent / "data"

    print("Building actual curves from test data...")
    actuals = build_actuals(data_dir)
    print(f"  {len(actuals)} actual curves in test period ({TEST_START} to {TEST_END})")

    print(f"\nLoading predictions from {args.predictions}...")
    with open(args.predictions) as f:
        predictions = json.load(f)
    print(f"  {len(predictions)} predictions loaded")

    print("\nEvaluating...")
    results = evaluate(predictions, actuals)
    print_results(results)

    # Save results
    output_path = args.output or Path(args.predictions).parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
