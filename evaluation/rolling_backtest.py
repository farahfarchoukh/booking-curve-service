"""
Every other evaluation in this repo scores ONE train/test split: train on
everything through June 30, test on Jul-Sep. That's a single sample of
"how does this pipeline do when trained up to some date and asked about
the following months" — and it's exactly the split that turned out to sit
entirely outside the training season (see model.py's
_extrapolation_correction_damp docstring). A single split can't tell us
whether the error level and the beats-baseline margin are stable, or
whether we just got a favorable (or unfavorable) three months.

This retrains at four expanding-window walk-forward cutoffs and scores
each on the following month, using data.py's own construction rules the
whole way through (same TRAIN_FLOOR-anchored table, same seed) — nothing
here changes the shipped model or is used to pick a train_end; the
shipped model still trains on everything through 2025-06-30. This is
purely a robustness read on that choice.

Usage: python evaluation/rolling_backtest.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (train_end, test_start, test_end) — each fold's test month immediately
# follows its train cutoff, expanding-window (train floor is always fixed
# at data.TRAIN_FLOOR = 2025-01-01, only the cutoff moves). Stops at
# Aug 31 -> Sep test because Sep 30 is the last stay_date in this dataset.
FOLDS = [
    ("2025-05-31", "2025-06-01", "2025-06-30"),
    ("2025-06-30", "2025-07-01", "2025-07-31"),
    ("2025-07-31", "2025-08-01", "2025-08-31"),
    ("2025-08-31", "2025-09-01", "2025-09-30"),
]


def main():
    # Runs everything in-process (not via the CLI scripts): run_training,
    # predict_curve_batch and evaluate() all need a per-fold window
    # override that isn't (deliberately) a CLI flag on the shipped
    # scripts — those stay matched to what the shipped pipeline actually
    # runs, and this reuses their underlying functions directly instead.
    import pandas as pd

    from evaluation.evaluate import build_actuals, evaluate
    from src.data import hotels_with_reservations, load_reservations, load_static_context
    from src.model import BookingCurveModel
    from src.registry import resolve_model_dir
    from src.train import run_training

    data_dir = ROOT / "data"
    workdir = ROOT / "artifacts" / "_rolling_backtest_scratch"
    workdir.mkdir(parents=True, exist_ok=True)

    static = load_static_context(data_dir)
    reservations = load_reservations(data_dir)
    hotels = hotels_with_reservations(reservations)

    rows = []
    for train_end, test_start, test_end in FOLDS:
        print(f"Fold: train <= {train_end}, test {test_start}..{test_end}")
        model_base = workdir / f"model_{train_end}"
        run_training(data_dir, model_base, version="v", promote=True, seed=42, train_end=train_end)
        model_dir = resolve_model_dir(model_base, None)
        model = BookingCurveModel.load(model_dir)

        test_nights = pd.date_range(test_start, test_end)
        keys = pd.DataFrame(
            [
                {"hotel_id": h, "room_type_code": rt, "stay_date": night}
                for h in hotels
                for rt in sorted(static.known_room_types(h))
                for night in test_nights
            ]
        )
        results = model.predict_curve_batch(static, keys)
        preds = [
            {
                "hotel_id": row["hotel_id"],
                "room_type_code": row["room_type_code"],
                "stay_date": row["stay_date"].strftime("%Y-%m-%d"),
                "predictions": r["point"],
            }
            for (_, row), r in zip(keys.iterrows(), results)
        ]

        import evaluation.evaluate as ev
        ev.TEST_START, ev.TEST_END = test_start, test_end
        actuals = build_actuals(data_dir)
        res = evaluate(preds, actuals)
        rows.append({
            "train_end": train_end,
            "test_window": f"{test_start}..{test_end}",
            "n_matched": res["matched_predictions"],
            "overall_mae": res["overall_mae"],
            "weighted_mae": res["weighted_mae"],
            "hotel_C_mae": res["by_property"].get("hotel_C", {}).get("overall_mae"),
            "hotel_H_mae": res["by_property"].get("hotel_H", {}).get("overall_mae"),
        })

    print("\n" + "=" * 78)
    print("  ROLLING (WALK-FORWARD, EXPANDING-WINDOW) BACKTEST")
    print("=" * 78)
    print(f"\n  {'train_end':>11}  {'test_window':>22}  {'n':>5}  {'overall':>8}  {'weighted':>9}  "
          f"{'hotel_C':>8}  {'hotel_H':>8}")
    for r in rows:
        print(f"  {r['train_end']:>11}  {r['test_window']:>22}  {r['n_matched']:>5}  "
              f"{r['overall_mae']:>8.4f}  {r['weighted_mae']:>9.4f}  "
              f"{(r['hotel_C_mae'] or 0):>8.4f}  {(r['hotel_H_mae'] or 0):>8.4f}")
    print()

    out_path = ROOT / "evaluation" / "rolling_backtest.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Results saved to {out_path}")

    shutil.rmtree(workdir, ignore_errors=True)
    print(f"Cleaned up scratch artifacts under {workdir}")


if __name__ == "__main__":
    main()
