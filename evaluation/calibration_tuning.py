"""
EXTRAPOLATION_GAMMA and INTERVAL_WIDEN_K (src/train.py) have been an
honest but unvalidated prior since they were introduced: "widen by this
much per 30 days outside the training season" and "widen by this much
per sqrt(hotel_n)" are order-of-magnitude guesses, never actually checked
against held-out coverage. This searches for values that do better,
using the walk-forward folds from evaluation/rolling_backtest.py as
nested cross-validation — NOT the official Jul-Sep test split.

That split matters methodologically: fold 2 of rolling_backtest.py
(train <= 2025-06-30, test = Jul) uses the exact train cutoff the shipped
model ships with, and Jul-Sep is the window every other number in this
repo is reported against. Tuning against it would be the identical
mistake already caught once this project (see VALIDATION.md §6.8's
MIN_ROOM_N_FOR_POOLING). So this tunes ONLY on folds 1/3/4 (Jun/Aug/Sep,
train cutoffs May 31 / Jul 31 / Aug 31) and reports the winning
configuration's performance on the official split afterward, once, for
disclosure — never as a selection criterion.

Objective is mean pinball loss (a proper scoring rule that penalizes
both wrong AND needlessly wide intervals), pooled across all three
tuning folds' checkpoints — not PICP directly, which a search could game
by just making everything wider. PICP is reported as a diagnostic.

Two-stage search (not one big grid): first extrapolation_gamma with
interval_widen_k held at its current default, then interval_widen_k with
gamma fixed at whatever won stage 1. A small number of deliberately
chosen candidates, not a fine grid — with only 3 tuning folds, a wide
search invites overfitting the *search* to those 3 folds, which is a
smaller version of the same problem this script exists to avoid.

Usage: python evaluation/calibration_tuning.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.interval_metrics import build_actual_curves, evaluate as eval_intervals  # noqa: E402
from src.data import hotels_with_reservations, load_reservations, load_static_context  # noqa: E402
from src.model import BookingCurveModel  # noqa: E402
from src.registry import resolve_model_dir  # noqa: E402
from src.train import INTERVAL_WIDEN_K as DEFAULT_WIDEN_K  # noqa: E402
from src.train import run_training  # noqa: E402

# Excludes (2025-06-30, Jul-01..Jul-31) on purpose — see module docstring.
TUNING_FOLDS = [
    ("2025-05-31", "2025-06-01", "2025-06-30"),
    ("2025-07-31", "2025-08-01", "2025-08-31"),
    ("2025-08-31", "2025-09-01", "2025-09-30"),
]
# The split every other number in this repo is reported against — used
# ONLY to disclose the winning config's performance, never to pick it.
OFFICIAL_FOLD = ("2025-06-30", "2025-07-01", "2025-09-30")

GAMMA_CANDIDATES = [0.0, 1.0, 2.0, 3.0, 5.0]
WIDEN_K_CANDIDATES = [1.0, 1.5, 2.5, 4.0]


def score_config(gamma: float, widen_k: float, folds, static, hotels, data_dir, workdir) -> dict:
    """Trains one model per fold at this (gamma, widen_k), scores each
    fold's held-out month, and returns interval metrics pooled across all
    folds' checkpoints (not averaged fold-by-fold — pooling is correct
    here since fold sizes differ)."""
    all_preds, all_actuals = [], {}
    for train_end, test_start, test_end in folds:
        model_base = workdir / f"g{gamma}_k{widen_k}_{train_end}"
        run_training(
            data_dir, model_base, version="v", promote=True, seed=42,
            train_end=train_end, extrapolation_gamma=gamma, interval_widen_k=widen_k,
        )
        model = BookingCurveModel.load(resolve_model_dir(model_base, None))

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
        for (_, row), r in zip(keys.iterrows(), results):
            all_preds.append({
                "hotel_id": row["hotel_id"],
                "room_type_code": row["room_type_code"],
                "stay_date": row["stay_date"].strftime("%Y-%m-%d"),
                "intervals": {"p10": r["p10"], "p50": r["p50"], "p90": r["p90"]},
            })

        import evaluation.interval_metrics as im
        im.TEST_START, im.TEST_END = test_start, test_end
        all_actuals.update(build_actual_curves(data_dir))

        shutil.rmtree(model_base, ignore_errors=True)

    metrics = eval_intervals(all_preds, all_actuals)
    return metrics


def main():
    data_dir = ROOT / "data"
    workdir = ROOT / "artifacts" / "_calibration_tuning_scratch"
    workdir.mkdir(parents=True, exist_ok=True)

    static = load_static_context(data_dir)
    reservations = load_reservations(data_dir)
    hotels = hotels_with_reservations(reservations)

    print("=" * 78)
    print("  STAGE 1: extrapolation_gamma (interval_widen_k held at current default"
          f" {DEFAULT_WIDEN_K})")
    print("=" * 78)
    stage1 = []
    for gamma in GAMMA_CANDIDATES:
        m = score_config(gamma, DEFAULT_WIDEN_K, TUNING_FOLDS, static, hotels, data_dir, workdir)
        print(f"  gamma={gamma:>4}: PICP={m['picp']:.1%}  pinball={m['mean_pinball_loss']:.4f}  "
              f"width={m['mean_interval_width']:.4f}")
        stage1.append({"gamma": gamma, **m})
    best_gamma = min(stage1, key=lambda r: r["mean_pinball_loss"])["gamma"]
    print(f"\n  -> best gamma by pooled mean pinball loss: {best_gamma}")

    print("\n" + "=" * 78)
    print(f"  STAGE 2: interval_widen_k (extrapolation_gamma fixed at {best_gamma})")
    print("=" * 78)
    stage2 = []
    for k in WIDEN_K_CANDIDATES:
        m = score_config(best_gamma, k, TUNING_FOLDS, static, hotels, data_dir, workdir)
        print(f"  widen_k={k:>4}: PICP={m['picp']:.1%}  pinball={m['mean_pinball_loss']:.4f}  "
              f"width={m['mean_interval_width']:.4f}")
        stage2.append({"widen_k": k, **m})
    best_k = min(stage2, key=lambda r: r["mean_pinball_loss"])["widen_k"]
    print(f"\n  -> best interval_widen_k by pooled mean pinball loss: {best_k}")

    print("\n" + "=" * 78)
    print(f"  DISCLOSURE ONLY: chosen config (gamma={best_gamma}, widen_k={best_k}) on the")
    print("  OFFICIAL Jul-Sep split — not used to select anything above")
    print("=" * 78)
    official = score_config(best_gamma, best_k, [OFFICIAL_FOLD], static, hotels, data_dir, workdir)
    baseline_official = score_config(1.0, DEFAULT_WIDEN_K, [OFFICIAL_FOLD], static, hotels, data_dir, workdir)
    print(f"  chosen  (gamma={best_gamma}, k={best_k}):  PICP={official['picp']:.1%}  "
          f"pinball={official['mean_pinball_loss']:.4f}  width={official['mean_interval_width']:.4f}")
    print(f"  current (gamma=1.0, k={DEFAULT_WIDEN_K}):  PICP={baseline_official['picp']:.1%}  "
          f"pinball={baseline_official['mean_pinball_loss']:.4f}  width={baseline_official['mean_interval_width']:.4f}")

    out = {
        "stage1_gamma_search": stage1,
        "best_gamma": best_gamma,
        "stage2_widen_k_search": stage2,
        "best_widen_k": best_k,
        "official_split_disclosure": {"chosen": official, "current_shipped": baseline_official},
    }
    out_path = ROOT / "evaluation" / "calibration_tuning.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
