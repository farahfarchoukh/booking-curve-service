"""
evaluation/calibration_tuning.py searched for a single EXTRAPOLATION_GAMMA
and never converged — pinball loss kept improving to the edge of the
tested range. The diagnosis (recorded in that script's output and
DESIGN.md §6.11): the widening mechanism was symmetric (P10 and P90 widen
by the same factor) while the actual miscalibration is one-sided (actuals
miss high far more than low), so a shared knob has to over-widen the low
side just to stretch the high side far enough — a structurally inefficient
fix that never stops "improving" because it's still under-covering on the
side that matters.

This searches the now-decoupled `extrapolation_gamma_lo`/`_hi`
(model.py's asymmetric widening) instead, on the same nested-CV folds and
with the same discipline as calibration_tuning.py: tuned only on 3
rolling folds excluding the official split, official split disclosed
once at the end for reporting, never for selection.

Two-stage search, same reasoning as calibration_tuning.py for keeping it
small: gamma_hi first (the side actually carrying evidence of
miscalibration, searched over a wider range since the single-gamma
search's problem was specifically running out of room on this side),
then gamma_lo with gamma_hi fixed at the stage-1 winner. If gamma_hi
*still* doesn't converge to an interior optimum even decoupled, that's
reported as plainly as the first non-convergence was — this script does
not force a number just because the previous one couldn't be shipped
either.

Usage: python evaluation/asymmetric_calibration_tuning.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.interval_metrics import build_actual_curves  # noqa: E402
from evaluation.interval_metrics import evaluate as eval_intervals  # noqa: E402
from src.data import hotels_with_reservations, load_reservations, load_static_context  # noqa: E402
from src.model import BookingCurveModel  # noqa: E402
from src.registry import resolve_model_dir  # noqa: E402
from src.train import INTERVAL_WIDEN_K as DEFAULT_WIDEN_K  # noqa: E402
from src.train import run_training  # noqa: E402

TUNING_FOLDS = [
    ("2025-05-31", "2025-06-01", "2025-06-30"),
    ("2025-07-31", "2025-08-01", "2025-08-31"),
    ("2025-08-31", "2025-09-01", "2025-09-30"),
]
OFFICIAL_FOLD = ("2025-06-30", "2025-07-01", "2025-09-30")

# gamma_lo held modest a priori (the low side was never the problem);
# gamma_hi searched over a wider range than the old single-gamma sweep
# (0-5) specifically because that sweep ran out of room on this side.
GAMMA_LO_DEFAULT = 0.5
GAMMA_HI_CANDIDATES = [1.0, 2.0, 4.0, 8.0, 16.0]
GAMMA_LO_CANDIDATES = [0.0, 0.5, 1.0, 2.0]


def score_config(gamma_lo: float, gamma_hi: float, folds, static, hotels, data_dir, workdir) -> dict:
    all_preds, all_actuals = [], {}
    for train_end, test_start, test_end in folds:
        model_base = workdir / f"lo{gamma_lo}_hi{gamma_hi}_{train_end}"
        run_training(
            data_dir, model_base, version="v", promote=True, seed=42,
            train_end=train_end, interval_widen_k=DEFAULT_WIDEN_K,
            extrapolation_gamma_lo=gamma_lo, extrapolation_gamma_hi=gamma_hi,
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

    return eval_intervals(all_preds, all_actuals)


def _is_interior(candidates, best):
    """True if the winner isn't sitting at either edge of its own
    candidate list — the thing the single-gamma search failed at."""
    return candidates[0] < best < candidates[-1]


def main():
    data_dir = ROOT / "data"
    workdir = ROOT / "artifacts" / "_asym_calibration_tuning_scratch"
    workdir.mkdir(parents=True, exist_ok=True)

    static = load_static_context(data_dir)
    reservations = load_reservations(data_dir)
    hotels = hotels_with_reservations(reservations)

    print("=" * 78)
    print(f"  STAGE 1: extrapolation_gamma_hi (gamma_lo held at {GAMMA_LO_DEFAULT})")
    print("=" * 78)
    stage1 = []
    for gamma_hi in GAMMA_HI_CANDIDATES:
        m = score_config(GAMMA_LO_DEFAULT, gamma_hi, TUNING_FOLDS, static, hotels, data_dir, workdir)
        print(f"  gamma_hi={gamma_hi:>5}: PICP={m['picp']:.1%}  pinball={m['mean_pinball_loss']:.4f}  "
              f"width={m['mean_interval_width']:.4f}  above_p90={m['share_above_p90']:.1%}")
        stage1.append({"gamma_hi": gamma_hi, **m})
    best_hi = min(stage1, key=lambda r: r["mean_pinball_loss"])["gamma_hi"]
    converged_hi = _is_interior(GAMMA_HI_CANDIDATES, best_hi)
    print(f"\n  -> best gamma_hi by pooled pinball loss: {best_hi} "
          f"({'interior optimum — converged' if converged_hi else 'AT THE EDGE — did not converge'})")

    print("\n" + "=" * 78)
    print(f"  STAGE 2: extrapolation_gamma_lo (gamma_hi fixed at {best_hi})")
    print("=" * 78)
    stage2 = []
    for gamma_lo in GAMMA_LO_CANDIDATES:
        m = score_config(gamma_lo, best_hi, TUNING_FOLDS, static, hotels, data_dir, workdir)
        print(f"  gamma_lo={gamma_lo:>5}: PICP={m['picp']:.1%}  pinball={m['mean_pinball_loss']:.4f}  "
              f"width={m['mean_interval_width']:.4f}  below_p10={m['share_below_p10']:.1%}")
        stage2.append({"gamma_lo": gamma_lo, **m})
    best_lo = min(stage2, key=lambda r: r["mean_pinball_loss"])["gamma_lo"]
    converged_lo = _is_interior(GAMMA_LO_CANDIDATES, best_lo)
    print(f"\n  -> best gamma_lo by pooled pinball loss: {best_lo} "
          f"({'interior optimum — converged' if converged_lo else 'at the edge — did not converge'})")

    result = {
        "stage1_gamma_hi_search": stage1,
        "best_gamma_hi": best_hi,
        "gamma_hi_converged": converged_hi,
        "stage2_gamma_lo_search": stage2,
        "best_gamma_lo": best_lo,
        "gamma_lo_converged": converged_lo,
    }

    if converged_hi and converged_lo:
        print("\n" + "=" * 78)
        print(f"  DISCLOSURE ONLY: chosen config (lo={best_lo}, hi={best_hi}) on the")
        print("  OFFICIAL Jul-Sep split — not used to select anything above")
        print("=" * 78)
        official = score_config(best_lo, best_hi, [OFFICIAL_FOLD], static, hotels, data_dir, workdir)
        current = score_config(1.0, 1.0, [OFFICIAL_FOLD], static, hotels, data_dir, workdir)
        print(f"  chosen  (lo={best_lo}, hi={best_hi}):  PICP={official['picp']:.1%}  "
              f"pinball={official['mean_pinball_loss']:.4f}  width={official['mean_interval_width']:.4f}")
        print(f"  current (lo=1.0, hi=1.0, symmetric):  PICP={current['picp']:.1%}  "
              f"pinball={current['mean_pinball_loss']:.4f}  width={current['mean_interval_width']:.4f}")
        result["official_split_disclosure"] = {"chosen": official, "current_shipped": current}
        result["recommendation"] = "SHIP — both sides converged to an interior optimum on nested CV"
    else:
        print("\n" + "=" * 78)
        print("  NOT SHIPPING: at least one side did not converge to an interior optimum.")
        print("  Reporting the search honestly rather than picking an edge-of-grid value.")
        print("=" * 78)
        result["recommendation"] = "DO NOT SHIP — see gamma_hi_converged/gamma_lo_converged"

    out_path = ROOT / "evaluation" / "asymmetric_calibration_tuning.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")

    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
