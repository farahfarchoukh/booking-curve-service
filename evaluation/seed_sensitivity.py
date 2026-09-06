"""
Determinism (src/train.py's _DETERMINISM block) guarantees the SAME seed
reproduces the SAME model, bit-for-bit. It says nothing about whether the
story we're telling — "the two-stage model with shrinkage beats the
baseline by about this much" — is an artifact of having happened to draw
seed=42, or holds up under any other arbitrary seed. This retrains the
full pipeline under several different seeds (each a genuinely separate
subprocess — no shared module state) and reports how much the headline
numbers actually move.

This is a robustness check, not a hyperparameter search: no seed here is
chosen or preferred based on its test-set score, and the shipped
artifact keeps using the module default (SEED=42 in src/train.py). Test
data touched only for reporting, not selection.

Usage: python evaluation/seed_sensitivity.py [--seeds 7 42 123 2024 90210]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [42, 7, 123, 2024, 90210]


def train_and_eval(seed: int, workdir: Path) -> dict:
    model_dir = workdir / f"model_seed{seed}"
    preds_path = workdir / f"preds_seed{seed}.json"
    results_path = workdir / f"results_seed{seed}.json"

    subprocess.run(
        [sys.executable, "-m", "src.train", "--seed", str(seed), "--out", str(model_dir)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "src.predict", "--generate-eval",
         "--model-dir", str(model_dir), "--out", str(preds_path)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, "evaluation/evaluate.py",
         "--predictions", str(preds_path), "--output", str(results_path)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    res = json.load(open(results_path))
    return {
        "seed": seed,
        "overall_mae": res["overall_mae"],
        "weighted_mae": res["weighted_mae"],
        "hotel_C_mae": res["by_property"].get("hotel_C", {}).get("overall_mae"),
        "hotel_H_mae": res["by_property"].get("hotel_H", {}).get("overall_mae"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--keep-artifacts", action="store_true", help="Don't delete per-seed models/predictions after")
    args = ap.parse_args()

    workdir = ROOT / "artifacts" / "_seed_sensitivity_scratch"
    workdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        print(f"Training + evaluating seed={seed} ...")
        rows.append(train_and_eval(seed, workdir))

    weighted = np.array([r["weighted_mae"] for r in rows])
    overall = np.array([r["overall_mae"] for r in rows])

    print("\n" + "=" * 70)
    print("  SEED SENSITIVITY")
    print("=" * 70)
    print(f"\n  {'seed':>8}  {'overall_mae':>12}  {'weighted_mae':>13}  {'hotel_C':>9}  {'hotel_H':>9}")
    for r in rows:
        print(f"  {r['seed']:>8}  {r['overall_mae']:>12.4f}  {r['weighted_mae']:>13.4f}  "
              f"{r['hotel_C_mae']:>9.4f}  {r['hotel_H_mae']:>9.4f}")

    summary = {
        "seeds": rows,
        "weighted_mae_mean": float(weighted.mean()),
        "weighted_mae_std": float(weighted.std(ddof=1)),
        "weighted_mae_range": [float(weighted.min()), float(weighted.max())],
        "overall_mae_mean": float(overall.mean()),
        "overall_mae_std": float(overall.std(ddof=1)),
        "overall_mae_range": [float(overall.min()), float(overall.max())],
    }
    print(f"\n  weighted_mae: mean={summary['weighted_mae_mean']:.4f}  "
          f"std={summary['weighted_mae_std']:.4f}  range={summary['weighted_mae_range']}")
    print(f"  overall_mae:  mean={summary['overall_mae_mean']:.4f}  "
          f"std={summary['overall_mae_std']:.4f}  range={summary['overall_mae_range']}")

    # rough read: is the spread across seeds small relative to the margin
    # we found over the baseline in significance_test.py? Load it if present.
    sig_path = ROOT / "evaluation" / "significance_test.json"
    if sig_path.exists():
        sig = json.load(open(sig_path))
        margin = abs(sig["point_estimate_diff"])
        spread = summary["weighted_mae_range"][1] - summary["weighted_mae_range"][0]
        print(f"\n  Seed-to-seed weighted_mae spread: {spread:.4f}")
        print(f"  Margin over baseline (from significance_test.py): {margin:.4f}")
        if spread < margin:
            print("  -> Seed noise is smaller than the margin over baseline: the")
            print("     'we beat the baseline' conclusion is not seed-dependent.")
        else:
            print("  -> Seed noise is comparable to (or larger than) the margin over")
            print("     baseline — treat the beats-baseline claim with more caution.")

    out_path = ROOT / "evaluation" / "seed_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if not args.keep_artifacts:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"Cleaned up scratch artifacts under {workdir}")


if __name__ == "__main__":
    main()
