"""
Feature importance for the shipped level and shape boosters — doesn't
exist anywhere in this repo until now. For a Pricing Intelligence
audience, "why does the model think this" is often as load-bearing as
the number itself, and DESIGN.md §6.1 has been asserting a specific claim
("with only 2 fitted hotels, country/currency are collinear with
region_type/pms_type/primary_rate_mode, the latter three get zero gain")
without a checked-in script backing it — same class of gap
interval_metrics.py closed for the PICP numbers. This is that script.

Reports both gain-based (how much a feature's splits reduced loss —
what "importance" usually means) and split-count-based (how often a
feature was used at all — catches a feature LightGBM leans on
constantly for small, low-gain adjustments, which pure gain can hide)
importance for both stages, since the two occasionally disagree and the
disagreement itself is informative.

Usage: python evaluation/feature_importance.py [--model-dir ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import BookingCurveModel  # noqa: E402
from src.predict import _default_paths  # noqa: E402
from src.registry import resolve_model_dir  # noqa: E402


def importances(booster, feature_names: list) -> list:
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    gain_pct = gain / gain.sum() * 100 if gain.sum() > 0 else gain
    rows = [
        {"feature": f, "gain_pct": float(g), "split_count": int(s)}
        for f, g, s in zip(feature_names, gain_pct, split)
    ]
    return sorted(rows, key=lambda r: -r["gain_pct"])


def print_table(title: str, rows: list):
    print(f"\n{title}")
    print(f"  {'feature':22s}  {'gain %':>8s}  {'split count':>11s}")
    for r in rows:
        flag = "  <- zero gain" if r["gain_pct"] == 0 else ""
        print(f"  {r['feature']:22s}  {r['gain_pct']:>7.2f}%  {r['split_count']:>11d}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="Base dir of versioned artifacts (default: artifacts/model)")
    ap.add_argument("--model-version", default=None)
    args = ap.parse_args()

    _, default_model_dir = _default_paths()
    model_base = Path(args.model_dir) if args.model_dir else Path(default_model_dir)
    model_dir = resolve_model_dir(model_base, args.model_version)
    model = BookingCurveModel.load(model_dir)

    level_rows = importances(model.level_booster, list(model.level_booster.feature_name()))
    shape_rows = importances(model.shape_booster, list(model.shape_booster.feature_name()))

    print("=" * 70)
    print(f"  FEATURE IMPORTANCE — {model_dir.name}")
    print("=" * 70)
    print_table("LEVEL model (final occupancy):", level_rows)
    print_table("SHAPE model (booking pace):", shape_rows)

    zero_gain_level = [r["feature"] for r in level_rows if r["gain_pct"] == 0]
    zero_gain_shape = [r["feature"] for r in shape_rows if r["gain_pct"] == 0]
    print(
        f"\nZero-gain features — LEVEL: {zero_gain_level or 'none'}\n"
        f"Zero-gain features — SHAPE: {zero_gain_shape or 'none'}"
    )
    print(
        "\nDESIGN.md §6.1 claims pms_type/region_type/primary_rate_mode get zero"
        " gain (collinear with country/currency at only 2 fitted hotels). This"
        " is the checked-in confirmation of that claim, not an assertion."
    )

    out = {
        "model_version": model_dir.name,
        "level_importance": level_rows,
        "shape_importance": shape_rows,
        "zero_gain_level": zero_gain_level,
        "zero_gain_shape": zero_gain_shape,
    }
    out_path = ROOT / "evaluation" / "feature_importance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
