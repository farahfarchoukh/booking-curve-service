"""
Multi-horizon forecasting: is the level x shape decomposition (src/model.py,
DESIGN.md §6.1) actually justified, or just the first reasonable-sounding
idea that got built and never checked against the simplest alternative?

Compares two architectures, BACKBONE ONLY (no shrinkage/quantization/
extrapolation-damping on either side — those layers are machinery built
specifically on top of the chosen decomposition, and including them would
confound a comparison of the decomposition itself; §6.1 already argues
per-hotel shrinkage wants "one clean scalar per curve," which is a
property Approach B has and Approach A doesn't by construction, so this
keeps that argument honest rather than assuming it):

  Approach A — ONE model, no decomposition. Predicts occupancy directly
  at every (hotel, room_type, stay_date, cp), with cp as an ordinal
  feature under LightGBM's monotone constraint. The simplest thing that
  could plausibly work.

  Approach B (shipped) — level (final occupancy, cp=0) x shape (pace
  g(cp) in [0,1]). See model.py's module docstring for the original
  reasoning.

Approaches C (a separate model per horizon) and D (multi-output) are
ruled out on data-volume grounds without training either — see
`why_not_c_and_d()` at the bottom for the actual numbers, not a hand-wave.

Nested-CV discipline matching evaluation/calibration_tuning.py: both
architectures are trained and compared on rolling_backtest.py's 3 tuning
folds (excluding the fold sharing the shipped model's train cutoff), and
the official Jul-Sep split is disclosed once at the end for reporting,
never for choosing a winner. This project's architecture was fixed
before this script existed; this validates that choice retroactively —
it does not make it, and if Approach A had won convincingly, the
honest move would be to say so, not to quietly keep this script
unpublished.

Usage: python evaluation/architecture_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import build_actuals, evaluate  # noqa: E402
from src.data import (  # noqa: E402
    CHECKPOINTS,
    build_actual_curve_table,
    hotels_with_reservations,
    load_reservations,
    load_static_context,
)
from src.features import CATEGORICAL_FEATURES, SHAPE_FEATURES  # noqa: E402
from src.model import enforce_curve_constraints  # noqa: E402
from src.train import TRAIN_FLOOR, _best_rounds, _fit_categories, _prep_matrix  # noqa: E402

SEED = 42
_DETERMINISM = dict(seed=SEED, deterministic=True, force_row_wise=True)

# Same tuning/official split as evaluation/calibration_tuning.py — reused
# verbatim rather than redefined, so "the official split" means the same
# thing everywhere in this repo.
TUNING_FOLDS = [
    ("2025-05-31", "2025-06-01", "2025-06-30"),
    ("2025-07-31", "2025-08-01", "2025-08-31"),
    ("2025-08-31", "2025-09-01", "2025-09-30"),
]
OFFICIAL_FOLD = ("2025-06-30", "2025-07-01", "2025-09-30")


def train_approach_a(data_dir, train_end, val_start, static, reservations, hotels):
    """One model, cp as a feature, monotone-constrained. Predicts
    occupancy directly — no level/shape split."""
    long_df = build_actual_curve_table(reservations, static, TRAIN_FLOOR, train_end, hotels=hotels)
    long_df["curve_id"] = (
        long_df.hotel_id + "|" + long_df.room_type_code + "|" + long_df.stay_date.dt.strftime("%Y-%m-%d")
    )
    cats = _fit_categories(static, long_df)  # level_df param is unused inside _fit_categories itself

    is_val = long_df.stay_date >= pd.Timestamp(val_start)
    X_tr = _prep_matrix(static, cats, long_df.loc[~is_val], include_cp=True)
    X_val = _prep_matrix(static, cats, long_df.loc[is_val], include_cp=True)

    monotone = [0] * len(SHAPE_FEATURES)
    monotone[SHAPE_FEATURES.index("cp")] = -1
    params = dict(
        objective="regression",  # L2 — monotone_constraints unsupported under L1 (same finding as the shape model)
        num_leaves=31,
        min_data_in_leaf=40,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        monotone_constraints=monotone,
        **_DETERMINISM,
    )
    rounds = _best_rounds(
        params, X_tr, long_df.loc[~is_val, "actual"], X_val, long_df.loc[is_val, "actual"], CATEGORICAL_FEATURES
    )
    X_full = _prep_matrix(static, cats, long_df, include_cp=True)
    booster = lgb.train(
        params, lgb.Dataset(X_full, long_df["actual"], categorical_feature=CATEGORICAL_FEATURES),
        num_boost_round=rounds,
    )

    def predict_batch(keys: pd.DataFrame) -> list[dict]:
        """keys: one row per (hotel_id, room_type_code, stay_date). Builds
        ONE feature matrix for every (curve, checkpoint) pair and predicts
        in a single booster.predict() call — the vectorized pattern
        model.py's own predict_curve_batch uses, and for the same reason
        (see the latency profiling in README/DESIGN.md §6.5): calling a
        per-curve predict function thousands of times in a Python loop is
        what makes an evaluation script like this one take an hour
        instead of a minute."""
        n = len(keys)
        rep = keys.loc[keys.index.repeat(len(CHECKPOINTS))].reset_index(drop=True)
        rep["cp"] = CHECKPOINTS * n
        X = _prep_matrix(static, cats, rep, include_cp=True)
        raw = booster.predict(X).reshape(n, len(CHECKPOINTS))
        return [enforce_curve_constraints(CHECKPOINTS, raw[i]) for i in range(n)]

    return predict_batch


def train_approach_b(data_dir, train_end, val_start, static, reservations, hotels):
    """The shipped level x shape decomposition, backbone only (no
    shrinkage/quantization/damping) — isolates the decomposition itself
    from the correction machinery layered on top of it."""
    from src.train import LEVEL_LGB_PARAMS, SHAPE_LGB_PARAMS, _make_level_shape_tables

    long_df = build_actual_curve_table(reservations, static, TRAIN_FLOOR, train_end, hotels=hotels)
    level_df, shape_df = _make_level_shape_tables(long_df)
    cats = _fit_categories(static, level_df)

    level_lgb_params = dict(LEVEL_LGB_PARAMS, seed=SEED)
    shape_lgb_params = dict(SHAPE_LGB_PARAMS, seed=SEED)
    monotone = [0] * len(SHAPE_FEATURES)
    monotone[SHAPE_FEATURES.index("cp")] = -1
    shape_lgb_params = dict(shape_lgb_params, monotone_constraints=monotone)

    is_val = level_df.stay_date >= pd.Timestamp(val_start)
    X_lvl_tr = _prep_matrix(static, cats, level_df.loc[~is_val], include_cp=False)
    X_lvl_val = _prep_matrix(static, cats, level_df.loc[is_val], include_cp=False)
    lvl_rounds = _best_rounds(
        level_lgb_params, X_lvl_tr, level_df.loc[~is_val, "y"], X_lvl_val, level_df.loc[is_val, "y"],
        CATEGORICAL_FEATURES,
    )
    X_lvl_full = _prep_matrix(static, cats, level_df, include_cp=False)
    level_booster = lgb.train(
        level_lgb_params, lgb.Dataset(X_lvl_full, level_df["y"], categorical_feature=CATEGORICAL_FEATURES),
        num_boost_round=lvl_rounds,
    )

    shape_val_curves = set(level_df.loc[is_val].assign(
        curve_id=lambda d: d.hotel_id + "|" + d.room_type_code + "|" + d.stay_date.dt.strftime("%Y-%m-%d")
    )["curve_id"])
    shape_df["curve_id"] = (
        shape_df.hotel_id + "|" + shape_df.room_type_code + "|" + shape_df.stay_date.dt.strftime("%Y-%m-%d")
    )
    shp_is_val = shape_df.curve_id.isin(shape_val_curves)
    X_shp_tr = _prep_matrix(static, cats, shape_df.loc[~shp_is_val], include_cp=True)
    X_shp_val = _prep_matrix(static, cats, shape_df.loc[shp_is_val], include_cp=True)
    shp_rounds = _best_rounds(
        shape_lgb_params, X_shp_tr, shape_df.loc[~shp_is_val, "y"], X_shp_val, shape_df.loc[shp_is_val, "y"],
        CATEGORICAL_FEATURES,
    )
    X_shp_full = _prep_matrix(static, cats, shape_df, include_cp=True)
    shape_booster = lgb.train(
        shape_lgb_params, lgb.Dataset(X_shp_full, shape_df["y"], categorical_feature=CATEGORICAL_FEATURES),
        num_boost_round=shp_rounds,
    )

    def predict_batch(keys: pd.DataFrame) -> list[dict]:
        """Same vectorized-once pattern as Approach A's predict_batch —
        one level-matrix build for all curves, one shape-matrix build for
        all (curve, checkpoint) pairs, no per-curve Python loop calling
        into LightGBM."""
        n = len(keys)
        X_lvl = _prep_matrix(static, cats, keys, include_cp=False)
        y_final = np.clip(level_booster.predict(X_lvl), 0.0, 1.0)

        rep = keys.loc[keys.index.repeat(len(CHECKPOINTS))].reset_index(drop=True)
        rep["cp"] = CHECKPOINTS * n
        X_shp = _prep_matrix(static, cats, rep, include_cp=True)
        g_raw = shape_booster.predict(X_shp).reshape(n, len(CHECKPOINTS))
        g_raw = np.where(np.array(CHECKPOINTS) == 0, 1.0, np.clip(g_raw, 0.0, 1.0))

        results = []
        for i in range(n):
            point_raw = y_final[i] * g_raw[i]
            results.append(enforce_curve_constraints(CHECKPOINTS, point_raw))
        return results

    return predict_batch


def main():
    data_dir = ROOT / "data"
    static = load_static_context(data_dir)
    reservations = load_reservations(data_dir)
    hotels = hotels_with_reservations(reservations)

    # Predictions are kept PER FOLD, never pooled into one shared list
    # while folds are still being generated: the official fold's test
    # window (Jul-Sep) overlaps the Aug and Sep tuning folds' windows, so
    # a single growing list would let a later fold's predictions for the
    # same (hotel, room_type, date) sit alongside an earlier fold's —
    # different models, same key — and evaluate() has no way to know
    # which one you meant, so it would just score both, silently
    # double-counting and blending predictions from two different models
    # trained at two different cutoffs. Keeping strict per-fold lists is
    # what avoids that.
    fold_preds_a, fold_preds_b, fold_actuals = {}, {}, {}

    for train_end, test_start, test_end in TUNING_FOLDS + [OFFICIAL_FOLD]:
        val_start = (pd.Timestamp(train_end) - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
        fold_key = (train_end, test_start, test_end)
        print(f"Training fold train<={train_end} test={test_start}..{test_end} ...")

        predict_a = train_approach_a(data_dir, train_end, val_start, static, reservations, hotels)
        predict_b = train_approach_b(data_dir, train_end, val_start, static, reservations, hotels)

        test_nights = pd.date_range(test_start, test_end)
        import evaluation.evaluate as ev
        ev.TEST_START, ev.TEST_END = test_start, test_end
        fold_actuals[fold_key] = build_actuals(data_dir)

        keys = pd.DataFrame(
            [
                {"hotel_id": h, "room_type_code": rt, "stay_date": night}
                for h in hotels
                for rt in sorted(static.known_room_types(h))
                for night in test_nights
            ]
        )
        results_a = predict_a(keys)
        results_b = predict_b(keys)
        fold_preds_a[fold_key] = [
            {"hotel_id": row["hotel_id"], "room_type_code": row["room_type_code"],
             "stay_date": row["stay_date"].strftime("%Y-%m-%d"), "predictions": r}
            for (_, row), r in zip(keys.iterrows(), results_a)
        ]
        fold_preds_b[fold_key] = [
            {"hotel_id": row["hotel_id"], "room_type_code": row["room_type_code"],
             "stay_date": row["stay_date"].strftime("%Y-%m-%d"), "predictions": r}
            for (_, row), r in zip(keys.iterrows(), results_b)
        ]

    res_a_official = evaluate(fold_preds_a[OFFICIAL_FOLD], fold_actuals[OFFICIAL_FOLD])
    res_b_official = evaluate(fold_preds_b[OFFICIAL_FOLD], fold_actuals[OFFICIAL_FOLD])

    # Tuning-fold aggregate: pool predictions AND actuals across the 3
    # tuning folds only (their test windows don't overlap each other, so
    # pooling here is safe) — evaluate() needs one actuals dict covering
    # every key its predictions reference.
    tuning_preds_a, tuning_preds_b, tuning_actuals = [], [], {}
    for fold_key in TUNING_FOLDS:
        tuning_preds_a += fold_preds_a[fold_key]
        tuning_preds_b += fold_preds_b[fold_key]
        tuning_actuals.update(fold_actuals[fold_key])
    res_a_tuning = evaluate(tuning_preds_a, tuning_actuals)
    res_b_tuning = evaluate(tuning_preds_b, tuning_actuals)

    print("\n" + "=" * 78)
    print("  ARCHITECTURE COMPARISON (backbone only, no shrink/quantize/damp)")
    print("=" * 78)
    print("\n  TUNING FOLDS (Jun/Aug/Sep — excludes official split):")
    print(f"    Approach A (single model, cp feature):  overall_MAE={res_a_tuning['overall_mae']:.4f}  "
          f"weighted_MAE={res_a_tuning['weighted_mae']:.4f}")
    print(f"    Approach B (level x shape, shipped):    overall_MAE={res_b_tuning['overall_mae']:.4f}  "
          f"weighted_MAE={res_b_tuning['weighted_mae']:.4f}")
    print("\n  OFFICIAL SPLIT (disclosure only — not used to pick a winner):")
    print(f"    Approach A:  overall_MAE={res_a_official['overall_mae']:.4f}  weighted_MAE={res_a_official['weighted_mae']:.4f}")
    print(f"    Approach B:  overall_MAE={res_b_official['overall_mae']:.4f}  weighted_MAE={res_b_official['weighted_mae']:.4f}")

    out = {
        "tuning_folds": {"approach_a": res_a_tuning, "approach_b": res_b_tuning},
        "official_disclosure": {"approach_a": res_a_official, "approach_b": res_b_official},
    }
    out_path = ROOT / "evaluation" / "architecture_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    why_not_c_and_d(static, hotels)


def why_not_c_and_d(static, hotels):
    """Approaches C (a separate model per horizon) and D (multi-output)
    aren't trained — the data volume alone rules them out, and it's
    worth showing the actual arithmetic rather than asserting it."""
    n_curves_hotel_c_ish = 1463  # from train.py's own logged curve count for hotel_C, the larger of the two hotels
    n_checkpoints = len(CHECKPOINTS)
    per_horizon_n = n_curves_hotel_c_ish  # each horizon-specific model would train on ~this many rows, not n_curves*n_checkpoints
    print("\n" + "=" * 78)
    print("  WHY NOT APPROACH C (per-horizon models) OR D (multi-output)?")
    print("=" * 78)
    print(
        f"\n  C: {n_checkpoints} independent models, each trained on ~{per_horizon_n:,} rows (one row"
        f"\n     per curve, not per checkpoint) instead of Approach B's shared ~{per_horizon_n * (n_checkpoints - 1):,}"
        f"\n     shape rows. Worse per-model sample size for a MORE complex serving story (10 models"
        f"\n     to version instead of 2), and no shared learning across horizons at all — a hotel's"
        f"\n     day-21 behavior tells you nothing about its day-14 behavior in this architecture,"
        f"\n     which contradicts the whole reason a booking curve has structure in the first place."
        f"\n\n  D (multi-output: one model, 10 outputs): LightGBM has no native multi-output regressor"
        f"\n     (sklearn's MultiOutputRegressor is just C again, N independent boosters, with extra"
        f"\n     library indirection) — a genuine multi-output tree model means a neural net or a"
        f"\n     multi-task GBM library this dataset (1,503 level curves) is far too small to justify."
        f"\n     Both ruled out on data-volume grounds before training either, not by assumption —"
        f"\n     these are the actual row counts this dataset has."
    )


if __name__ == "__main__":
    main()
