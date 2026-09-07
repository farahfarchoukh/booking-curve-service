"""
CLI: build curve tables -> tune -> fit final two-stage model -> save artifact.

Usage:
    python src/train.py [--data-dir DATA_DIR] [--out artifacts/model]

What this does, in order:
  1. Build the long-format actual-curve table for hotel_C/hotel_H, train
     window only (stay_date <= 2025-06-30), via src/data.py — the exact same
     construction rule the grader uses for ground truth.
  2. Split off the last 3 weeks of the train window as an internal
     time-based validation slice, purely to pick a boosting round count via
     early stopping (never used to fit anything that touches test data).
  3. Refit final "production" boosters on the *entire* train window at that
     round count.
  4. Run grouped 5-fold CV (grouped by curve, i.e. by
     (hotel, room_type, stay_date)) purely to get honest out-of-fold
     residuals per hotel, which become the per-hotel shrinkage correction
     (see model.py's docstring for why this stands in for a full
     hierarchical model).
  5. Fit quantile boosters (P10/P50/P90) on the level target for prediction
     intervals.
  6. Save everything to artifacts/model/, and print an internal validation
     report (this is NOT the official grading metric — run
     evaluation/evaluate.py on evaluation/predictions.json for that).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .data import (
    TRAIN_END,
    build_actual_curve_table,
    hotels_with_reservations,
    load_reservations,
    load_static_context,
)
from .features import CATEGORICAL_FEATURES, LEVEL_FEATURES, SHAPE_FEATURES
from .logging_config import get_logger
from .model import BookingCurveModel, room_key
from .registry import new_version, save_pointer

log = get_logger(__name__)

VAL_START = "2025-06-09"  # last ~3 weeks of train window, for round-count tuning only
TRAIN_FLOOR = "2025-01-01"  # earliest plausible stay date in the whole dataset

LEVEL_SHRINK_K = 15.0
SHAPE_SHRINK_K = 30.0
INTERVAL_WIDEN_K = 1.5
# Second (finer) pooling level: (hotel, room_type) correction fit on the
# residual left over after the per-hotel correction above — see
# model.py's docstring for why room types within a hotel aren't
# exchangeable. Larger K than the hotel level on purpose: a single room
# type's curve count is always a subset of its hotel's, so this shrinks
# harder by construction already; the larger K on top of that is a
# deliberate extra brake, since a second correction layered on a first is
# more exposed to overfitting noise in a small dataset than the first
# layer was on its own.
ROOM_LEVEL_SHRINK_K = 20.0
ROOM_SHAPE_SHRINK_K = 40.0
# A floor below which a room type's own OOF residual mean isn't a
# meaningfully identifiable second level at all — the shrink weight
# n/(n+K) discounts it smoothly, but "smoothly small" is not the same
# question as "reliably estimated." A room type seen on a handful of
# curves (hotel_H's thinnest types: 1, 1, 2, 7 curves in this dataset)
# has an OOF residual mean with enough of its own variance that even a
# heavily shrunk correction can move the point estimate in the wrong
# direction more often than not; a hard floor says "with this little of
# its own history, fall back to the hotel-level correction alone,
# exactly as if this room type had never been split out." Chosen from
# looking at this dataset's own room-type curve-count *distribution*
# (there's a real gap between hotel_C's room types, all 76-93 curves,
# and hotel_H's, all <=15) — not by tuning against the July-Sept test
# set, which would be leakage for a hyperparameter choice.
MIN_ROOM_N_FOR_POOLING = 20
# Extra interval inflation per 30 days of stay-date sitting outside the
# observed training day-of-year range (see model.py._extrapolation_widen).
# Chosen as a modest, order-of-magnitude prior for "uncertainty grows the
# further we extrapolate outside the season we've trained on" — NOT fit
# against Jul-Sep test outcomes. See README/DESIGN.md §6.7 for why this
# only partially closes the coverage gap: it widens around a median that
# can still be biased low for an unseen season, which a symmetric interval
# can't fully repair.
EXTRAPOLATION_GAMMA = 1.0

# A single source of truth for "the seed" — the DEFAULT one, used unless
# run_training(..., seed=...) or `--seed` overrides it (see
# evaluation/seed_sensitivity.py, which retrains as a subprocess per seed
# so there's no import-order/module-global-mutation footgun). Overriding it
# changes both the LightGBM training seed AND the OOF KFold split below, so
# "seed sensitivity" means "what if the whole pipeline had drawn a
# different arbitrary seed," not just the boosters in isolation.
SEED = 42

# seed + deterministic + force_row_wise: without these, two runs on
# identical data can pick different split ties under multi-threaded
# histogram building, which is exactly what happened during development —
# retraining without any code or data change moved weighted MAE by ~0.01
# and shuffled which of two near-tied features got the split. That's a real
# problem for a production model: "retrain" should be reproducible enough
# that CI can build an image and know it's bit-for-bit the model it tested,
# and a rollback comparison isn't chasing training noise. Determinism
# (bit-for-bit reproducibility under the SAME seed) and seed sensitivity
# (whether conclusions hold under a DIFFERENT seed) are different
# questions — see evaluation/seed_sensitivity.py for the latter.
_DETERMINISM = dict(seed=SEED, deterministic=True, force_row_wise=True)

LEVEL_LGB_PARAMS = dict(
    objective="regression_l1",
    num_leaves=15,
    min_data_in_leaf=20,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbosity=-1,
    **_DETERMINISM,
)

SHAPE_LGB_PARAMS = dict(
    objective="regression",  # L2; LightGBM monotone_constraints unsupported under regression_l1
    num_leaves=31,
    min_data_in_leaf=40,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbosity=-1,
    **_DETERMINISM,
)


def _make_level_shape_tables(long_df: pd.DataFrame):
    """long_df has columns hotel_id, room_type_code, stay_date, cp, actual."""
    level_df = long_df[long_df.cp == 0][
        ["hotel_id", "room_type_code", "stay_date", "actual"]
    ].rename(columns={"actual": "y"})

    shape_src = long_df.merge(
        level_df.rename(columns={"y": "final"}),
        on=["hotel_id", "room_type_code", "stay_date"],
    )
    shape_src = shape_src[shape_src.cp != 0].copy()
    shape_src["y"] = shape_src["actual"] / shape_src["final"].clip(lower=1e-6)
    shape_src["y"] = shape_src["y"].clip(0.0, 1.0)
    shape_df = shape_src[["hotel_id", "room_type_code", "stay_date", "cp", "y"]]
    return level_df.reset_index(drop=True), shape_df.reset_index(drop=True)


def _fit_categories(static, level_df: pd.DataFrame) -> dict:
    from .features import build_static_feature_frame

    static_feats = build_static_feature_frame(static, static.room_static)
    cats = {}
    for c in CATEGORICAL_FEATURES:
        vals = sorted(set(static_feats[c].dropna().astype(str)) | {"unknown"})
        cats[c] = vals
    return cats


def _prep_matrix(static, cats, df, include_cp):
    from .features import build_feature_matrix

    X = build_feature_matrix(static, df, include_cp=include_cp)
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c].astype(object), categories=cats[c])
    cols = SHAPE_FEATURES if include_cp else LEVEL_FEATURES
    return X[cols]


def _best_rounds(params, X_tr, y_tr, X_val, y_val, cat_cols):
    dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols, reference=dtrain, free_raw_data=False)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return max(booster.best_iteration, 10)


def run_training(
    data_dir: Path,
    model_base_dir: Path,
    version: str | None = None,
    promote: bool = True,
    seed: int = SEED,
    train_end: str | None = None,
    val_start: str | None = None,
    extrapolation_gamma: float | None = None,
    interval_widen_k: float | None = None,
    extrapolation_gamma_lo: float | None = None,
    extrapolation_gamma_hi: float | None = None,
) -> Path:
    """Runs the full pipeline and saves a versioned artifact under
    `model_base_dir/<version>/`. Returns that directory.

    `promote=False` trains and saves without updating `current.json` — used
    by tests and by a "canary" workflow where you want the artifact on disk
    for evaluation before it's live (see DESIGN.md §6.5's shadow/canary
    discussion; this is the concrete hook for it).

    `seed` overrides the module default (see SEED above) for both the
    LightGBM boosters and the OOF KFold split — the hook
    evaluation/seed_sensitivity.py uses to retrain under different seeds
    without mutating module state.

    `train_end` overrides the module default (data.TRAIN_END) — the hook
    evaluation/rolling_backtest.py uses to retrain at several different
    walk-forward cutoffs. `val_start` (the internal round-tuning holdout)
    defaults to 21 days before whatever `train_end` ends up being, not a
    fixed calendar date, so it stays "the last ~3 weeks of the train
    window" at every cutoff instead of silently drifting outside the
    training window (or swallowing the whole thing) as train_end moves.

    `extrapolation_gamma`/`interval_widen_k` override the module defaults
    (see EXTRAPOLATION_GAMMA/INTERVAL_WIDEN_K above) — the hook
    evaluation/calibration_tuning.py uses to search for values that
    actually hit target interval coverage on held-out rolling folds,
    instead of shipping an unvalidated order-of-magnitude guess forever.

    `extrapolation_gamma_lo`/`_hi` override the per-side widening gammas
    (model.py's asymmetric widening — see its own docstring for why a
    single shared gamma never converged) — the hook
    evaluation/asymmetric_calibration_tuning.py uses. Left `None`, both
    default to whatever `extrapolation_gamma` resolves to, so an
    ordinary training run (no side-specific override) behaves exactly
    like the old single-gamma version.
    """
    version = new_version(version)
    out_dir = model_base_dir / version

    train_end = train_end or TRAIN_END
    val_start = val_start or (pd.Timestamp(train_end) - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
    extrapolation_gamma = EXTRAPOLATION_GAMMA if extrapolation_gamma is None else extrapolation_gamma
    interval_widen_k = INTERVAL_WIDEN_K if interval_widen_k is None else interval_widen_k
    extrapolation_gamma_lo = extrapolation_gamma if extrapolation_gamma_lo is None else extrapolation_gamma_lo
    extrapolation_gamma_hi = extrapolation_gamma if extrapolation_gamma_hi is None else extrapolation_gamma_hi

    level_lgb_params = dict(LEVEL_LGB_PARAMS, seed=seed)
    shape_lgb_params_base = dict(SHAPE_LGB_PARAMS, seed=seed)

    log.info(f"Loading data from {data_dir} ...")
    static = load_static_context(data_dir)
    reservations = load_reservations(data_dir)

    reservation_hotels = hotels_with_reservations(reservations)
    log.info(f"Hotels with reservation history in this data: {reservation_hotels}")
    log.info(f"Building actual-curve training table (stay_date <= {train_end}) ...")
    long_df = build_actual_curve_table(
        reservations, static, TRAIN_FLOOR, train_end, hotels=reservation_hotels
    )
    log.info(f"  {long_df['stay_date'].nunique()} distinct nights, {len(long_df):,} long rows")
    level_df, shape_df = _make_level_shape_tables(long_df)
    log.info(f"  level rows (curves): {len(level_df):,} | shape rows: {len(shape_df):,}")
    log.info("  curves per hotel:\n%s", level_df.groupby("hotel_id").size().to_string())

    cats = _fit_categories(static, level_df)

    # curve id used to group folds / time-split consistently across stages
    level_df["curve_id"] = (
        level_df.hotel_id + "|" + level_df.room_type_code + "|" + level_df.stay_date.dt.strftime("%Y-%m-%d")
    )
    shape_df["curve_id"] = (
        shape_df.hotel_id + "|" + shape_df.room_type_code + "|" + shape_df.stay_date.dt.strftime("%Y-%m-%d")
    )

    # ---- 1-2. time-based split for round-count tuning -------------------
    is_val = level_df.stay_date >= pd.Timestamp(val_start)
    val_curve_ids = set(level_df.loc[is_val, "curve_id"])
    log.info(
        f"Internal time-split validation: {is_val.sum()} / {len(level_df)} curves "
        f"(stay_date >= {val_start})"
    )

    X_level_tr = _prep_matrix(static, cats, level_df.loc[~is_val], include_cp=False)
    X_level_val = _prep_matrix(static, cats, level_df.loc[is_val], include_cp=False)
    level_rounds = _best_rounds(
        level_lgb_params,
        X_level_tr, level_df.loc[~is_val, "y"],
        X_level_val, level_df.loc[is_val, "y"],
        CATEGORICAL_FEATURES,
    )
    log.info(f"  level model: best_iteration = {level_rounds}")

    monotone = [0] * len(SHAPE_FEATURES)
    monotone[SHAPE_FEATURES.index("cp")] = -1
    shape_probe_params = dict(shape_lgb_params_base, monotone_constraints=monotone)

    shape_is_val = shape_df.curve_id.isin(val_curve_ids)
    X_shape_tr = _prep_matrix(static, cats, shape_df.loc[~shape_is_val], include_cp=True)
    X_shape_val = _prep_matrix(static, cats, shape_df.loc[shape_is_val], include_cp=True)
    shape_rounds = _best_rounds(
        shape_probe_params,
        X_shape_tr, shape_df.loc[~shape_is_val, "y"],
        X_shape_val, shape_df.loc[shape_is_val, "y"],
        CATEGORICAL_FEATURES,
    )
    log.info(f"  shape model: best_iteration = {shape_rounds}")

    # quick internal-validation sanity read (combined curve MAE on the June holdout)
    level_probe = lgb.train(
        level_lgb_params,
        lgb.Dataset(X_level_tr, level_df.loc[~is_val, "y"], categorical_feature=CATEGORICAL_FEATURES),
        num_boost_round=level_rounds,
    )
    val_level_pred = level_probe.predict(X_level_val)
    log.info(
        f"  [probe] level MAE on June holdout: "
        f"{np.mean(np.abs(val_level_pred - level_df.loc[is_val, 'y'])):.4f}"
    )

    # ---- 3. refit final boosters on the FULL train window ----------------
    log.info("Refitting final boosters on the full train window ...")
    X_level_full = _prep_matrix(static, cats, level_df, include_cp=False)
    level_booster = lgb.train(
        level_lgb_params,
        lgb.Dataset(X_level_full, level_df["y"], categorical_feature=CATEGORICAL_FEATURES),
        num_boost_round=level_rounds,
    )

    X_shape_full = _prep_matrix(static, cats, shape_df, include_cp=True)
    shape_booster = lgb.train(
        shape_probe_params,
        lgb.Dataset(X_shape_full, shape_df["y"], categorical_feature=CATEGORICAL_FEATURES),
        num_boost_round=shape_rounds,
    )

    # ---- 4. grouped 5-fold OOF residuals -> per-hotel shrinkage -----------
    log.info("Computing grouped 5-fold OOF residuals for per-hotel shrinkage ...")
    curve_ids = level_df["curve_id"].to_numpy()
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_level = np.full(len(level_df), np.nan)
    oof_shape = np.full(len(shape_df), np.nan)
    oof_level_q = {q: np.full(len(level_df), np.nan) for q in (0.1, 0.5, 0.9)}
    for tr_idx, te_idx in kf.split(curve_ids):
        tr_ids = set(curve_ids[tr_idx])
        te_ids = set(curve_ids[te_idx])

        m_lvl = level_df.curve_id.isin(tr_ids)
        m_lvl_te = level_df.curve_id.isin(te_ids)
        X_lvl_tr_fold = _prep_matrix(static, cats, level_df.loc[m_lvl], include_cp=False)
        X_lvl_te_fold = _prep_matrix(static, cats, level_df.loc[m_lvl_te], include_cp=False)

        booster = lgb.train(
            level_lgb_params,
            lgb.Dataset(X_lvl_tr_fold, level_df.loc[m_lvl, "y"], categorical_feature=CATEGORICAL_FEATURES),
            num_boost_round=level_rounds,
        )
        oof_level[m_lvl_te.to_numpy()] = booster.predict(X_lvl_te_fold)

        # Same folds, quantile objective — needed to conformally calibrate
        # the P10/P50/P90 boosters (see below: a plain per-hotel mean-shift
        # correction, tuned for the L1 point model, badly undercovered when
        # reused for quantile heads with their own systematic bias).
        for q in (0.1, 0.5, 0.9):
            qparams = dict(level_lgb_params, objective="quantile", alpha=q)
            qbooster = lgb.train(
                qparams,
                lgb.Dataset(X_lvl_tr_fold, level_df.loc[m_lvl, "y"], categorical_feature=CATEGORICAL_FEATURES),
                num_boost_round=level_rounds,
            )
            oof_level_q[q][m_lvl_te.to_numpy()] = qbooster.predict(X_lvl_te_fold)

        m_shp = shape_df.curve_id.isin(tr_ids)
        booster_s = lgb.train(
            shape_probe_params,
            lgb.Dataset(
                _prep_matrix(static, cats, shape_df.loc[m_shp], include_cp=True),
                shape_df.loc[m_shp, "y"],
                categorical_feature=CATEGORICAL_FEATURES,
            ),
            num_boost_round=shape_rounds,
        )
        m_shp_te = shape_df.curve_id.isin(te_ids)
        oof_shape[m_shp_te.to_numpy()] = booster_s.predict(
            _prep_matrix(static, cats, shape_df.loc[m_shp_te], include_cp=True)
        )

    level_df["oof_resid"] = level_df["y"] - oof_level
    shape_df["oof_resid"] = shape_df["y"] - oof_shape

    hotel_n_obs = level_df.groupby("hotel_id").size().to_dict()
    level_shrink_raw = level_df.groupby("hotel_id")["oof_resid"].mean().to_dict()
    shape_shrink_raw = shape_df.groupby("hotel_id")["oof_resid"].mean().to_dict()

    level_shrink = {
        h: (n / (n + LEVEL_SHRINK_K)) * level_shrink_raw.get(h, 0.0)
        for h, n in hotel_n_obs.items()
    }
    shape_shrink = {
        h: (n / (n + SHAPE_SHRINK_K)) * shape_shrink_raw.get(h, 0.0)
        for h, n in hotel_n_obs.items()
    }
    level_shrink_raw_r = {k: round(v, 4) for k, v in level_shrink_raw.items()}
    level_shrink_r = {k: round(v, 4) for k, v in level_shrink.items()}
    shape_shrink_raw_r = {k: round(v, 4) for k, v in shape_shrink_raw.items()}
    shape_shrink_r = {k: round(v, 4) for k, v in shape_shrink.items()}
    log.info(f"  hotel_n_obs: {hotel_n_obs}")
    log.info(f"  raw level OOF residual (mean): {level_shrink_raw_r}")
    log.info(f"  applied level shrink correction: {level_shrink_r}")
    log.info(f"  raw shape OOF residual (mean): {shape_shrink_raw_r}")
    log.info(f"  applied shape shrink correction: {shape_shrink_r}")

    # ---- 4a. second-level (hotel, room_type) shrinkage ---------------------
    # Fit on what the hotel-level correction above left behind, not on the
    # raw OOF residual — this is nested empirical Bayes (backbone -> hotel
    # -> room_type), not two independent corrections competing for the same
    # signal. See model.py module docstring for the reasoning.
    log.info("Computing (hotel, room_type) second-level shrinkage ...")
    level_df["room_key"] = level_df.apply(lambda r: room_key(r.hotel_id, r.room_type_code), axis=1)
    shape_df["room_key"] = shape_df.apply(lambda r: room_key(r.hotel_id, r.room_type_code), axis=1)
    level_df["resid_after_hotel"] = level_df["oof_resid"] - level_df["hotel_id"].map(level_shrink)
    shape_df["resid_after_hotel"] = shape_df["oof_resid"] - shape_df["hotel_id"].map(shape_shrink)

    room_n_obs = level_df.groupby("room_key").size().to_dict()
    room_level_raw = level_df.groupby("room_key")["resid_after_hotel"].mean().to_dict()
    room_shape_raw = shape_df.groupby("room_key")["resid_after_hotel"].mean().to_dict()

    room_level_shrink = {
        rk: (n / (n + ROOM_LEVEL_SHRINK_K)) * room_level_raw.get(rk, 0.0)
        for rk, n in room_n_obs.items()
        if n >= MIN_ROOM_N_FOR_POOLING
    }
    # shape's room_n_obs would double-count checkpoint rows if taken from
    # shape_df directly (9 rows per curve) — reuse the curve-level count
    # from level_df's room_n_obs so the shrink weight reflects "how many
    # curves have we seen for this room type," the same unit as the
    # hotel-level weight, not "how many checkpoint-rows."
    room_shape_shrink = {
        rk: (n / (n + ROOM_SHAPE_SHRINK_K)) * room_shape_raw.get(rk, 0.0)
        for rk, n in room_n_obs.items()
        if n >= MIN_ROOM_N_FOR_POOLING
    }
    n_pooled = len(room_level_shrink)
    n_total = len(room_n_obs)
    log.info(
        f"  room types with enough history to get a second-level correction "
        f"(n >= {MIN_ROOM_N_FOR_POOLING}): {n_pooled}/{n_total}"
    )
    log.info(f"  room_n_obs: {room_n_obs}")
    log.info(
        "  applied room-level level-shrink correction: %s",
        {k: round(v, 4) for k, v in room_level_shrink.items()},
    )
    log.info(
        "  applied room-level shape-shrink correction: %s",
        {k: round(v, 4) for k, v in room_shape_shrink.items()},
    )

    # ---- 4b. conformal calibration of the quantile heads -------------------
    # A raw LightGBM "quantile" objective booster is not automatically
    # calibrated (pinball loss minimizes conditional quantile risk on the
    # training distribution, which drifts under regularization + this
    # dataset's small-n, heavily-censored-at-1.0 target). We fix marginal
    # coverage the standard way (split-conformal / CQR-style): shift each
    # quantile's prediction by the empirical q-th quantile of its own OOF
    # residuals, pooled across hotels (per-hotel recentering is handled
    # separately by `level_shrink`, reused here for the same reason the
    # point estimate needs it).
    log.info("Conformal-calibrating quantile heads from OOF residuals ...")
    level_q_shift = {}
    for q in (0.1, 0.5, 0.9):
        resid = level_df["y"].to_numpy() - oof_level_q[q]
        level_q_shift[q] = float(np.quantile(resid, q))
    level_q_shift_r = {k: round(v, 4) for k, v in level_q_shift.items()}
    log.info(f"  conformal shift (P10/P50/P90): {level_q_shift_r}")

    # post-calibration OOF coverage check (P10/P90), reported for honesty in
    # DESIGN.md / README rather than assumed
    lo_c = oof_level_q[0.1] + level_q_shift[0.1]
    hi_c = oof_level_q[0.9] + level_q_shift[0.9]
    covered = (level_df["y"].to_numpy() >= lo_c - 1e-9) & (level_df["y"].to_numpy() <= hi_c + 1e-9)
    log.info(f"  OOF PICP after conformal shift (target 0.80): {covered.mean():.3f}")

    # ---- 5. quantile boosters on the level target -------------------------
    log.info("Fitting P10/P50/P90 quantile boosters (level target) ...")
    level_q_boosters = {}
    for q in (0.1, 0.5, 0.9):
        params = dict(level_lgb_params)
        params["objective"] = "quantile"
        params["alpha"] = q
        level_q_boosters[q] = lgb.train(
            params,
            lgb.Dataset(X_level_full, level_df["y"], categorical_feature=CATEGORICAL_FEATURES),
            num_boost_round=level_rounds,
        )

    # ---- 6. save -----------------------------------------------------------
    model = BookingCurveModel(
        level_booster=level_booster,
        shape_booster=shape_booster,
        level_q_boosters=level_q_boosters,
        cat_categories=cats,
        level_shrink=level_shrink,
        shape_shrink=shape_shrink,
        level_shrink_k=LEVEL_SHRINK_K,
        shape_shrink_k=SHAPE_SHRINK_K,
        hotel_n_obs=hotel_n_obs,
        interval_widen_k=interval_widen_k,
        room_level_shrink=room_level_shrink,
        room_shape_shrink=room_shape_shrink,
        room_level_shrink_k=ROOM_LEVEL_SHRINK_K,
        room_shape_shrink_k=ROOM_SHAPE_SHRINK_K,
        room_n_obs=room_n_obs,
        level_q_shift=level_q_shift,
        train_doy_range=(int(level_df.stay_date.dt.dayofyear.min()), int(level_df.stay_date.dt.dayofyear.max())),
        extrapolation_gamma=extrapolation_gamma,
        extrapolation_gamma_lo=extrapolation_gamma_lo,
        extrapolation_gamma_hi=extrapolation_gamma_hi,
        meta={
            "level_rounds": level_rounds,
            "shape_rounds": shape_rounds,
            "train_end": train_end,
            "val_start": val_start,
            "n_level_rows": len(level_df),
            "n_shape_rows": len(shape_df),
            "seed": seed,
            "extrapolation_gamma": extrapolation_gamma,
            "interval_widen_k": interval_widen_k,
            "extrapolation_gamma_lo": extrapolation_gamma_lo,
            "extrapolation_gamma_hi": extrapolation_gamma_hi,
        },
    )
    model.save(out_dir)
    log.info(f"Saved model artifact {version} to {out_dir}")

    if promote:
        save_pointer(model_base_dir, version)
        log.info(f"Promoted {version} to current.json in {model_base_dir}")
    else:
        log.info(f"{version} saved but NOT promoted (promote=False) — current.json unchanged")

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--out",
        default=None,
        help="Base directory for versioned artifacts (default: artifacts/model). "
        "A new versioned subfolder is created under this on every run.",
    )
    ap.add_argument(
        "--version",
        default=None,
        help="Explicit version id (e.g. a git SHA from CI). Default: UTC timestamp, "
        "or $MODEL_VERSION if set.",
    )
    ap.add_argument(
        "--no-promote",
        action="store_true",
        help="Save the versioned artifact but don't update current.json — "
        "for shadow/canary training runs that shouldn't go live yet.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Override the training seed (default: {SEED}). Used by "
        "evaluation/seed_sensitivity.py to check whether results hold up "
        "under a different arbitrary seed, not just reproduce bit-for-bit "
        "under the same one.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"
    model_base_dir = Path(args.out) if args.out else repo_root / "artifacts" / "model"

    run_training(
        data_dir, model_base_dir, version=args.version, promote=not args.no_promote, seed=args.seed
    )


if __name__ == "__main__":
    main()
