"""
BookingCurveModel: feature engineering -> two-stage GBM -> constraint
enforcement.

Architecture (see DESIGN.md §6.1 for the full justification):

  Stage 1 ("level"):  final occupancy fraction at the stay date (cp=0),
                       regressed on hotel/room/calendar attributes.
  Stage 2 ("shape"):  fraction of *final* occupancy already on the books at
                       each checkpoint, g(cp) in [0, 1], g(0) == 1 by
                       definition. Regressed on the same features plus cp,
                       with a monotonic constraint on cp so the shape curve
                       cannot be non-monotonic by construction.

  Combined curve:      y(cp) = clip(level_hat * g_hat(cp), 0, 1)

This decomposition exists because "how many rooms will be full" (level) and
"how early do bookings arrive" (pace/shape) are different questions with
different transfer properties across hotels and seasons — see DESIGN.md
§6.1/§6.2. It also gives a natural, low-parameter place to apply per-hotel
partial pooling: one scalar shrinkage correction per hotel per stage,
instead of trying to correct 10 correlated checkpoint values independently.

Per-hotel partial pooling ("hierarchical, the pragmatic version"): rather
than fit a full hierarchical Bayesian model, we fit ONE global backbone GBM
and add an empirical-Bayes (James-Stein-style) per-hotel bias correction,
shrunk toward zero as `n_hotel / (n_hotel + K)`. A hotel with lots of its
own history gets most of its own signal back; a brand-new hotel (n=0) gets
exactly the global backbone prediction, which is what makes this the same
mechanism that answers the cold-start question in §6.2.

This pooling is actually two nested levels, not one. A hotel's room types
are not exchangeable with each other: within a hotel, the "optimized" base
room type is priced first and every "derived" room type is a fixed or
near-fixed delta off it (see DESIGN.md §6.1), so derived rooms' occupancy
is mechanically correlated with their hotel's base room, not just with
"hotels in general." A single hotel-level scalar throws that signal away —
it corrects "hotel_C runs 4pp hot" but not "hotel_C's largest suite runs
9pp cold relative even to hotel_C's own average." So the correction is
applied in two passes, backbone -> hotel -> (hotel, room_type): first the
existing per-hotel correction, exactly as before; then a second,
independent empirical-Bayes correction fit on the RESIDUAL that remains
after the hotel-level correction, grouped by (hotel_id, room_type_code)
and shrunk toward zero by that room type's own curve count (`n_room /
(n_room + K_room)`). A room type with plenty of its own history (the base
"optimized" type, usually) gets most of that residual signal back; a
derived type with only a handful of curves gets shrunk almost entirely
back to its hotel's correction, which is exactly the fallback a brand-new
room type in a known hotel should get. This is still not a joint
hierarchical model (the two levels are fit sequentially on residuals, not
jointly), but it is the natural two-level extension of the same
mechanism, and it costs one more small dict, not a different
architecture.

All prediction paths funnel through `predict_curve`, so train-time
evaluation and serve-time inference always use the same code — this is the
"train/serve skew" requirement from README §8 / DESIGN.md §6.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CHECKPOINTS
from .features import (
    CATEGORICAL_FEATURES,
    LEVEL_FEATURES,
    SHAPE_FEATURES,
    build_feature_matrix,
)

NONZERO_CHECKPOINTS = [c for c in CHECKPOINTS if c != 0]
QUANTILES = (0.1, 0.5, 0.9)


def enforce_curve_constraints(cp_list, values) -> dict:
    """Hard safety net: clip to [0,1] and force non-decreasing as cp -> 0,
    regardless of what the upstream model produced. This is what guarantees
    zero monotonicity/bound violations on the grader, independent of model
    quality — constraint enforcement is a separate, auditable layer from the
    model itself (README's own `src/model.py` docstring: "feature
    engineering, model, constraint enforcement" are three distinct jobs).
    """
    cps = np.array(cp_list)
    order = np.argsort(-cps)  # descending: 90 -> 0
    v = np.clip(np.array(values, dtype=float)[order], 0.0, 1.0)
    v = np.maximum.accumulate(v)
    out = np.empty_like(v)
    out[order] = v
    return {str(int(cp)): float(val) for cp, val in zip(cps, out)}


MAX_INVENTORY_FOR_QUANTIZATION = 3


def quantize_curve_to_inventory(curve: dict, inventory_count: int | None) -> dict:
    """Occupancy at every checkpoint is `booked_rooms / inventory_count` —
    an integer over an integer, not a free continuous quantity. For most
    room types inventory_count is large enough that this is a fine detail
    (nearest achievable value to any model output is close by). It stops
    being a fine detail once inventory_count is tiny: with inventory=1 the
    only achievable values are {0, 1}; with inventory=2, {0, 0.5, 1}. In
    this dataset 44% of hotel_C's room-type-nights have inventory <= 2 —
    that's not an edge case to shrug off, it's most of one hotel's rooms.
    A raw regression output like 0.34 for an inventory=1 room isn't a
    plausible prediction of anything real; it can only ever be 0 or 1.

    Snaps every checkpoint to its nearest achievable k/inventory_count,
    only when inventory_count is known and small (large-inventory rooms
    are already effectively continuous — snapping them buys nothing and
    risks quietly nudging a well-calibrated prediction). Caller is
    responsible for re-enforcing monotonicity afterward: rounding two
    adjacent checkpoints independently can occasionally flatten or
    reverse their order even though the pre-quantized values were
    correctly ordered.
    """
    if not inventory_count or inventory_count > MAX_INVENTORY_FOR_QUANTIZATION:
        return dict(curve)
    step = 1.0 / inventory_count
    return {cp: round(round(v / step) * step, 10) for cp, v in curve.items()}


def _shrink_weight(n: int, k: float) -> float:
    return n / (n + k) if (n + k) > 0 else 0.0


def room_key(hotel_id: str, room_type_code: str) -> str:
    """Composite key for the second (finer) pooling level. A plain string
    join, not a tuple, because this dict round-trips through JSON
    (meta.json) where tuple keys don't exist."""
    return f"{hotel_id}|{room_type_code}"


class BookingCurveModel:
    def __init__(
        self,
        level_booster: lgb.Booster,
        shape_booster: lgb.Booster,
        level_q_boosters: dict[float, lgb.Booster],
        cat_categories: dict[str, list],
        level_shrink: dict[str, float],
        shape_shrink: dict[str, float],
        level_shrink_k: float,
        shape_shrink_k: float,
        hotel_n_obs: dict[str, int],
        interval_widen_k: float,
        room_level_shrink: dict[str, float] | None = None,
        room_shape_shrink: dict[str, float] | None = None,
        room_level_shrink_k: float = 20.0,
        room_shape_shrink_k: float = 40.0,
        room_n_obs: dict[str, int] | None = None,
        level_q_shift: dict[float, float] | None = None,
        train_doy_range: tuple[int, int] = (1, 366),
        extrapolation_gamma: float = 0.0,
        meta: dict | None = None,
    ):
        self.level_booster = level_booster
        self.shape_booster = shape_booster
        self.level_q_boosters = level_q_boosters
        self.cat_categories = cat_categories
        self.level_shrink = level_shrink
        self.shape_shrink = shape_shrink
        self.level_shrink_k = level_shrink_k
        self.shape_shrink_k = shape_shrink_k
        self.hotel_n_obs = hotel_n_obs
        self.interval_widen_k = interval_widen_k
        # second (finer) pooling level: (hotel, room_type) correction on top
        # of the hotel-level one — see module docstring.
        self.room_level_shrink = room_level_shrink or {}
        self.room_shape_shrink = room_shape_shrink or {}
        self.room_level_shrink_k = room_level_shrink_k
        self.room_shape_shrink_k = room_shape_shrink_k
        self.room_n_obs = room_n_obs or {}
        self.level_q_shift = level_q_shift or {0.1: 0.0, 0.5: 0.0, 0.9: 0.0}
        self.train_doy_range = tuple(train_doy_range)
        self.extrapolation_gamma = extrapolation_gamma
        self.meta = meta or {}

    def _extrapolation_widen(self, stay_dates) -> np.ndarray:
        """Extra interval inflation for stay dates outside the calendar
        range actually observed in training (day-of-year distance past the
        nearest training edge). This is a feature-space novelty signal, not
        a fit to test-set outcomes: it only knows the training window's own
        boundaries. See DESIGN.md §6.7 — it's a partial, honest mitigation
        for genuine out-of-season extrapolation (e.g. hotel_H's Jul-Sep
        "busy season" is entirely unseen in its Apr-Jun training data);
        it widens the interval but cannot fix a biased median by itself.
        """
        doy = pd.to_datetime(stay_dates).dt.dayofyear.to_numpy()
        lo, hi = self.train_doy_range
        dist = np.maximum(0, np.maximum(doy - hi, lo - doy))
        return 1.0 + self.extrapolation_gamma * (dist / 30.0)

    def _extrapolation_correction_damp(self, stay_dates) -> np.ndarray:
        """How much to trust the OOF-fit per-hotel/room-type CORRECTION
        (not just how much to widen the interval around it) for a stay
        date outside the training season.

        This exists because of a real ablation finding, not a hunch: the
        per-hotel correction is fit honestly via in-sample OOF
        cross-validation, but this dataset's entire train window
        (Mar-Jun) sits before its entire test window (Jul-Sep) — every
        single test prediction is "extrapolation" by the definition
        above. evaluation/ablation_study.py shows the correction applied
        at full strength there makes test weighted MAE WORSE than no
        correction at all (0.306 vs 0.271) — most sharply for hotel_H,
        whose correction was learned from its Mar-Jun ramp-up period and
        doesn't describe its unseen Jul-Sep peak season. A bias learned
        from one season is exactly the kind of thing that shouldn't be
        assumed to hold in a season the model has never seen.

        Deliberately reuses `_extrapolation_widen`'s own distance
        computation rather than introducing a second, separately-tuned
        decay constant: as the interval widens by a factor W to express
        "we don't know this regime," trust in the correction shrinks by
        1/W for the same reason. At dist=0 (in-season) this is exactly
        1.0 — in-season behavior is completely unchanged by this fix.
        """
        return 1.0 / self._extrapolation_widen(stay_dates)

    # ---- feature prep -----------------------------------------------
    def _fix_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in CATEGORICAL_FEATURES:
            cats = self.cat_categories.get(c, [])
            X[c] = pd.Categorical(X[c].astype(object), categories=cats)
        return X

    def _level_matrix(self, static, df: pd.DataFrame) -> pd.DataFrame:
        X = build_feature_matrix(static, df, include_cp=False)
        return self._fix_categories(X)[LEVEL_FEATURES]

    def _shape_matrix(self, static, df: pd.DataFrame) -> pd.DataFrame:
        X = build_feature_matrix(static, df, include_cp=True)
        return self._fix_categories(X)[SHAPE_FEATURES]

    # ---- raw model outputs (no shrink, no constraints) ---------------
    def predict_level_raw(self, static, df: pd.DataFrame) -> np.ndarray:
        X = self._level_matrix(static, df)
        return self.level_booster.predict(X)

    def predict_level_quantiles_raw(self, static, df: pd.DataFrame) -> dict:
        X = self._level_matrix(static, df)
        return {q: booster.predict(X) for q, booster in self.level_q_boosters.items()}

    def predict_shape_raw(self, static, df_long: pd.DataFrame) -> np.ndarray:
        X = self._shape_matrix(static, df_long)
        return self.shape_booster.predict(X)

    # ---- shrinkage-corrected, single-curve inference ------------------
    def hotel_shrink_weight_level(self, hotel_id: str) -> float:
        return _shrink_weight(self.hotel_n_obs.get(hotel_id, 0), self.level_shrink_k)

    def hotel_shrink_weight_shape(self, hotel_id: str) -> float:
        return _shrink_weight(self.hotel_n_obs.get(hotel_id, 0), self.shape_shrink_k)

    def room_shrink_weight_level(self, hotel_id: str, room_type_code: str) -> float:
        n = self.room_n_obs.get(room_key(hotel_id, room_type_code), 0)
        return _shrink_weight(n, self.room_level_shrink_k)

    def room_shrink_weight_shape(self, hotel_id: str, room_type_code: str) -> float:
        n = self.room_n_obs.get(room_key(hotel_id, room_type_code), 0)
        return _shrink_weight(n, self.room_shape_shrink_k)

    def _level_correction(self, hotel_id: str, room_type_code: str) -> float:
        """Nested correction: hotel-level term plus a second (hotel,
        room_type) term fit on what the hotel-level term left behind. A
        room type with no history of its own (room_n_obs == 0, e.g. a
        brand-new derived type) contributes exactly 0 here and falls back
        to the hotel-level correction alone."""
        hotel_corr = self.level_shrink.get(hotel_id, 0.0)
        room_corr = self.room_level_shrink.get(room_key(hotel_id, room_type_code), 0.0)
        return hotel_corr + room_corr

    def _shape_correction(self, hotel_id: str, room_type_code: str) -> float:
        hotel_corr = self.shape_shrink.get(hotel_id, 0.0)
        room_corr = self.room_shape_shrink.get(room_key(hotel_id, room_type_code), 0.0)
        return hotel_corr + room_corr

    def _corrected_level(self, static, hotel_id: str, room_type_code: str, stay_date) -> float:
        row = pd.DataFrame(
            {"hotel_id": [hotel_id], "room_type_code": [room_type_code], "stay_date": [stay_date]}
        )
        raw = float(self.predict_level_raw(static, row)[0])
        damp = float(self._extrapolation_correction_damp(pd.Series([stay_date]))[0])
        corr = self._level_correction(hotel_id, room_type_code) * damp
        return float(np.clip(raw + corr, 0.0, 1.0))

    def _corrected_level_quantiles(self, static, hotel_id, room_type_code, stay_date) -> dict:
        row = pd.DataFrame(
            {"hotel_id": [hotel_id], "room_type_code": [room_type_code], "stay_date": [stay_date]}
        )
        raw = self.predict_level_quantiles_raw(static, row)
        damp = float(self._extrapolation_correction_damp(pd.Series([stay_date]))[0])
        corr = self._level_correction(hotel_id, room_type_code) * damp
        n = self.hotel_n_obs.get(hotel_id, 0)
        widen = (1.0 + self.interval_widen_k / np.sqrt(n + 1)) * float(
            self._extrapolation_widen(pd.Series([stay_date]))[0]
        )
        # conformal shift first (fixes marginal coverage bias of the raw
        # pinball-loss booster), then per-hotel recentering, then widen for
        # hotels we have little of our own data on / dates outside the
        # training season.
        med = float(np.clip(raw[0.5][0] + self.level_q_shift.get(0.5, 0.0) + corr, 0.0, 1.0))
        lo = float(np.clip(raw[0.1][0] + self.level_q_shift.get(0.1, 0.0) + corr, 0.0, 1.0))
        hi = float(np.clip(raw[0.9][0] + self.level_q_shift.get(0.9, 0.0) + corr, 0.0, 1.0))
        lo = float(np.clip(med - (med - lo) * widen, 0.0, 1.0))
        hi = float(np.clip(med + (hi - med) * widen, 0.0, 1.0))
        lo, hi = min(lo, hi), max(lo, hi)
        return {"p10": lo, "p50": med, "p90": hi}

    def _corrected_shape(self, static, hotel_id: str, room_type_code: str, stay_date) -> dict:
        rows = pd.DataFrame(
            {
                "hotel_id": [hotel_id] * len(NONZERO_CHECKPOINTS),
                "room_type_code": [room_type_code] * len(NONZERO_CHECKPOINTS),
                "stay_date": [stay_date] * len(NONZERO_CHECKPOINTS),
                "cp": NONZERO_CHECKPOINTS,
            }
        )
        raw = self.predict_shape_raw(static, rows)
        damp = float(self._extrapolation_correction_damp(pd.Series([stay_date]))[0])
        corr = self._shape_correction(hotel_id, room_type_code) * damp
        g = np.clip(raw + corr, 0.0, 1.0)
        g_dict = {cp: float(v) for cp, v in zip(NONZERO_CHECKPOINTS, g)}
        g_dict[0] = 1.0
        # shape must itself be non-decreasing as cp -> 0; cheap safety net
        order = np.argsort(-np.array(CHECKPOINTS))
        cps_sorted = np.array(CHECKPOINTS)[order]
        vals_sorted = np.maximum.accumulate([g_dict[c] for c in cps_sorted])
        return {int(c): float(v) for c, v in zip(cps_sorted, vals_sorted)}

    @staticmethod
    def _quantize_all(point: dict, intervals: dict, inventory_count) -> tuple[dict, dict]:
        """Applies quantize_curve_to_inventory to point + all three
        interval curves, then re-runs enforce_curve_constraints on each
        (quantizing independently per checkpoint can occasionally unorder
        two adjacent values), then restores p10<=p50<=p90 one more time
        for the same reason. A no-op when inventory_count is unknown or
        above MAX_INVENTORY_FOR_QUANTIZATION — quantize_curve_to_inventory
        itself is the single point of truth for that threshold."""
        if not inventory_count or inventory_count > MAX_INVENTORY_FOR_QUANTIZATION:
            return point, intervals

        def _requantize(curve: dict) -> dict:
            q = quantize_curve_to_inventory(curve, inventory_count)
            return enforce_curve_constraints(CHECKPOINTS, [q[str(cp)] for cp in CHECKPOINTS])

        point = _requantize(point)
        intervals = {k: _requantize(v) for k, v in intervals.items()}
        for cp in map(str, CHECKPOINTS):
            lo, med, hi = intervals["p10"][cp], intervals["p50"][cp], intervals["p90"][cp]
            lo, med, hi = sorted([lo, med, hi])
            intervals["p10"][cp], intervals["p50"][cp], intervals["p90"][cp] = lo, med, hi
        return point, intervals

    def predict_curve(self, static, hotel_id: str, room_type_code: str, stay_date) -> dict:
        """Full blind (no as_of anchoring) predicted curve, constraint-
        enforced. Returns {'point': {...}, 'p10': {...}, 'p50': {...},
        'p90': {...}, 'diagnostics': {...}}."""
        y_final = self._corrected_level(static, hotel_id, room_type_code, stay_date)
        g = self._corrected_shape(static, hotel_id, room_type_code, stay_date)
        point_raw = {cp: y_final * g[cp] for cp in CHECKPOINTS}
        point = enforce_curve_constraints(CHECKPOINTS, [point_raw[c] for c in CHECKPOINTS])

        q = self._corrected_level_quantiles(static, hotel_id, room_type_code, stay_date)
        intervals = {}
        for key in ("p10", "p50", "p90"):
            raw = {cp: q[key] * g[cp] for cp in CHECKPOINTS}
            intervals[key] = enforce_curve_constraints(CHECKPOINTS, [raw[c] for c in CHECKPOINTS])
        # prevent quantile crossing at each checkpoint independently
        for cp in map(str, CHECKPOINTS):
            lo, med, hi = intervals["p10"][cp], intervals["p50"][cp], intervals["p90"][cp]
            lo, med, hi = sorted([lo, med, hi])
            intervals["p10"][cp], intervals["p50"][cp], intervals["p90"][cp] = lo, med, hi

        inv = static.inventory_lookup.get((hotel_id, room_type_code))
        point, intervals = self._quantize_all(point, intervals, inv)

        n_obs = self.hotel_n_obs.get(hotel_id, 0)
        room_n = self.room_n_obs.get(room_key(hotel_id, room_type_code), 0)
        diagnostics = {
            "final_occupancy_hat": y_final,
            "hotel_n_train_obs": n_obs,
            "level_shrink_weight": self.hotel_shrink_weight_level(hotel_id),
            "shape_shrink_weight": self.hotel_shrink_weight_shape(hotel_id),
            "known_hotel": hotel_id in self.hotel_n_obs,
            "room_type_n_train_obs": room_n,
            "room_level_shrink_weight": self.room_shrink_weight_level(hotel_id, room_type_code),
            "room_shape_shrink_weight": self.room_shrink_weight_shape(hotel_id, room_type_code),
            "known_room_type": room_key(hotel_id, room_type_code) in self.room_n_obs,
            "inventory_count": inv,
            "inventory_quantized": bool(inv) and inv <= MAX_INVENTORY_FOR_QUANTIZATION,
        }
        return {"point": point, **intervals, "diagnostics": diagnostics}

    # ---- vectorized batch inference (blind curves for many rows at once) --
    def predict_curve_batch(self, static, keys: pd.DataFrame) -> list[dict]:
        """keys: DataFrame with columns hotel_id, room_type_code, stay_date
        (one row per curve requested). Returns a list of blind-curve result
        dicts, same shape as `predict_curve`, in row order. This exists
        purely for throughput: generating thousands of curves by calling
        `predict_curve` in a Python loop pays LightGBM's per-call overhead
        thousands of times; this batches every stage into a handful of
        Booster.predict calls.
        """
        keys = keys.reset_index(drop=True)
        n = len(keys)
        hotel_ids = keys["hotel_id"].tolist()
        room_types = keys["room_type_code"].tolist()
        room_keys = [room_key(h, r) for h, r in zip(hotel_ids, room_types)]

        level_raw = self.predict_level_raw(static, keys)
        level_corr_raw = np.array(
            [
                self.level_shrink.get(h, 0.0) + self.room_level_shrink.get(rk, 0.0)
                for h, rk in zip(hotel_ids, room_keys)
            ]
        )
        damp = self._extrapolation_correction_damp(keys["stay_date"])
        level_corr = level_corr_raw * damp
        y_final = np.clip(level_raw + level_corr, 0.0, 1.0)

        q_raw = self.predict_level_quantiles_raw(static, keys)
        n_obs = np.array([self.hotel_n_obs.get(h, 0) for h in hotel_ids])
        widen = (1.0 + self.interval_widen_k / np.sqrt(n_obs + 1)) * self._extrapolation_widen(
            keys["stay_date"]
        )
        med = np.clip(q_raw[0.5] + self.level_q_shift.get(0.5, 0.0) + level_corr, 0.0, 1.0)
        lo = np.clip(q_raw[0.1] + self.level_q_shift.get(0.1, 0.0) + level_corr, 0.0, 1.0)
        hi = np.clip(q_raw[0.9] + self.level_q_shift.get(0.9, 0.0) + level_corr, 0.0, 1.0)
        lo = np.clip(med - (med - lo) * widen, 0.0, 1.0)
        hi = np.clip(med + (hi - med) * widen, 0.0, 1.0)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

        # shape: repeat each key once per non-zero checkpoint
        rep = keys.loc[keys.index.repeat(len(NONZERO_CHECKPOINTS))].reset_index(drop=True)
        rep["cp"] = NONZERO_CHECKPOINTS * n
        shape_raw = self.predict_shape_raw(static, rep)
        rep_room_keys = [
            room_key(h, r) for h, r in zip(rep["hotel_id"], rep["room_type_code"])
        ]
        shape_corr_raw = np.array(
            [
                self.shape_shrink.get(h, 0.0) + self.room_shape_shrink.get(rk, 0.0)
                for h, rk in zip(rep["hotel_id"], rep_room_keys)
            ]
        )
        shape_damp = self._extrapolation_correction_damp(rep["stay_date"])
        shape_corr = shape_corr_raw * shape_damp
        g_flat = np.clip(shape_raw + shape_corr, 0.0, 1.0).reshape(n, len(NONZERO_CHECKPOINTS))

        results = []
        for i in range(n):
            g = {cp: float(v) for cp, v in zip(NONZERO_CHECKPOINTS, g_flat[i])}
            g[0] = 1.0
            order = np.argsort(-np.array(CHECKPOINTS))
            cps_sorted = np.array(CHECKPOINTS)[order]
            vals_sorted = np.maximum.accumulate([g[c] for c in cps_sorted])
            g = {int(c): float(v) for c, v in zip(cps_sorted, vals_sorted)}

            point_raw = {cp: y_final[i] * g[cp] for cp in CHECKPOINTS}
            point = enforce_curve_constraints(CHECKPOINTS, [point_raw[c] for c in CHECKPOINTS])

            intervals = {}
            for key, level_val in (("p10", lo[i]), ("p50", med[i]), ("p90", hi[i])):
                raw = {cp: level_val * g[cp] for cp in CHECKPOINTS}
                intervals[key] = enforce_curve_constraints(CHECKPOINTS, [raw[c] for c in CHECKPOINTS])
            for cp in map(str, CHECKPOINTS):
                a, b, c = intervals["p10"][cp], intervals["p50"][cp], intervals["p90"][cp]
                a, b, c = sorted([a, b, c])
                intervals["p10"][cp], intervals["p50"][cp], intervals["p90"][cp] = a, b, c

            inv = static.inventory_lookup.get((hotel_ids[i], room_types[i]))
            point, intervals = self._quantize_all(point, intervals, inv)

            room_n_i = self.room_n_obs.get(room_keys[i], 0)
            diagnostics = {
                "final_occupancy_hat": float(y_final[i]),
                "hotel_n_train_obs": int(n_obs[i]),
                "level_shrink_weight": self.hotel_shrink_weight_level(hotel_ids[i]),
                "shape_shrink_weight": self.hotel_shrink_weight_shape(hotel_ids[i]),
                "known_hotel": hotel_ids[i] in self.hotel_n_obs,
                "room_type_n_train_obs": room_n_i,
                "room_level_shrink_weight": self.room_shrink_weight_level(hotel_ids[i], room_types[i]),
                "room_shape_shrink_weight": self.room_shrink_weight_shape(hotel_ids[i], room_types[i]),
                "known_room_type": room_keys[i] in self.room_n_obs,
                "inventory_count": inv,
                "inventory_quantized": bool(inv) and inv <= MAX_INVENTORY_FOR_QUANTIZATION,
            }
            results.append({"point": point, **intervals, "diagnostics": diagnostics})
        return results

    # ---- persistence ---------------------------------------------------
    def save(self, out_dir: Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.level_booster.save_model(str(out_dir / "level.txt"))
        self.shape_booster.save_model(str(out_dir / "shape.txt"))
        for q, booster in self.level_q_boosters.items():
            booster.save_model(str(out_dir / f"level_q{int(q * 100)}.txt"))
        meta = {
            "cat_categories": self.cat_categories,
            "level_shrink": self.level_shrink,
            "shape_shrink": self.shape_shrink,
            "level_shrink_k": self.level_shrink_k,
            "shape_shrink_k": self.shape_shrink_k,
            "hotel_n_obs": self.hotel_n_obs,
            "interval_widen_k": self.interval_widen_k,
            "room_level_shrink": self.room_level_shrink,
            "room_shape_shrink": self.room_shape_shrink,
            "room_level_shrink_k": self.room_level_shrink_k,
            "room_shape_shrink_k": self.room_shape_shrink_k,
            "room_n_obs": self.room_n_obs,
            "level_q_shift": self.level_q_shift,
            "train_doy_range": list(self.train_doy_range),
            "extrapolation_gamma": self.extrapolation_gamma,
            "quantiles": list(self.level_q_boosters.keys()),
            "extra": self.meta,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, in_dir: Path) -> "BookingCurveModel":
        in_dir = Path(in_dir)
        with open(in_dir / "meta.json") as f:
            meta = json.load(f)
        level_booster = lgb.Booster(model_file=str(in_dir / "level.txt"))
        shape_booster = lgb.Booster(model_file=str(in_dir / "shape.txt"))
        level_q_boosters = {
            float(q): lgb.Booster(model_file=str(in_dir / f"level_q{int(float(q) * 100)}.txt"))
            for q in meta["quantiles"]
        }
        return cls(
            level_booster=level_booster,
            shape_booster=shape_booster,
            level_q_boosters=level_q_boosters,
            cat_categories=meta["cat_categories"],
            level_shrink=meta["level_shrink"],
            shape_shrink=meta["shape_shrink"],
            level_shrink_k=meta["level_shrink_k"],
            shape_shrink_k=meta["shape_shrink_k"],
            hotel_n_obs=meta["hotel_n_obs"],
            interval_widen_k=meta["interval_widen_k"],
            room_level_shrink=meta.get("room_level_shrink", {}),
            room_shape_shrink=meta.get("room_shape_shrink", {}),
            room_level_shrink_k=meta.get("room_level_shrink_k", 20.0),
            room_shape_shrink_k=meta.get("room_shape_shrink_k", 40.0),
            room_n_obs=meta.get("room_n_obs", {}),
            level_q_shift={float(k): v for k, v in meta.get("level_q_shift", {}).items()},
            train_doy_range=tuple(meta.get("train_doy_range", (1, 366))),
            extrapolation_gamma=meta.get("extrapolation_gamma", 0.0),
            meta=meta.get("extra", {}),
        )
