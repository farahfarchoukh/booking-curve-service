"""
Shared feature engineering — used verbatim by train.py and predict.py.

Design decision (see DESIGN.md §6.1/§6.2/§6.3): the model never sees raw
`hotel_id` or `room_type_code` as a split feature. It only sees attributes
that exist for *any* hotel at onboarding time, before a single reservation
has landed: country, currency, PMS type, rate mode, region type, room-count
scale. This is what lets the exact same backbone model produce a sane
prediction for hotel_A/D/E/F/G (no reservation history at all) and for a
hypothetical hotel #401 tomorrow. Tenant identity re-enters only through the
separate, explicitly-shrunk per-hotel bias correction in model.py — never by
letting a tree split on "hotel_id == hotel_C".

All calendar signal is encoded as continuous/cyclical features (day-of-year
sin/cos, ISO week) rather than a raw month categorical. This matters here
specifically: hotel_C and hotel_H only have training examples for
Mar-Jun 2025, and the test window is Jul-Sep 2025 — a month never seen in
training. A discrete month category has no learned behavior for "month=8".
A continuous day-of-year embedding at least degrades gracefully (August sits
geometrically between June, which we've seen, and December).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import StaticContext

CATEGORICAL_FEATURES = [
    "country",
    "currency",
    "pms_type",
    "primary_rate_mode",
    "region_type",
    "room_type_kind",
]

NUMERIC_FEATURES = [
    "total_rooms",
    "inventory_count",
    "room_share",
    "dow",
    "is_weekend",
    "doy_sin",
    "doy_cos",
    "week_of_year",
]

# cp is only present for the pace/shape model; kept separate so the final-
# occupancy (level) model's feature list is identical minus this one column.
CP_FEATURE = "cp"

LEVEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
SHAPE_FEATURES = LEVEL_FEATURES + [CP_FEATURE]


def _room_static_row(static: StaticContext, hotel_id: str, room_type_code: str) -> dict:
    match = static.room_static[
        (static.room_static.hotel_id == hotel_id)
        & (static.room_static.room_type_code == room_type_code)
    ]
    if match.empty:
        # Unknown room type on a known (or unknown) hotel: fall back to
        # hotel-level attributes with neutral room-level defaults. This is
        # the cold-start path for a brand-new room type on an existing
        # property, or a hotel we've never seen at all.
        hotel_row = static.hotels[static.hotels.hotel_id == hotel_id]
        if hotel_row.empty:
            return {
                "country": "unknown",
                "currency": "unknown",
                "pms_type": "unknown",
                "primary_rate_mode": "unknown",
                "region_type": "unknown",
                "room_type_kind": "unknown",
                "total_rooms": np.nan,
                "inventory_count": np.nan,
                "room_share": np.nan,
            }
        h = hotel_row.iloc[0]
        total_rooms = static.total_rooms_lookup.get(hotel_id, np.nan)
        return {
            "country": h["country"],
            "currency": h["currency"],
            "pms_type": h["pms_type"],
            "primary_rate_mode": h["primary_rate_mode"],
            "region_type": h["region_type"],
            "room_type_kind": "unknown",
            "total_rooms": total_rooms,
            "inventory_count": np.nan,
            "room_share": np.nan,
        }
    r = match.iloc[0]
    return {
        "country": r["country"],
        "currency": r["currency"],
        "pms_type": r["pms_type"],
        "primary_rate_mode": r["primary_rate_mode"],
        "region_type": r["region_type"],
        "room_type_kind": r["room_type_kind"],
        "total_rooms": r["total_rooms"],
        "inventory_count": r["inventory_count"],
        "room_share": r["room_share"],
    }


def build_static_feature_frame(static: StaticContext, keys: pd.DataFrame) -> pd.DataFrame:
    """keys: DataFrame with columns hotel_id, room_type_code (deduplicated
    internally). Returns one static-attribute row per unique key, in the
    same row order as the de-duplicated keys — join back on
    (hotel_id, room_type_code)."""
    uniq = keys[["hotel_id", "room_type_code"]].drop_duplicates()
    recs = [
        _room_static_row(static, h, r)
        for h, r in zip(uniq["hotel_id"], uniq["room_type_code"])
    ]
    out = pd.DataFrame(recs, index=uniq.index)
    out["hotel_id"] = uniq["hotel_id"].values
    out["room_type_code"] = uniq["room_type_code"].values
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "stay_date") -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col])
    doy = dates.dt.dayofyear.astype(float)
    days_in_year = np.where(dates.dt.is_leap_year, 366.0, 365.0)
    angle = 2 * np.pi * doy / days_in_year
    out = df.copy()
    out["dow"] = dates.dt.dayofweek.astype(float)
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(float)
    out["doy_sin"] = np.sin(angle)
    out["doy_cos"] = np.cos(angle)
    out["week_of_year"] = dates.dt.isocalendar().week.astype(float)
    return out


def build_feature_matrix(
    static: StaticContext, df: pd.DataFrame, include_cp: bool
) -> pd.DataFrame:
    """df must have hotel_id, room_type_code, stay_date, and cp (if
    include_cp). Returns a feature-only DataFrame with categorical columns
    typed as pandas 'category' so LightGBM can use native categorical
    splits, ready to hand to Booster.predict / lgb.Dataset.
    """
    static_feats = build_static_feature_frame(static, df)
    merged = df.merge(static_feats, on=["hotel_id", "room_type_code"], how="left")
    merged = add_calendar_features(merged, "stay_date")

    cols = LEVEL_FEATURES + ([CP_FEATURE] if include_cp else [])
    X = merged[cols].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X
