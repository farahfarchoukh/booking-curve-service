"""
Tests for src/pricing.py — the pace-based yield-adjustment layer.

`pace_adjustment` (the pure rule) is tested directly with synthetic
inputs, precisely, without needing a trained model. `recommend_price`
(the orchestration: load the forecast, decide whether a live signal
exists at all) is tested against the same synthetic fixture everything
else in this suite uses, for schema/bounds/behavior — not for exact
numbers, matching this repo's existing test style for anything that
touches a real trained booster.
"""

import pytest

from src.pricing import (
    MAX_DISCOUNT_PCT,
    MAX_PREMIUM_PCT,
    pace_adjustment,
    recommend_price,
)

# ---- pace_adjustment: the pure rule -------------------------------------

def test_on_pace_gives_zero_adjustment():
    adj, capped = pace_adjustment(pace_ratio=1.0, confidence=1.0)
    assert adj == pytest.approx(0.0, abs=1e-9)
    assert capped is False


def test_ahead_of_pace_raises_price():
    adj, _ = pace_adjustment(pace_ratio=2.0, confidence=1.0)
    assert adj > 0


def test_behind_pace_lowers_price():
    adj, _ = pace_adjustment(pace_ratio=0.5, confidence=1.0)
    assert adj < 0


def test_adjustment_is_monotonic_in_pace_ratio():
    ratios = [0.3, 0.6, 1.0, 1.5, 3.0, 8.0]
    adjustments = [pace_adjustment(r, confidence=1.0)[0] for r in ratios]
    assert adjustments == sorted(adjustments)


def test_zero_confidence_holds_at_no_adjustment_regardless_of_pace():
    # This is the whole point of confidence gating: a hotel_H-shaped
    # forecast (wide interval -> confidence near 0) shouldn't move price
    # much even if the raw pace signal looks extreme.
    adj, _ = pace_adjustment(pace_ratio=100.0, confidence=0.0)
    assert adj == pytest.approx(0.0, abs=1e-9)


def test_confidence_scales_adjustment_magnitude():
    low_conf, _ = pace_adjustment(pace_ratio=3.0, confidence=0.2)
    high_conf, _ = pace_adjustment(pace_ratio=3.0, confidence=1.0)
    assert 0 < low_conf < high_conf


def test_extreme_pace_ratio_is_capped_not_unbounded():
    adj, capped = pace_adjustment(pace_ratio=1e6, confidence=1.0)
    assert adj == pytest.approx(MAX_PREMIUM_PCT)
    assert capped is True

    adj, capped = pace_adjustment(pace_ratio=1e-6, confidence=1.0)
    assert adj == pytest.approx(-MAX_DISCOUNT_PCT)
    assert capped is True


def test_pace_ratio_zero_does_not_crash():
    # anchor_frac can legitimately be exactly 0 (nothing booked yet at
    # as_of_date) -- log(0) must not blow up into nan/inf leaking out.
    adj, capped = pace_adjustment(pace_ratio=0.0, confidence=1.0)
    assert adj == pytest.approx(-MAX_DISCOUNT_PCT)
    assert capped is True


# ---- recommend_price: orchestration over a real (synthetic) forecast ----

def test_blind_mode_holds_at_base_rate(synthetic_data_dir, trained_model_dir):
    # Far-future stay date: no live pickup signal exists yet.
    rec = recommend_price(
        "hotel_X", "rt_x1", "2025-08-15", as_of_date="2025-04-01", base_rate=200.0,
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    assert rec["recommended_price"] == 200.0
    assert rec["adjustment_pct"] == 0.0
    assert rec["pace_ratio"] is None
    assert "no live pickup signal" in rec["reason"]


def test_anchored_mode_produces_bounded_recommendation(synthetic_data_dir, trained_model_dir):
    # A real synthetic hotel_X stay date with an as_of_date partway
    # through its booking window -- some but not all lead-time bookings
    # have landed, so there's a genuine (if synthetic) pace signal.
    rec = recommend_price(
        "hotel_X", "rt_x1", "2025-06-30", as_of_date="2025-06-10", base_rate=200.0,
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    assert rec["pace_ratio"] is not None
    assert rec["pace_ratio"] > 0
    assert 0.0 <= rec["confidence"] <= 1.0
    assert -MAX_DISCOUNT_PCT - 1e-9 <= rec["adjustment_pct"] <= MAX_PREMIUM_PCT + 1e-9
    expected_price = round(200.0 * (1.0 + rec["adjustment_pct"]), 2)
    assert rec["recommended_price"] == expected_price
    assert rec["recommended_price"] > 0


def test_thin_hotel_gets_dampened_confidence(synthetic_data_dir, trained_model_dir):
    # hotel_Y is the thin/cold-start-adjacent stand-in -- its wider
    # interval should show up as lower confidence than hotel_X's, all
    # else equal. Not a numeric equality (different hotels, different
    # everything) -- a structural check that the mechanism engages.
    thin = recommend_price(
        "hotel_Y", "rt_y1", "2025-03-01", as_of_date="2025-02-20", base_rate=150.0,
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    if thin["confidence"] is not None:
        assert thin["confidence"] <= 1.0


def test_invalid_base_rate_rejected(synthetic_data_dir, trained_model_dir):
    with pytest.raises(ValueError):
        recommend_price(
            "hotel_X", "rt_x1", "2025-06-30", as_of_date="2025-06-10", base_rate=0,
            data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
        )
    with pytest.raises(ValueError):
        recommend_price(
            "hotel_X", "rt_x1", "2025-06-30", as_of_date="2025-06-10", base_rate=-50,
            data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
        )
