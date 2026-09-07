"""
Pace-based yield-adjustment layer on top of the booking-curve forecast —
the bridge from "predict occupancy" to an actual price recommendation.

What this deliberately does NOT do: fit a price-elasticity curve from
this dataset. `evaluation/price_demand_eda.py` already found why not —
`suggested_prices.csv` is Ampliphi's own pricer's output, not an
independent price experiment, so any price -> occupancy correlation
learned from it mostly reflects the pricer's own demand-responsiveness
(it raises price *because* occupancy is already running high), not a
customer's actual response to price. Fitting "optimal price" as a
supervised regression against that history would launder that reverse
causality straight into a model and call the result ML pricing. It
wouldn't be one, and DESIGN.md §6.10/§6.11 already commit to not
shipping that.

What this DOES do, which the data fully supports: react to how a stay's
ACTUAL booking pace compares to the model's OWN expectation for this
exact lead time (days-until-stay) — a pace-deviation signal, the same
one classical (non-ML) revenue-management systems have used for decades,
except here "expectation" comes from a real, honestly-validated ML
demand forecast (this project's booking-curve model, §6.9) instead of a
hand-fit parametric curve. Running ahead of pace -> scarcity -> raise
price. Running behind -> raise the discount to stimulate demand. This
only requires the forecast to predict "how fast a stay like this
normally fills up," which has been validated end-to-end. It does NOT
require knowing how customers respond to price, which is exactly the
piece §6.10 found isn't supportable here — so this module never claims
to know that, and never tries to learn it.

Confidence gating: the adjustment is scaled by the model's own
calibrated P10/P90 interval width at the stay date — not a new metric
invented for pricing. A thin or out-of-season hotel (hotel_H-shaped)
gets a wide interval, and its pace signal is dampened toward "don't move
the price much, this forecast isn't trustworthy enough to act on
aggressively" — the `low_confidence` guardrail DESIGN.md §6.2/§6.11
already calls for, now actually wired to something.

Bounds: capped at +/-25%, chosen by looking at the empirical range of
Ampliphi's own historical suggested_price-vs-base_rate adjustments for
hotel_C (5th/95th percentile: about -20%/+29% —
`evaluation/price_demand_eda.py`'s own data) as a sanity reference for
"what an adjustment in this business actually looks like," not fit to
any revenue outcome — there is no revenue ground truth in this dataset
to fit a cap against, and pretending otherwise would be the same mistake
as fitting elasticity.

Only engages when a live pickup signal exists (`as_of_date` given, and
this stay already has at least one day of booking history to compare
against expectation). For a stay with no bookings yet there's no pace to
react to, and this returns the base rate unadjusted with an explicit
reason rather than inventing a signal from nothing.

See DESIGN.md §6.12 for the full design discussion and its own honest
GO/NO-GO — this is a recommendation-and-guardrail tool for a human
pricing decision, not an autonomous pricing engine.
"""

from __future__ import annotations

import numpy as np

from .predict import predict_booking_curve

# Business-rule constants, not fit to any outcome — see module docstring
# for where each one's magnitude actually comes from.
MAX_DISCOUNT_PCT = 0.25
MAX_PREMIUM_PCT = 0.25
BASE_SENSITIVITY = 0.35  # log(pace_ratio) multiplier at full confidence
MIN_DUS_FOR_PACE = 1  # dus_now=0 (arrival day itself): no runway left to react


def pace_adjustment(pace_ratio: float, confidence: float) -> tuple[float, bool]:
    """Pure function, the actual pricing rule — separated from
    `recommend_price`'s I/O (loading the model, computing the forecast)
    so it's directly unit-testable without a trained model fixture.
    Returns (adjustment_pct, was_capped).
    """
    raw = BASE_SENSITIVITY * confidence * np.log(max(pace_ratio, 1e-3))
    clipped = float(np.clip(raw, -MAX_DISCOUNT_PCT, MAX_PREMIUM_PCT))
    return clipped, bool(abs(raw - clipped) > 1e-9)


def recommend_price(
    hotel_id: str,
    room_type_code: str,
    stay_date: str,
    as_of_date: str,
    base_rate: float,
    *,
    data_dir: str | None = None,
    model_dir: str | None = None,
    model_version: str | None = None,
) -> dict:
    """Returns a recommendation dict: `recommended_price`, `base_rate`,
    `adjustment_pct`, `pace_ratio`, `confidence`, a human-readable
    `reason` (a revenue manager needs to see *why*, not just a number),
    and the underlying forecast `diagnostics` for audit.

    `base_rate` is the rate this recommendation adjusts *from* — where it
    comes from (a rate plan, a cost floor, last season's rate) is a
    separate rate-management concern this function doesn't model; it
    only decides how far to move off of it, and only when it has a real
    signal to justify moving at all.
    """
    if not base_rate or base_rate <= 0:
        raise ValueError(f"base_rate must be positive, got {base_rate!r}")

    result = predict_booking_curve(
        hotel_id,
        room_type_code,
        stay_date,
        as_of_date=as_of_date,
        data_dir=data_dir,
        model_dir=model_dir,
        model_version=model_version,
    )
    diag = result["diagnostics"]

    def _hold(reason: str) -> dict:
        return {
            "recommended_price": round(base_rate, 2),
            "base_rate": base_rate,
            "adjustment_pct": 0.0,
            "pace_ratio": None,
            "confidence": None,
            "reason": reason,
            "diagnostics": diag,
        }

    if diag.get("mode") != "anchored":
        return _hold(
            f"no live pickup signal to react to (mode={diag.get('mode')!r}) — "
            "holding at base_rate rather than inventing a demand signal from nothing"
        )

    dus_now = diag["dus_now"]
    if dus_now < MIN_DUS_FOR_PACE:
        return _hold(f"dus_now={dus_now} — too close to arrival to act on a price change")

    anchor_frac = diag["anchor_frac"]
    model_at_anchor = diag["model_at_anchor"]

    # pace_ratio > 1: booking faster than the model expected at this lead
    # time (scarcity signal, raise price). < 1: booking slower than
    # expected (stimulate demand, discount). Both sides floored away from
    # zero: a near-zero model expectation early in the curve would
    # otherwise make the ratio explode on a single early booking, and
    # anchor_frac itself can legitimately be exactly 0.
    pace_ratio = max(anchor_frac, 1e-3) / max(model_at_anchor, 0.02)

    p10 = result["intervals"]["p10"]["0"]
    p90 = result["intervals"]["p90"]["0"]
    interval_width = max(p90 - p10, 0.0)
    confidence = float(np.clip(1.0 - interval_width, 0.0, 1.0))

    adjustment_pct, capped = pace_adjustment(pace_ratio, confidence)
    recommended_price = round(base_rate * (1.0 + adjustment_pct), 2)

    direction = "ahead of" if pace_ratio > 1 else "behind"
    reason = (
        f"{direction} expected pace ({pace_ratio:.2f}x model's own expectation "
        f"at {dus_now}d out), confidence {confidence:.2f}"
        + (" — capped at bound" if capped else "")
    )

    return {
        "recommended_price": recommended_price,
        "base_rate": base_rate,
        "adjustment_pct": round(adjustment_pct, 4),
        "pace_ratio": round(float(pace_ratio), 4),
        "confidence": round(confidence, 4),
        "reason": reason,
        "diagnostics": diag,
    }


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hotel-id", required=True)
    ap.add_argument("--room-type-code", required=True)
    ap.add_argument("--stay-date", required=True)
    ap.add_argument("--as-of-date", required=True, help="Pricing needs a live pickup signal — unlike predict.py, not optional here.")
    ap.add_argument("--base-rate", required=True, type=float)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--model-version", default=None)
    args = ap.parse_args()

    result = recommend_price(
        args.hotel_id, args.room_type_code, args.stay_date, args.as_of_date, args.base_rate,
        data_dir=args.data_dir, model_dir=args.model_dir, model_version=args.model_version,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
