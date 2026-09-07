"""
Walkthrough of src/pricing.py against real Ampliphi data — not an
accuracy benchmark. There is no ground truth in this dataset for "was
this the right price" (VALIDATION.md §6.10: no genuine price experimentation exists
here), so unlike evaluate.py/interval_metrics.py this script does not,
and should not, produce a score. What it demonstrates instead: the
mechanism behaves the way VALIDATION.md §6.12 says it does, on real stays —
ahead-of-pace raises price, behind-pace discounts, a thin/uncertain
hotel gets dampened toward no change, and a stay with no live pickup yet
holds at base rate rather than inventing a signal.

Usage: python evaluation/pricing_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pricing import recommend_price  # noqa: E402

DATA_DIR = str(ROOT / "data")
BASE_RATE = 220.0  # illustrative reference rate — this demo isn't about the number

SCENARIOS = [
    # (label, hotel_id, room_type_code, stay_date, as_of_date)
    ("hotel_C, in-season (Mar-Jun train window itself)", "hotel_C", "rt_11ef219078", "2025-05-15", "2025-05-01"),
    ("hotel_C, well ahead of pace check", "hotel_C", "rt_11ef219078", "2025-08-29", "2025-08-01"),
    ("hotel_C, near arrival", "hotel_C", "rt_11ef219078", "2025-07-29", "2025-07-25"),
    ("hotel_C, early lead time (little pickup yet)", "hotel_C", "rt_11ef219078", "2025-09-15", "2025-08-20"),
    ("hotel_H, thin-data hotel (confidence check)", "hotel_H", None, "2025-08-15", "2025-08-01"),
    ("hotel_C, far future (no live signal at all)", "hotel_C", "rt_11ef219078", "2025-09-15", "2025-05-01"),
]


def resolve_room_type(hotel_id, room_type_code):
    if room_type_code is not None:
        return room_type_code
    import pandas as pd

    res = pd.read_csv(Path(DATA_DIR) / "reservations.csv")
    sub = res[res.hotel_id == hotel_id]
    return sub.room_type_code.value_counts().index[0]


def main():
    print("=" * 78)
    print("  PRICING DEMO — src/pricing.py against real hotel_C / hotel_H data")
    print("  (a behavior walkthrough, NOT an accuracy benchmark — see module docstring)")
    print("=" * 78)

    for label, hotel_id, room_type_code, stay_date, as_of_date in SCENARIOS:
        rt = resolve_room_type(hotel_id, room_type_code)
        rec = recommend_price(
            hotel_id, rt, stay_date, as_of_date, BASE_RATE, data_dir=DATA_DIR,
        )
        print(f"\n{label}")
        print(f"  {hotel_id}/{rt}  stay={stay_date}  as_of={as_of_date}")
        print(f"  base_rate=${rec['base_rate']:.2f}  ->  recommended=${rec['recommended_price']:.2f}"
              f"  ({rec['adjustment_pct']:+.1%})")
        if rec["pace_ratio"] is not None:
            print(f"  pace_ratio={rec['pace_ratio']:.2f}  confidence={rec['confidence']:.2f}")
        print(f"  reason: {rec['reason']}")

    print(
        "\nNote: every real hotel_C/hotel_H stay in this dataset with actual pickup"
        " history to demo against falls in Jul-Sep — which is entirely outside the"
        " Mar-Jun training window (train ends before test begins, by this project's"
        " own split). That's exactly the extrapolation regime DESIGN.md §6.7 / VALIDATION.md §6.8"
        " already documents as low-confidence, so most scenarios above correctly"
        " show confidence near 0 and barely move off base_rate — the guardrail"
        " engaging, not a bug. The first scenario (an in-season May date) shows"
        " what a well-supported prediction's confidence actually looks like"
        " (~0.7 vs ~0.01-0.14 out of season) for contrast."
    )

    print("\n" + "=" * 78)
    print("  Try your own scenario:")
    print("  python -m src.pricing --hotel-id hotel_C --room-type-code <code> \\")
    print("      --stay-date 2025-08-15 --as-of-date 2025-08-01 --base-rate 220")
    print("=" * 78)


if __name__ == "__main__":
    main()
