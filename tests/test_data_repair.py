"""
Tests for the room-type dimension-table repair (src/data.py). This exists
because the real Ampliphi extract has reservations referencing room types
absent from room_types.csv (~24% of test-window curves would silently
vanish from evaluation without this) — see README's "data-quality" finding.
The synthetic fixture reproduces the same *shape* of problem without
touching the real data.
"""

import json

import pandas as pd

from src.data import load_static_context


def test_missing_room_type_gets_patched_with_inferred_inventory(tmp_path):
    hotels = pd.DataFrame(
        [{"hotel_id": "hotel_Q", "hotel_name": "Q", "city_label": "Q", "country": "US",
          "currency": "USD", "timezone": "America/New_York", "pms_type": "agora",
          "primary_rate_mode": "lowest", "lat_coarse": 0.0, "lon_coarse": 0.0,
          "region_type": "urban_core"}]
    )
    hotels.to_csv(tmp_path / "hotels.csv", index=False)

    # room_types.csv deliberately omits "rt_ghost" even though reservations
    # reference it — the exact defect found in the real extract.
    pd.DataFrame(
        [{"hotel_id": "hotel_Q", "room_type_code": "rt_known", "inventory_count": 5,
          "room_type_kind": "optimized", "display_position": 0}]
    ).to_csv(tmp_path / "room_types.csv", index=False)

    (tmp_path / "property_metadata.json").write_text(
        json.dumps({"hotel_Q": {"country": "US", "currency": "USD", "timezone": "America/New_York",
                                 "pms_type": "agora", "primary_rate_mode": "lowest", "total_rooms": 5,
                                 "room_types": {"rt_known": {"inventory_count": 5, "kind": "optimized"}}}})
    )

    # 3 reservations for rt_ghost overlap on exactly 2 nights at once
    # (peak concurrency = 2), so the repair should infer inventory_count=2.
    reservations = pd.DataFrame(
        [
            {"reservation_id": "r1", "hotel_id": "hotel_Q", "room_type_code": "rt_ghost",
             "stay_date": "2025-03-01", "checkout_date": "2025-03-05", "booking_date": "2025-02-01",
             "status": "checked out"},
            {"reservation_id": "r2", "hotel_id": "hotel_Q", "room_type_code": "rt_ghost",
             "stay_date": "2025-03-02", "checkout_date": "2025-03-04", "booking_date": "2025-02-02",
             "status": "checked out"},
            {"reservation_id": "r3", "hotel_id": "hotel_Q", "room_type_code": "rt_ghost",
             "stay_date": "2025-03-10", "checkout_date": "2025-03-11", "booking_date": "2025-02-03",
             "status": "checked out"},
        ]
    )
    reservations.to_csv(tmp_path / "reservations.csv", index=False)

    static = load_static_context(tmp_path)

    assert ("hotel_Q", "rt_ghost") in static.inventory_lookup
    assert static.inventory_lookup[("hotel_Q", "rt_ghost")] == 2
    assert "rt_ghost" in static.known_room_types("hotel_Q")
    # the pre-existing, correctly-declared room type is untouched
    assert static.inventory_lookup[("hotel_Q", "rt_known")] == 5


def test_no_missing_room_types_is_a_no_op(synthetic_data_dir):
    # The main synthetic fixture has no dimension-table gaps; repair should
    # leave it alone (also exercises the early-return path).
    static = load_static_context(synthetic_data_dir)
    assert static.known_room_types("hotel_X") == ["rt_x1"]
