"""
The as_of_date leakage test promised in DESIGN.md §6.3: a property test
asserting predict_booking_curve(..., as_of_date=d) is invariant to any
reservation booked after d. This is the exact class of bug the data
dictionary warns about for `daily_inventory.actual_occupancy` — reading a
snapshot that already contains information from after the cutoff.

Design: clone the synthetic fixture, add one reservation booked strictly
*after* a chosen as_of_date, and assert the prediction is byte-identical
to the untampered fixture's — for every checkpoint, not just the final
one. A positive control (moving as_of_date to *after* the new booking)
proves the test would actually fail if leakage were real, rather than
passing vacuously.
"""

import shutil

import pandas as pd
import pytest

from src.predict import predict_booking_curve

TARGET_STAY = "2025-04-10"  # a date with no pre-existing hotel_X reservations
LEAK_BOOKING_DATE = "2025-03-25"


@pytest.fixture()
def tampered_data_dir(tmp_path, synthetic_data_dir):
    dest = tmp_path / "tampered"
    shutil.copytree(synthetic_data_dir, dest)

    res = pd.read_csv(dest / "reservations.csv")
    leak_row = {
        "reservation_id": "res_LEAK", "hotel_id": "hotel_X", "room_type_code": "rt_x1",
        "rate_plan_code": "BAR", "stay_date": TARGET_STAY,
        "checkout_date": "2025-04-11", "booking_date": LEAK_BOOKING_DATE,
        "cancelled_at": "", "num_rooms": 1, "num_adults": 2, "num_children": "",
        "rate_amount": "", "total_amount": 100.0, "currency_code": "USD",
        "booking_source": "web", "status": "checked out",
    }
    res = pd.concat([res, pd.DataFrame([leak_row])], ignore_index=True)
    res.to_csv(dest / "reservations.csv", index=False)
    return dest


def test_booking_after_as_of_date_does_not_change_the_prediction(
    synthetic_data_dir, tampered_data_dir, trained_model_dir
):
    as_of_date = "2025-03-20"  # strictly before LEAK_BOOKING_DATE
    clean = predict_booking_curve(
        "hotel_X", "rt_x1", TARGET_STAY, as_of_date=as_of_date,
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    tampered = predict_booking_curve(
        "hotel_X", "rt_x1", TARGET_STAY, as_of_date=as_of_date,
        data_dir=str(tampered_data_dir), model_dir=str(trained_model_dir),
    )
    assert clean["predictions"] == tampered["predictions"], (
        "a reservation booked after as_of_date changed the output — leakage"
    )
    assert clean["diagnostics"].get("anchor_frac") == tampered["diagnostics"].get("anchor_frac")


def test_positive_control_booking_before_as_of_date_does_change_it(
    synthetic_data_dir, tampered_data_dir, trained_model_dir
):
    # Proves the test above isn't vacuous: once as_of_date is *after* the
    # leaked booking, it's legitimately observed and the curve must move.
    as_of_date = "2025-03-30"  # strictly after LEAK_BOOKING_DATE
    clean = predict_booking_curve(
        "hotel_X", "rt_x1", TARGET_STAY, as_of_date=as_of_date,
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    tampered = predict_booking_curve(
        "hotel_X", "rt_x1", TARGET_STAY, as_of_date=as_of_date,
        data_dir=str(tampered_data_dir), model_dir=str(trained_model_dir),
    )
    assert clean["predictions"] != tampered["predictions"]
