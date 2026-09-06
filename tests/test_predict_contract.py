"""
Contract tests for predict_booking_curve — the function the take-home
brief specifies verbatim. These run against a model trained on the
synthetic fixture (see conftest.py), not the real Ampliphi extract.
"""

from src.data import CHECKPOINTS
from src.predict import predict_booking_curve


def _assert_valid_curve(curve: dict):
    ordered = sorted(CHECKPOINTS, reverse=True)
    values = [curve[str(cp)] for cp in ordered]
    assert all(0.0 <= v <= 1.0 for v in values), values
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:])), values


def test_returns_required_schema(synthetic_data_dir, trained_model_dir):
    result = predict_booking_curve(
        "hotel_X", "rt_x1", "2025-08-15",
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    assert set(result.keys()) >= {"hotel_id", "room_type_code", "stay_date", "predictions"}
    assert result["hotel_id"] == "hotel_X"
    assert result["room_type_code"] == "rt_x1"
    assert result["stay_date"] == "2025-08-15"
    assert set(result["predictions"].keys()) == {str(cp) for cp in CHECKPOINTS}


def test_known_hotel_curve_is_valid(synthetic_data_dir, trained_model_dir):
    result = predict_booking_curve(
        "hotel_X", "rt_x1", "2025-08-15",
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    _assert_valid_curve(result["predictions"])


def test_thin_data_hotel_curve_is_valid(synthetic_data_dir, trained_model_dir):
    # hotel_Y has 3 reservations total — the cold-start-adjacent case.
    result = predict_booking_curve(
        "hotel_Y", "rt_y1", "2025-08-15",
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    _assert_valid_curve(result["predictions"])
    assert result["diagnostics"]["hotel_n_train_obs"] < 10


def test_completely_unseen_hotel_does_not_crash(synthetic_data_dir, trained_model_dir):
    # hotel_Z exists nowhere in training data or room_types.csv — the true
    # cold-start case: signing up tomorrow, zero history anywhere.
    result = predict_booking_curve(
        "hotel_Z", "rt_z1", "2025-08-15",
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    _assert_valid_curve(result["predictions"])
    assert result["diagnostics"]["known_hotel"] is False
    assert result["diagnostics"]["hotel_n_train_obs"] == 0


def test_intervals_are_ordered_and_in_bounds(synthetic_data_dir, trained_model_dir):
    result = predict_booking_curve(
        "hotel_X", "rt_x1", "2025-08-15",
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    for cp in map(str, CHECKPOINTS):
        lo = result["intervals"]["p10"][cp]
        med = result["intervals"]["p50"][cp]
        hi = result["intervals"]["p90"][cp]
        assert 0.0 <= lo <= med <= hi <= 1.0, (cp, lo, med, hi)


def test_far_future_as_of_date_matches_blind_mode(synthetic_data_dir, trained_model_dir):
    # as_of_date more than 90 days before the stay: no checkpoint has
    # happened yet, so this must be identical to the blind forecast.
    blind = predict_booking_curve(
        "hotel_X", "rt_x1", "2025-08-15",
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    anchored = predict_booking_curve(
        "hotel_X", "rt_x1", "2025-08-15", as_of_date="2025-04-01",
        data_dir=str(synthetic_data_dir), model_dir=str(trained_model_dir),
    )
    assert blind["predictions"] == anchored["predictions"]
