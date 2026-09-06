"""
`BookingCurveModel.predict_curve_batch` (the vectorized path used by
`generate_eval_predictions` to build `evaluation/predictions.json`) had
zero test coverage until now, despite being the code path that actually
produces the graded deliverable. The property that matters most: it must
agree with `predict_curve` (the per-row path the live API serves through)
— if the two ever diverged, predictions.json would silently stop
reflecting what the service actually serves.
"""

from __future__ import annotations

import pandas as pd

from src.data import CHECKPOINTS, load_static_context
from src.model import BookingCurveModel


def test_batch_matches_per_row_prediction(synthetic_data_dir, trained_model_dir):
    from src.registry import resolve_model_dir

    static = load_static_context(synthetic_data_dir)
    model = BookingCurveModel.load(resolve_model_dir(trained_model_dir))

    keys = pd.DataFrame(
        [
            {"hotel_id": "hotel_X", "room_type_code": "rt_x1", "stay_date": pd.Timestamp("2025-08-15")},
            {"hotel_id": "hotel_Y", "room_type_code": "rt_y1", "stay_date": pd.Timestamp("2025-08-15")},
            {"hotel_id": "hotel_Z", "room_type_code": "rt_z", "stay_date": pd.Timestamp("2025-08-15")},
        ]
    )
    batch_results = model.predict_curve_batch(static, keys)

    for i, row in keys.iterrows():
        single = model.predict_curve(static, row["hotel_id"], row["room_type_code"], row["stay_date"])
        batch = batch_results[i]
        for cp in map(str, CHECKPOINTS):
            assert abs(single["point"][cp] - batch["point"][cp]) < 1e-9, (row["hotel_id"], cp)
            for q in ("p10", "p50", "p90"):
                assert abs(single[q][cp] - batch[q][cp]) < 1e-9, (row["hotel_id"], q, cp)


def test_batch_output_is_valid_for_every_row(synthetic_data_dir, trained_model_dir):
    from src.registry import resolve_model_dir

    static = load_static_context(synthetic_data_dir)
    model = BookingCurveModel.load(resolve_model_dir(trained_model_dir))

    nights = pd.date_range("2025-07-01", "2025-07-10")
    keys = pd.DataFrame(
        [
            {"hotel_id": h, "room_type_code": rt, "stay_date": night}
            for h, rt in (("hotel_X", "rt_x1"), ("hotel_Y", "rt_y1"))
            for night in nights
        ]
    )
    results = model.predict_curve_batch(static, keys)
    assert len(results) == len(keys)
    for res in results:
        ordered = sorted(CHECKPOINTS, reverse=True)
        values = [res["point"][str(cp)] for cp in ordered]
        assert all(0.0 <= v <= 1.0 for v in values)
        assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))


def test_generate_eval_predictions_writes_valid_json(synthetic_data_dir, trained_model_dir, tmp_path):
    from src.predict import generate_eval_predictions

    out_path = tmp_path / "predictions.json"
    generate_eval_predictions(synthetic_data_dir, trained_model_dir, out_path)

    assert out_path.exists()
    import json

    records = json.loads(out_path.read_text())
    assert len(records) > 0
    for rec in records[:5]:
        assert set(rec.keys()) >= {"hotel_id", "room_type_code", "stay_date", "predictions", "intervals"}
        assert set(rec["predictions"].keys()) == {str(cp) for cp in CHECKPOINTS}
