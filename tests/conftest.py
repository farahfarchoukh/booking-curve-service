"""
Shared test fixtures.

The dataset itself lives in `src/demo_data.py` — shared with
`scripts/generate_demo_data.py` so the fixture tests run against and the
dataset a human can actually explore are the exact same generator, not two
copies that can drift. See that module's docstring for what `hotel_X`,
`hotel_Y`, and `hotel_Z` each stand in for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.demo_data import write_demo_dataset


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    return write_demo_dataset(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, synthetic_data_dir) -> Path:
    """Trains once per test session on the synthetic fixture and returns
    the *base* (versioned-parent) model directory, with current.json
    already pointing at the trained version — exactly what predict.py
    expects."""
    from src.train import run_training

    base = tmp_path_factory.mktemp("model")
    run_training(synthetic_data_dir, base, version="test-fixture")
    return base
