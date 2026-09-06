# Contributing

```bash
pip install -r requirements-dev.txt
pre-commit install   # optional but recommended — runs `ruff check src tests` before each commit
```

Before opening a PR:

```bash
ruff check src tests
pytest -q --cov=src --cov-report=term-missing
pip-audit -r requirements.txt -r requirements-dev.txt
```

CI runs all three, plus a Docker build and a live smoke test of the built
container, on every push — `main` requires `lint-and-test` and
`docker-build-and-smoke-test` to pass before a merge.

**The real Ampliphi dataset is never committed here** (see README "Data").
Tests run against `src/demo_data.py`'s synthetic fixture instead — add
new tests against that, not against a local copy of the real extract.

A dependency bump touching `pandas`, `numpy`, `scikit-learn`, or
`lightgbm` can change model output, not just behavior — retrain
(`python -m src.train`) and re-run `evaluation/evaluate.py` against it
before merging, don't just take a green CI as sufficient (CI trains
nothing; there's no real data for it to train on).
