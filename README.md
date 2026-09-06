# Booking Curve Service

[![CI](https://github.com/farahfarchoukh/booking-curve-service/actions/workflows/ci.yml/badge.svg)](https://github.com/farahfarchoukh/booking-curve-service/actions/workflows/ci.yml)

A forecasting service that predicts, for any hotel/room-type/stay-date, the cumulative booking curve leading up to the stay — how full a room type will be at 90, 60, 45, 30, 21, 14, 7, 3, 1, and 0 days out. Built against anonymized production data from Ampliphi (a multi-tenant hotel revenue-management platform): 8 properties across 5 countries, 5 currencies, 2 PMS integrations, and reservation counts per hotel ranging from a few thousand to effectively zero.

The interesting part of this problem was never "fit a curve to one hotel's history." It's that two of the eight properties have any bookable history at all, they look nothing alike, and the service still has to say something sensible about the other six — and about hotel #401, whenever it signs up. That constraint shaped every decision below: a shared backbone model that never depends on knowing which hotel it's looking at, an explicit partial-pooling layer that decides how much to trust each tenant's own data, and a prediction interval that's honest about exactly where that trust runs out.

**Start with `DESIGN.md`** for the full engineering rationale, organized by decision (one model or many, cold start, feature provenance, multi-currency handling, serving infrastructure, production evaluation, uncertainty). Second: `src/model.py`'s docstring, which explains the level+shape decomposition everything else hangs off of.

## Where to look first

`evaluation/compare_production.py`'s output below is the fastest way in — it's the one place this model is compared against Ampliphi's own production heuristic, not just an internal baseline, and it's also where the most important honest finding (prediction-interval coverage collapses outside the training season) is easiest to see right next to the win.

## Presentation

`presentation/index.html` — open it directly in a browser (self-contained, no server needed). Covers the same ground as this README and `DESIGN.md` in a more visual form: the 8-hotel heterogeneity problem, the level+shape architecture, results vs. baseline and production curve, and the honest cold-start/interval-coverage finding.

## Results

Test window: 2025-07-01 → 2025-09-30, `hotel_C` + `hotel_H`, scored by `evaluation/evaluate.py` (Ampliphi's own scoring harness). Numbers below are from the exact artifact shipped in `artifacts/model/` — training is deterministic (see "Engineering hardening"), so re-running `python -m src.train` reproduces this artifact byte-for-byte rather than drifting run to run.

| Model | Overall MAE | Weighted MAE | Monotonicity violations | Bound violations |
|---|---:|---:|---:|---:|
| Heuristic baseline (`starter/baseline_model.py`) | 0.3478 | 0.3476 | 0.0% | 0.0% |
| **This model** | **0.3008** | **0.3061** | **0.0%** | **0.0%** |

~13–14% relative reduction in both MAE metrics, zero constraint violations (enforced structurally, not just empirically — see `enforce_curve_constraints` in `src/model.py`).

Head-to-head against Ampliphi's production parametric curve is only possible on `hotel_C`'s base room type (`rt_ea30c05c4c`) — the only room `expected_booking_curves.csv` covers for hotel_C (`python evaluation/compare_production.py`):

| Model | n | Overall MAE | Weighted MAE |
|---|---:|---:|---:|
| Heuristic baseline | 92 | 0.3117 | 0.3332 |
| Ampliphi `expected_booking_curves` (production) | 23* | 0.2798 | 0.2623 |
| **This model** | 92 | **0.1850** | **0.1766** |

\*production curve only has all 10 checkpoints present for 23 of the 92 test nights — small-sample, but directionally consistent with the full-grid result above.

Per-hotel breakdown (`evaluation/results.json`) is worth reading past the headline number: `hotel_C` MAE *improves* from 0.29 (90d out) to 0.18 (day-of) — the normal pattern for a hotel with real training history. `hotel_H` runs the other way — 0.11 (90d out) degrading to ~0.43 near the stay — because its Jul–Sep busy season is unseen in its own Apr–Jun training data, so the gap between our under-anchored level and true late-arriving demand *widens* as the stay approaches. This is diagnosed in depth in `DESIGN.md` §6.2/§6.7 and is the single most important honest finding in this project — not smoothed over.

## How to run

```bash
pip install -r requirements.txt
# place Ampliphi's anonymized data extract at ./data, alongside src/ —
# not included in this repo (see "Data" below)

python -m src.train                              # trains + saves artifacts/model/<version>/, promotes it
python -m src.predict --hotel-id hotel_C \
    --room-type-code rt_ea30c05c4c --stay-date 2025-08-15   # single lookup
python -m src.predict --hotel-id hotel_C --room-type-code rt_ea30c05c4c \
    --stay-date 2025-08-15 --as-of-date 2025-08-01           # pickup-anchored lookup
python -m src.predict --generate-eval             # writes evaluation/predictions.json
python evaluation/evaluate.py --predictions evaluation/predictions.json --data-dir data
python evaluation/compare_production.py           # vs. baseline + production curve
```

Run everything from the repo root (module form `python -m src.train`, not `python src/train.py`, since the package uses relative imports).

```bash
pip install -r requirements-dev.txt   # fastapi/uvicorn, pytest, ruff, notebook tooling
pytest -q                              # 29 tests, ~7s, against a synthetic fixture (never the real data)
ruff check src tests                   # lint
uvicorn src.api:app --reload           # serve locally without Docker
```

**Docker** (built and verified end-to-end — see "Engineering hardening"):

```bash
docker build -t booking-curve-service .

# serve
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/artifacts:/app/artifacts" \
  booking-curve-service

# train / predict as a batch job — same image, override CMD
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" -v "$(pwd)/artifacts:/app/artifacts" \
  booking-curve-service python -m src.train
```

On Windows with Git Bash specifically: prefix `docker run` with `MSYS_NO_PATHCONV=1`, or the `-v` paths get silently mangled into Windows paths before Docker sees them (`/app/data` becomes something like `C:/Program Files/Git/app/data`) and the container starts with no data mounted at all. PowerShell and a plain Linux/macOS shell don't have this problem.

## What's implemented

- **`predict_booking_curve(hotel_id, room_type_code, stay_date, as_of_date=None)`** exactly as specified (`src/predict.py`), including full `as_of_date` support: past checkpoints are returned as exact realized fractions from `reservations.csv`, future checkpoints are model-forecast and rescaled to connect continuously to the realized anchor (see the module docstring for the math). `as_of_date=None` is the blind ex-ante forecast used to generate `evaluation/predictions.json`, so the model is compared to the baseline and production curve on equal footing (none of them get to see realized pickup either).
- **Two-stage model** (`src/model.py`): final-occupancy "level" + booking-pace "shape" (monotonic-constrained in `cp` by construction), combined and then run through a hard constraint-enforcement layer (clip + cumulative-max) that guarantees zero bound/monotonicity violations regardless of what the model produced upstream.
- **Per-hotel partial pooling** via empirical-Bayes shrinkage on both stages — the mechanism that also answers cold start (§6.1/§6.2 of `DESIGN.md`).
- **Prediction intervals** (P10/P50/P90) via quantile LightGBM, conformally calibrated from out-of-fold residuals. In-distribution OOF coverage is 90%; true test-window coverage is honestly reported at ~38% with a root-cause diagnosis (see `DESIGN.md` §6.7) — I chose to report this rather than hand-tune the interval to the test outcomes.
- **Shared feature code** (`src/features.py`) used identically by training and inference — the actual mechanism against train/serve skew, not a claim.
- **A data-quality fix I found, not one I was told about**: `reservations.csv` references 4 `room_type_code`s (655 + 536 reservations on hotel_C alone — not a rounding error) that don't exist in `room_types.csv`. Without patching this, ~24% of the real test-window curves are silently dropped from evaluation. `src/data.py::_repair_missing_room_types` detects and patches this (inferring inventory from peak concurrent bookings) and logs it loudly rather than failing silently.
- **`evaluation/compare_production.py`**: an honest three-way comparison (ours / baseline / Ampliphi's own production curve) that plain `evaluate.py` doesn't give you out of the box.

## Engineering hardening

Everything below was added after a direct "is this actually production-ready?" review — the honest answer at that point was no: DESIGN.md described a production architecture that didn't exist in the code yet. This is what closed that gap, verified rather than assumed:

- **Determinism, verified by diffing bytes, not by re-reading code.** Retraining on identical data used to move weighted MAE by ~0.01 between runs. Root cause was two things stacking: `LightGBM`'s RNG wasn't seeded (`seed`/`deterministic`/`force_row_wise` are now set), *and* a training-table builder iterated a Python `set` whose order depends on the per-process hash seed — fixed by sorting it (`src/data.py::known_room_types`). Confirmed by training twice — locally and again inside the built Docker image — and diffing the resulting model files: byte-identical both times.
- **A real test suite** (`tests/`, 29 tests, pytest): constraint enforcement (fuzzed with 500 random inputs), the shrinkage-weight formula, the room-type dimension-table repair, the model version registry, `predict_booking_curve`'s contract (schema, bounds, monotonicity, the unseen-hotel cold-start path), and a leakage test proving `as_of_date` output is invariant to any reservation booked after it, with a positive control proving the test would actually fail if that weren't true. Runs against a synthetic fixture (`tests/conftest.py`) with the same schema as the real data — the proprietary extract itself is never committed to this repo.
- **A minimal model registry** (`src/registry.py`): `python -m src.train` saves to `artifacts/model/<timestamp>/` and atomically updates `artifacts/model/current.json`. Rollback is `--model-version <older-timestamp>` or editing that pointer directly — a real, testable mechanic, not a paragraph in DESIGN.md §6.5.
- **Structured logging** (`src/logging_config.py`) in place of bare `print()`, level controlled by `LOG_LEVEL`.
- **A hardened FastAPI service** (`src/api.py`): the model loads at startup (a broken artifact fails the readiness probe at deploy time, not on a customer's first request), `/readyz` reflects real load state, every response carries an `X-Request-ID`, errors split into 400 (bad input) vs. 500 (a bug, logged with a stack trace, never leaked to the caller), and a togglable API-key check (`BOOKING_CURVE_API_KEY`) stands in for real auth. Tested over live HTTP, not just unit tests: the happy path, an anchored `as_of_date`, a totally unseen hotel (cold start — confirmed 200, not a crash), and `/readyz` correctly reporting 503 with no model present.
- **`Dockerfile` + `.dockerignore`**, one image doing double duty — `docker run <image>` serves; override `CMD` to run `train` or `predict` as a batch job instead of maintaining a second image. Built and run end-to-end, and it surfaced two real bugs neither code review nor a local venv would have caught: `numpy==2.5.2` doesn't exist for Python 3.11 at all (only 3.12+) — repinned to the exact versions `pip` actually resolves inside the image; and `python:3.11-slim` ships without `libgomp.so.1`, which LightGBM's compiled core needs to even import — added `libgomp1` via `apt-get`. After both fixes: trained inside the container against the real data, generated predictions that scored identically to the local run (0.3008 / 0.3061, byte-for-byte reproducible on a second in-container run), and ran it as a live server — `/healthz`, `/readyz`, and a real `/booking-curve` call all verified over `curl`, with Docker's own `HEALTHCHECK` reporting `healthy`.
- **CI/CD** (`.github/workflows/ci.yml`): lint + the full test suite against the synthetic fixture, then a Docker build and a smoke test (starts the container, confirms `/healthz` is up and `/readyz` correctly reports 503 with no model mounted), then — on `main`, gated on both of those passing — publishes the image to `ghcr.io/<owner>/booking-curve-service` using the workflow's own built-in token, no external registry account to provision. Deliberately doesn't train on real data (never committed here) or deploy anywhere; that's the actual promotion path in DESIGN.md §6.5, not something to fake here.
- **Consolidated, pinned `requirements.txt`**, resolved and verified inside the actual target environment rather than assumed from a local dev shell on a different Python version — see the Docker bullet above for why that distinction turned out to matter.

**What's still genuinely missing**, stated plainly rather than left implicit: no database (everything is local files — fine at this scale, not at "hundreds of hotels"), no message queue / scheduler / actual deployment anywhere, no auth beyond the placeholder header check, no monitoring or drift detection actually wired up (DESIGN.md §6.5/§6.6 describe what I'd build; none of it runs), no load testing, no secrets management. The registry and hardened API are real, working mechanics; they are not a production deployment.

## What I'd build next

- A real `season_calendar` feature and cross-hotel-family conformal calibration — the two structural fixes for the hotel_H gap, both blocked on data this dataset doesn't include (see `DESIGN.md` §6.2/§6.7).
- Model-based (not just peak-concurrent-inferred) inventory reconciliation for the 4 orphaned room types, ideally by escalating the dimension-table gap upstream instead of silently patching it.
- A real off-policy evaluation of `suggested_prices` as the historical action — flagged in `DESIGN.md` §6.3 for why I'd want it gated behind more validation-window data first.
- Real cloud auth, a database, and monitoring in place of the current placeholders (see "What's still genuinely missing" above).

## Data

The `data/` folder (Ampliphi's anonymized production extract — 8 hotels, 5 countries, 5 currencies, 2 PMS integrations) is not included in this repo and is `.gitignore`d: it's proprietary, and a service like this should never depend on a real customer dataset being committed to source control. `tests/` runs against a synthetic fixture with the same schema instead — see `tests/conftest.py`. To run training/prediction yourself, place the extract at `./data`.

## Known data-quality notes (see `src/data.py` docstrings for the code-level version)

- ~2% of reservations have `booking_date > stay_date` (anonymization jitter, not real late bookings); clipped to `stay_date` for curve construction.
- A handful of `hotel_H` `booking_date`s land in 2026 — past the whole dataset. Left unclipped so training labels and the scoring script's ground truth stay defined the same way; this slightly understates true pickup for a few hotel_H reservations in both.
- `daily_inventory`/`daily_hotel_demand`/`competitor_rates`/`suggested_prices`/`expected_booking_curves` all only cover the *test* window for their listed hotels — confirmed by inspection, not assumed — which is why none of them are trained features (`DESIGN.md` §6.3).

## Repo layout

```
src/
  data.py            # raw loading, curve construction (train + inference share this), data-quality repair
  features.py        # shared feature engineering (train.py and predict.py both import this)
  model.py           # two-stage GBM, shrinkage, conformal calibration, constraint enforcement
  train.py           # CLI: builds curves -> tunes -> fits -> saves a versioned artifact -> promotes it
  predict.py         # predict_booking_curve(...) + CLI + batch eval-set generation
  api.py             # hardened FastAPI service (startup load, readyz, request ids, auth placeholder)
  registry.py         # versioned model artifacts + current.json pointer (rollback lever)
  logging_config.py   # structured logging setup shared by every entrypoint
tests/                 # 29 tests against a synthetic fixture — see conftest.py
artifacts/model/
  current.json         # pointer to the live version
  <timestamp>/          # one versioned artifact (level.txt, shape.txt, meta.json, ...)
evaluation/
  predictions.json      # test-window predictions, scored by evaluation/evaluate.py
  results.json           # evaluate.py output
  compare_production.py  # 3-way comparison script + its output above
notebooks/exploration.ipynb
presentation/index.html   # visual write-up (self-contained, open in a browser)
Dockerfile / .dockerignore
.github/workflows/ci.yml
requirements.txt / requirements-dev.txt
DESIGN.md
```
