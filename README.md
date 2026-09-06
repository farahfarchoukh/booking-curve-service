# Booking Curve Service

[![CI](https://github.com/farahfarchoukh/booking-curve-service/actions/workflows/ci.yml/badge.svg)](https://github.com/farahfarchoukh/booking-curve-service/actions/workflows/ci.yml)

A forecasting service that predicts, for any hotel/room-type/stay-date, the cumulative booking curve leading up to the stay — how full a room type will be at 90, 60, 45, 30, 21, 14, 7, 3, 1, and 0 days out. Built against anonymized production data from Ampliphi (a multi-tenant hotel revenue-management platform): 8 properties across 5 countries, 5 currencies, 2 PMS integrations, and reservation counts per hotel ranging from a few thousand to effectively zero.

The interesting part of this problem was never "fit a curve to one hotel's history." It's that two of the eight properties have any bookable history at all, they look nothing alike, and the service still has to say something sensible about the other six — and about hotel #401, whenever it signs up. That constraint shaped every decision below: a shared backbone model that never depends on knowing which hotel it's looking at, an explicit partial-pooling layer that decides how much to trust each tenant's own data, and a prediction interval that's honest about exactly where that trust runs out.

**Start with `DESIGN.md`** for the full engineering rationale, organized by decision (one model or many, cold start, feature provenance, multi-currency handling, serving infrastructure, production evaluation, uncertainty). Second: `src/model.py`'s docstring, which explains the level+shape decomposition everything else hangs off of.

**No access to the real data?** `python -m scripts.generate_demo_data` writes a small synthetic dataset with the same schema and prints the exact commands to train, predict, and serve against it — including a hotel that appears nowhere in the dataset at all, to see the cold-start path return a real answer. See "How to run" below.

```mermaid
flowchart LR
    A["Onboarding attributes plus calendar, no hotel_id"] --> B["Level model: final occupancy"]
    A --> C["Shape model: pace g of cp, monotonic in cp"]
    B --> D["y equals level times g of cp"]
    C --> D
    D --> E["Clip plus cummax: constraint enforcement"]
    E --> F["Booking curve: zero violations, by construction"]
    G["Per-hotel shrinkage plus conformal calibration"] -.-> B
    G -.-> C
```

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
# place Ampliphi's anonymized data extract at ./data, alongside src/ — not
# included in this repo (see "Data" below). No access to it? Run this
# instead and use its printed commands, pointed at ./demo_data:
#   python -m scripts.generate_demo_data

python -m src.train                              # trains + saves artifacts/model/<version>/, promotes it
python -m src.predict --hotel-id hotel_C \
    --room-type-code rt_ea30c05c4c --stay-date 2025-08-15   # single lookup
python -m src.predict --hotel-id hotel_C --room-type-code rt_ea30c05c4c \
    --stay-date 2025-08-15 --as-of-date 2025-08-01           # pickup-anchored lookup
python -m src.predict --generate-eval             # writes evaluation/predictions.json
python evaluation/evaluate.py --predictions evaluation/predictions.json --data-dir data
python evaluation/compare_production.py           # vs. baseline + production curve
python evaluation/interval_metrics.py              # PICP / pinball loss — the source of the numbers in DESIGN.md §6.7
```

Run everything from the repo root (module form `python -m src.train`, not `python src/train.py`, since the package uses relative imports).

```bash
pip install -r requirements-dev.txt   # fastapi/uvicorn, pytest, ruff, pip-audit, notebook tooling
pytest -q --cov=src --cov-report=term-missing  # 44 tests, ~15s, 94% coverage — synthetic fixture, never the real data
ruff check src tests                   # lint
pip-audit -r requirements.txt -r requirements-dev.txt  # dependency vulnerability scan
uvicorn src.api:app --reload           # serve locally without Docker
pre-commit install                     # optional: run the above lint automatically before each commit
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
- **Prediction intervals** (P10/P50/P90) via quantile LightGBM, conformally calibrated from out-of-fold residuals. In-distribution OOF coverage is 90%; true test-window coverage is honestly reported at 39.3% with a root-cause diagnosis (see `DESIGN.md` §6.7) — computed by `evaluation/interval_metrics.py`, checked into this repo, not a number asserted in prose. I chose to report this rather than hand-tune the interval to the test outcomes.
- **Shared feature code** (`src/features.py`) used identically by training and inference — the actual mechanism against train/serve skew, not a claim.
- **A data-quality fix I found, not one I was told about**: `reservations.csv` references 4 `room_type_code`s (655 + 536 reservations on hotel_C alone — not a rounding error) that don't exist in `room_types.csv`. Without patching this, ~24% of the real test-window curves are silently dropped from evaluation. `src/data.py::_repair_missing_room_types` detects and patches this (inferring inventory from peak concurrent bookings) and logs it loudly rather than failing silently.
- **`evaluation/compare_production.py`**: an honest three-way comparison (ours / baseline / Ampliphi's own production curve) that plain `evaluate.py` doesn't give you out of the box.

## Engineering hardening

Two rounds so far, each triggered by asking "what's still missing?" and then actually closing what came back, rather than leaving it as a list.

**Round 1 — is this production-ready at all:**

- **Determinism, verified by diffing bytes, not by re-reading code.** Retraining on identical data used to move weighted MAE by ~0.01 between runs. Root cause was two things stacking: `LightGBM`'s RNG wasn't seeded (`seed`/`deterministic`/`force_row_wise` are now set), *and* a training-table builder iterated a Python `set` whose order depends on the per-process hash seed — fixed by sorting it (`src/data.py::known_room_types`). Confirmed by training twice — locally and again inside the built Docker image — and diffing the resulting model files: byte-identical both times.
- **A model registry** (`src/registry.py`): `python -m src.train` saves to `artifacts/model/<timestamp>/` and atomically updates `artifacts/model/current.json`. Rollback is `--model-version <older-timestamp>` or editing that pointer directly — a real, testable mechanic, not a paragraph in DESIGN.md §6.5.
- **Structured logging** (`src/logging_config.py`) in place of bare `print()`, level controlled by `LOG_LEVEL`.
- **A hardened FastAPI service** (`src/api.py`): the model loads at startup, `/readyz` reflects real load state, every response carries an `X-Request-ID`, errors split into 400 vs. 500, a togglable API-key check stands in for real auth.
- **`Dockerfile` + `.dockerignore`**, one image serving or running batch jobs by overriding `CMD`. Building and actually running it surfaced two real bugs code review wouldn't have: `numpy==2.5.2` doesn't exist for Python 3.11 (only 3.12+); `python:3.11-slim` ships without `libgomp.so.1`, which LightGBM needs to even import.
- **CI/CD** (`.github/workflows/ci.yml`): lint + tests, Docker build + smoke test, then publish to `ghcr.io` on `main` using the workflow's own token.

**Round 2 — a second "what's still missing?" pass, closing what it found:**

- **`scripts/generate_demo_data.py`**: the gap behind "I can't see the output without the real data." Writes a synthetic dataset with the same schema (shared generator with the test fixture — `src/demo_data.py` — so they can't drift apart), and prints the exact commands to train, predict for a real hotel *and* for `hotel_Z` (which appears nowhere in the dataset — the true cold-start path), and serve it. Every command in its printed output was actually run to confirm it works verbatim, not just written.
- **`evaluation/interval_metrics.py`**: the PICP/pinball-loss numbers in DESIGN.md §6.7 were originally produced by a throwaway analysis script that was never committed — the claim wasn't actually reproducible from this repo. This is that script, for real, checked in with its output (`evaluation/interval_metrics.json`). Writing it for real caught a bug the throwaway version had papered over (a dict keyed inconsistently by quantile name vs. quantile value) — fixed, then re-verified the number it produces (39.3%) matches what was already in the docs, and surfaced a new finding along the way: `hotel_H`'s miscalibration is almost entirely one-directional (0% below p10, 61.2% above p90), which is *more* consistent with "one-directional level bias from an unseen season" than generic noise would be — now in DESIGN.md §6.7.
- **`tests/test_api.py`** (12 tests) and **`tests/test_model_batch.py`** (3 tests): the FastAPI service was previously verified only by hand with `curl`, and `predict_curve_batch` — the vectorized path that actually produces `evaluation/predictions.json` — had zero test coverage despite being the code path behind the graded deliverable. The batch test's most important assertion: it agrees with the per-row path the live API serves, checkpoint for checkpoint, quantile for quantile — if those two ever silently diverged, predictions.json would stop reflecting what the service actually serves.
- **Coverage measured, not guessed**: 94% (`pytest --cov=src`), up from an unmeasured baseline — `src/model.py` and `src/registry.py` are at 100%.
- **Dependency vulnerability scanning** (`pip-audit`, now in CI) — clean on both `requirements.txt` and `requirements-dev.txt` as of this writing, and now checked on every push, not just once by hand.
- **Dependabot** (`.github/dependabot.yml`) for pip, Docker, and GitHub Actions, with `pandas`/`numpy`/`scikit-learn`/`lightgbm` grouped into their own PR — a model-affecting dependency bump deserves a retrain + re-verify, not a drive-by merge alongside an unrelated `pydantic` bump.
- **Docker base image pinned by digest**, not just the `3.11-slim` tag — a floating tag means a rebuild next month silently pulls a different image with different CVEs than the one actually verified here.
- **A real graceful-shutdown bug, found by Docker's own linter and then actually fixed and verified**: the first attempt at a configurable worker count used shell-form `CMD` so `$WEB_CONCURRENCY` would expand — which wraps the process in `/bin/sh -c`, so `docker stop`'s `SIGTERM` hits the shell (which doesn't forward it) instead of uvicorn, forcing a hard kill after the grace period. Fixed with `docker-entrypoint.sh` (`exec uvicorn ...` — replaces the shell process instead of forking a child). Verified, not assumed: timed `docker stop` against the fixed image — 1.4s with uvicorn's own "Shutting down / Application shutdown complete" in the logs, not a 10s forced kill.
- **`BOOKING_CURVE_DATA_DIR` / `BOOKING_CURVE_MODEL_BASE_DIR` env vars** (`src/predict.py::_default_paths`) — the service was previously only configurable by editing code. This is also what let `tests/test_api.py` point a real `TestClient` at the synthetic fixture without monkeypatching internals.
- **CLI input validation** — a malformed `--stay-date` used to surface as a raw pandas traceback; the API already handled this correctly via Pydantic, the CLI didn't. Now it does.
- **`pre-commit` config**, scoped to match CI exactly (`ruff check` on `src`/`tests`, excluding notebooks and the byte-identical copy of Ampliphi's own `evaluate.py`) — deliberately *not* running an opinionated auto-formatter, which would have rewritten most of the repo for whitespace with no correctness value and fought the comments-next-to-code style used throughout. Consistency with CI was a deliberate choice, checked by actually running it, not assumed from the YAML.
- **Branch protection on `main`** (required status checks: `lint-and-test`, `docker-build-and-smoke-test`; no force-push, no deletion) and a **`CODEOWNERS`** file.

**What's still genuinely missing**, stated plainly rather than left implicit: no database (everything is local files — fine at this scale, not at "hundreds of hotels"), no message queue / scheduler / actual deployment anywhere, no auth beyond the placeholder header check, no metrics/tracing endpoint, no CORS or rate limiting, no automated retraining schedule, no monitoring or drift detection actually wired up (DESIGN.md §6.5/§6.6 describe what I'd build; none of it runs), no load testing, no secrets management, no API versioning, no multi-stage Docker build (checked whether one would help — it wouldn't: every heavy dependency here is a prebuilt wheel, nothing compiled to shed at a build stage). The registry and hardened API are real, working mechanics; they are not a production deployment.

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
  demo_data.py         # synthetic dataset generator — shared by tests/ and scripts/generate_demo_data.py
scripts/
  generate_demo_data.py  # writes a runnable demo dataset for anyone without the real extract
tests/                 # 44 tests, 94% coverage, all against the synthetic fixture — see conftest.py
artifacts/model/
  current.json         # pointer to the live version
  <timestamp>/          # one versioned artifact (level.txt, shape.txt, meta.json, ...)
evaluation/
  predictions.json      # test-window predictions, scored by evaluation/evaluate.py
  results.json           # evaluate.py output
  interval_metrics.py    # PICP / pinball loss — the source of DESIGN.md §6.7's numbers
  interval_metrics.json  # its output
  compare_production.py  # 3-way comparison script + its output above
notebooks/exploration.ipynb
presentation/index.html   # visual write-up (self-contained, open in a browser)
Dockerfile / .dockerignore / docker-entrypoint.sh
.github/workflows/ci.yml / dependabot.yml / CODEOWNERS
.pre-commit-config.yaml
requirements.txt / requirements-dev.txt
DESIGN.md
```
