# INTERVIEW_PREP.md — defending this submission

Short, specific answers grounded in this repo's actual files/numbers — not the generic version of each answer. Where a number appears, it's in `EXPERIMENTS.md` or a checked-in `evaluation/*.json`.

## ML

**Why LightGBM?** Small tabular dataset (1,503 level curves), mixed categorical/numeric features, need for monotone constraints (`cp`) and fast iteration on a laptop. Tree ensembles win on tabular data this size almost every published benchmark; LightGBM specifically for native categorical handling and `monotone_constraints`, which XGBoost/CatBoost also have but LightGBM's leaf-wise growth fits this row count faster.

**Why not a Transformer / deep learning?** Would need orders of magnitude more data to beat a tuned GBM here, and it's directly against the brief's own instruction not to reach for sophistication the data doesn't justify. `evaluation/architecture_comparison.py` shows even the *simpler* GBM alternative (one model, no decomposition) was competitive — there was never a signal that model capacity was the bottleneck.

**Why a global model, not per-hotel?** `hotel_C` has 1,463 curves, `hotel_H` has 40 — a per-hotel model for `hotel_H` has nothing to fit, and at "hundreds of hotels" scale most new tenants start exactly there. DESIGN.md §6.1. The alternative (raw `hotel_id` as a feature) is worse than it looks: degrades to the population mean for every hotel not yet seen, backwards from what a cold-start-heavy system needs.

**How did you prevent leakage?** Three layers: (1) features are attribute-based only, no aggregate touches post-cutoff data by construction; (2) `realized_fraction_as_of()` is the *only* function allowed to read post-`as_of_date` reservation activity; (3) `tests/test_leakage.py` is a property test — `predict_booking_curve(..., as_of_date=d)` must be invariant to any reservation with `booking_date > d` — with a positive control proving the test would actually fail if that weren't true. Runs in CI on every push.

**Why this validation strategy?** Time-aware throughout: internal round-tuning uses the last 21 days of the train window, never a random split. `evaluation/rolling_backtest.py` adds 4 walk-forward folds. `evaluation/calibration_tuning.py` and `evaluation/architecture_comparison.py` both tune only on folds that exclude the official split — nested CV, not "peek at the test set and pick the best."

**Why weighted MAE as the metric?** It's the grader's own metric (`evaluation/evaluate.py`, checkpoint weights rising toward day-of-stay because near-term accuracy matters more for pricing decisions) — I didn't design the metric, I validated against it honestly rather than optimizing a proxy.

**How do you handle monotonicity?** A hard post-hoc layer (`enforce_curve_constraints`: clip to [0,1], then cumulative max from cp=90 toward cp=0) applied to point AND all three quantile curves, independent of model quality — 0% violations by construction, not by hoping the model learned it. The shape booster is *also* monotone-constrained during training (`monotone_constraints=-1` on `cp`) as a second layer, so the hard constraint is rarely doing heavy lifting in practice.

**How do you handle cold start?** Same code path as everything else: `hotel_n_obs` returns 0 for an unseen hotel, empirical-Bayes shrink weight `n/(n+K)` is exactly 0, prediction is the pure attribute-based backbone. Not a special case — the mechanism that handles a hotel with 40 curves and a hotel with 0 are the same formula at different `n`.

**How would you improve the model with more data?** Two concrete asks, both already scoped: a `season_calendar` feature (peak/shoulder/low by month) — the single highest-leverage fix for hotel_H's out-of-season gap; and enough additional hotels per `region_type` to pool interval calibration across them instead of per-hotel.

**How do you quantify uncertainty?** Quantile LightGBM (P10/P50/P90), conformally calibrated (each quantile shifted by the empirical quantile of its own OOF residual — a raw pinball-loss booster on this little data measured 27% PICP uncalibrated). In-distribution OOF coverage 90%; true test-window coverage 62% (was 39% before finding and fixing the extrapolation-damping bug — see below). Honestly short of the 80% target, reported as such.

## Statistics

**What assumptions are you making?** That a hotel's attributes (country, currency, PMS, region type, room-count scale) carry transferable signal about its booking behavior even with only 2 hotels to estimate that relationship from — `evaluation/feature_importance.py` shows `region_type`/`pms_type`/`total_rooms` get exactly zero split gain at this sample size, i.e., the model can't yet tell those apart from `country`/`currency`/`room_share`. I state that limitation rather than pretend the 2-hotel sample resolved it.

**What happens under distribution shift?** Directly observed, not hypothetical: the entire test window (Jul–Sep) is a season the training window (Mar–Jun) never saw. `evaluation/rolling_backtest.py` shows the honest cost of that (hotel_H error 0.33→0.20 as season-overlap grows) and the ablation study (`evaluation/ablation_study.py`) found the per-hotel bias correction was actively *worse* than no correction under exactly this shift, until damped by extrapolation distance.

**How do you calibrate intervals?** Conformal shift from OOF residuals (see above) plus an extrapolation-distance widening term for dates outside the observed training day-of-year range. `evaluation/calibration_tuning.py` searched for better widening constants via nested CV and found the search doesn't converge — the mechanism is symmetric, the real miscalibration is one-sided (actuals miss high far more than low), so a shared widen factor is structurally the wrong shape. Reported, not shipped as if solved.

**How would you detect concept drift?** DESIGN.md §6.5/§6.6: rolling 7-day weighted MAE per hotel vs. a frozen baseline (per-hotel, not fleet-aggregate — a fleet average hides one rotting tenant), plus a same-day proxy that doesn't need resolved stays yet — KS-test drift on the `final_occupancy_hat` distribution vs. a trailing 30-day window. Described, not wired to live traffic (there is none).

## MLOps

**How do you version models?** `src/registry.py` — every `train.py` run saves to `artifacts/model/<timestamp>/` and atomically flips a `current.json` pointer. `--model-version` pins an older one directly; rollback is a file write, not a redeploy. Real and tested (`tests/test_registry.py`), not described.

**How do you reproduce a prediction?** Deterministic training (`seed`/`deterministic`/`force_row_wise` on every LightGBM call) — verified by retraining twice and byte-diffing the model files, both locally and inside the built Docker image. A prediction from a pinned model version + the same input is exactly reproducible; `meta.json` records the seed, rounds, train window, and the calibration constants used.

**How do you deploy a new model?** Described: shadow against the last 4 weeks, canary on 10% of hotels comparing weighted MAE and violation rate, promote via the registry pointer. The registry mechanics are real; the shadow/canary traffic-splitting infrastructure isn't built (no live deployment to build it against) — stated plainly in README "What's still genuinely missing," not left implicit.

**How do you roll back?** `--model-version <older-timestamp>` or editing `current.json` directly — same mechanism as promotion, just pointed backward. Trigger described (per-hotel weighted-MAE regression >20% sustained 3 days, or any monotonicity/bound violation) but not wired to an automated watcher.

**How do you monitor model degradation?** See "concept drift" above — designed, not live. What *is* live: CI's `pipeline-health.yml` runs the full train→predict→serve chain on synthetic data weekly, catching a pipeline break (not a quality regression) since real data can never be in CI.

**How do you prevent training-serving skew?** One feature module (`src/features.py`), imported by both `train.py` and `predict.py` — the actual guarantee, not a policy. `tests/test_model_batch.py` separately asserts the vectorized batch-prediction path (what generates `evaluation/predictions.json`) agrees checkpoint-for-checkpoint with the per-row path the live API serves — if those two ever silently diverged, the shipped predictions file would stop reflecting what the service actually returns.

## System design

**How does this scale to 500 hotels?** The backbone doesn't change — it never depends on hotel count, only on attributes any hotel has at onboarding. What changes: per-hotel shrinkage recomputes cheaply (a residual mean, not a refit) so 500 hotels' corrections aren't 500x the training cost; the local-file artifact store and CSV-loading data layer don't scale past a few hotels and would need a real feature store/database (README "What's still genuinely missing").

**What happens when PMS schemas differ?** `src/data.py` normalizes both PMS schemas (`agora`/`cms`) into one canonical `(hotel_id, room_type_code, stay_date, booking_date, checkout_date, status)` shape at ingestion; `pms_type` stays a *feature* (cancellation semantics and sync cadence genuinely differ) but never a branch in the model logic itself. DESIGN.md §6.4.

**What happens when data is missing?** Two real, found-not-assumed examples: `reservations.csv` references 4 room types absent from `room_types.csv` (655+536 reservations on hotel_C alone) — `src/data.py::_repair_missing_room_types` infers inventory from peak concurrent bookings and logs loudly rather than silently dropping ~24% of test curves. And every price-side table has zero rows for hotel_H — `evaluation/price_demand_eda.py` states that gap explicitly rather than quietly training on hotel_C's price data and implying it generalizes.

**What happens when inference fails?** `/readyz` fails the container's own health check rather than serving from a null model; 400 for a bad/unrecognized input the service understands, 500 (with the real error logged server-side, not leaked to the caller) for anything else. `tests/test_api.py` covers both paths.

**How do you handle latency?** Profiled, not assumed — `cProfile` found the "~13 sequential calls" figure quoted in an earlier pass was never measured (real count: 5), and the actual cost was building the same feature matrix twice per request. Fixed: 549ms→179ms single-request. Concurrent p95 needed a different fix (worker processes, not per-request speed — the GIL serializes thread-pooled work regardless): 4.4s→2.1s at 4 workers. `EXPERIMENTS.md` table 8 has the full before/after.

**How do you handle concurrent requests?** FastAPI's sync endpoint runs in a thread pool; real throughput scaling is via `WEB_CONCURRENCY` (uvicorn worker processes, already a Dockerfile knob, only now actually load-tested) rather than async rewrites — the bottleneck was CPU-bound Python/pandas work the GIL serializes, which more threads in one process can't fix but more processes can.

## Pricing intelligence

**Why isn't forecast accuracy enough?** A more accurate curve can still price worse if the pricer's response to it is miscalibrated (DESIGN.md §6.6) — the forecast is an input to a decision, not the decision. This is exactly why `src/pricing.py` exists as a separate, explicit layer rather than assuming "good forecast → good price" needs no further design.

**How would this eventually become a pricing policy?** Today: a bounded, confidence-gated pace-deviation rule (`src/pricing.py`) — not a policy in the RL/bandit sense, an auditable heuristic layered on a validated demand signal. The honest path to a real policy needs price-experimentation data this dataset doesn't have (§below).

**How would you evaluate a pricing recommendation offline?** You mostly can't, with this data — there's no counterfactual demand at a different price. `evaluation/price_demand_eda.py` is explicit that this dataset supports *behavior* validation (does the recommendation move the right direction, stay bounded, get dampened by low confidence) but not *accuracy* validation (was this actually the right price) — no revenue ground truth exists to score against.

**What is the danger of counterfactual estimation?** Exactly what `evaluation/price_demand_eda.py` found empirically: `suggested_prices.csv` shows a *positive* correlation between price and occupancy for hotel_C's base room (ρ=0.53, p=0.008, n=24) — read naively as a demand curve, that says raising price sells more rooms. It's reverse causality: the pricer raises price *because* occupancy is already running high. Any counterfactual estimate built on this kind of observational, policy-generated data risks learning the policy's own logic instead of true demand response, unless you have genuine experimentation or a valid instrument — neither exists here.

**How would contextual bandits eventually fit in?** Only after a switchback (DESIGN.md §6.6) generates real price-randomized outcome data — bandits need a reward signal from actual price variation, which is precisely what's missing today. Building one now, on `suggested_prices.csv`, would bake in the same reverse-causality problem the EDA already flagged. Sequenced correctly: switchback first → real experimentation data → *then* a bandit or a proper elasticity model, not before.
