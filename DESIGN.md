# DESIGN.md — Booking Curve Forecasting Service

*Extended validation (significance testing, seed sensitivity, backtest, ablation, calibration search, pricing, GO/NO-GO) lives in [VALIDATION.md](VALIDATION.md), kept separate so this stays a design doc, not a lab notebook. Every claim below is checked into `evaluation/`.*

## 6.1 One model or many?

**Global backbone + explicit per-hotel shrinkage — not per-hotel models, not raw hotel-identity as a feature.**

The data settles this. `hotel_C` has 1,463 trainable curves; `hotel_H` has 40 — essentially a cold-start case dressed up with off-season rows. A per-hotel model for `hotel_H` has nothing to fit, and at hundreds-of-hotels scale most new tenants will look like `hotel_H`. Raw `hotel_id` as a feature is worse than it looks: it only special-cases hotels already seen, degrading to the population mean for every unseen one — backwards, since those most need the shared model to generalize. So `src/features.py` never exposes `hotel_id`/`room_type_code` to the trees — only what's known at onboarding, before a reservation exists: `country`, `currency`, `pms_type`, `primary_rate_mode`, `region_type`, room-count scale, `room_type_kind`, calendar.

On top of it, an empirical-Bayes (James–Stein) per-hotel bias correction: `correction[h] = (n_h/(n_h+K)) · mean(OOF residual for h)`, fit separately for **level** (final occupancy, cp=0) and **shape** (fraction of final occupancy realized by each cp) — decomposed so shrinkage acts on one clean scalar per curve instead of 10 correlated checkpoints, and because level and shape transfer differently across seasons (§6.2). `hotel_C` (n=1,463, K=15) gets ~99% of its own signal back; `hotel_H` (n=40) gets ~73%; a brand-new hotel gets exactly the backbone. A hierarchical model's *behavior* — shrink to the population as n shrinks — without fitting one, since with 2 hotels there's nothing to estimate a group-level variance from. (Runs two levels deep in practice, within a hotel across room types too, and the decomposition itself is empirically validated against the simplest alternative, not just argued — VALIDATION.md §6.8.)

Caveat, checked not assumed (`evaluation/feature_importance.py`): with only 2 fitted hotels, `region_type`/`pms_type`/`total_rooms` get exactly zero split gain — collinear with `country`/`currency`/`room_share` here, so the mechanism is correctly wired but hasn't demonstrably learned the *right* attribute relationships yet. What the level model leans on: `doy_sin`/`doy_cos` (52% gain, seasonality) and `room_share`/`inventory_count` (32%, room-type scale); the shape model is 64% `cp` itself — lead time driving the pace curve, as it should.

## 6.2 Cold start

Day 0: a new hotel has a room dimension table and nothing else. `predict_booking_curve` handles this on the same code path as everything else — `hotel_n_obs` returns 0, shrinkage is 0, the curve is the attribute backbone alone.

The harder truth this dataset surfaces: `hotel_H` is a mountain resort whose busy season (Jul–Sep) is **entirely unseen** in its Apr–Jun training window, with no other resort-mountain property to borrow a season shape from. Measured, not assumed: OOF PICP is 90% in-distribution but only ~62% on the true test window (§6.7), one-sided — real seasonal shift, not a calibration bug. Widening the interval only partly helps, even asymmetrically (VALIDATION.md §6.11) — a symmetric interval can't fix a biased median. I did not hand-correct the bias using test outcomes; that's fitting the answer key, not solving cold start.

What I'd ship instead: (1) a `season_calendar` reference table (peak/shoulder/low by month) as a real feature — its absence is *why* hotel_H is hard here; (2) borrow *shape* (not level) from the nearest `region_type` peers once ≥3 exist per type; (3) flag predictions outside the hotel's observed day-of-year range as `low_confidence`. Graduation: week 1 = pure backbone; month 1 = shrinkage engages past ~15–40 curves; month 6, after a full seasonal cycle, a hotel's own weight dominates.

## 6.3 Features and the data plane

| Feature | Anchor | Leakage risk | Freshness | Availability |
|---|---|---|---|---|
| Calendar (dow, doy sin/cos, week) | `stay_date` | None | N/A | Universal |
| Hotel/room attributes | `hotels.csv`/`room_types.csv` | None (static) | Days | Universal |
| `cp` / days-until-stay | derived | None | N/A | Universal |
| Realized pickup-to-date | `reservations`, `as_of_date`-gated | **High** if the cutoff is wrong | Minutes | hotel_C/H here; any onboarded hotel in prod |

I excluded `daily_inventory`, `daily_hotel_demand`, `competitor_rates`, `suggested_prices`, and `expected_booking_curves` from training: every one covers *only* the Jul–Sep test window for hotel_B/C/F — zero training-period rows to learn a coefficient from, an untestable relationship (populated at inference, NaN at training) rather than classic leakage. The data dictionary's own warning about `daily_inventory.actual_occupancy` — at `as_of_date == stay_date` the snapshot *is* the cp=0 target — is the canonical version: joining an operational snapshot at the wrong instant. Caught structurally: `realized_fraction_as_of()` is the *only* function allowed to read post-cutoff activity, and `tests/test_leakage.py` asserts `predict_booking_curve(..., as_of_date=d)` is invariant to any reservation with `booking_date > d`, with a positive control proving the test would fail otherwise. Runs in CI on every push.

`expected_booking_curves` is an **evaluation baseline**, not a feature — same reason. `suggested_prices` is the pricer's own past output; as a feature it closes a feedback loop — useful for an off-policy read on the pricer (VALIDATION.md §6.10), never as a forecasting feature. `competitor_rates`/`hotel_comp_set` is sparse and its room-type label doesn't map to ours.

## 6.4 Multi-currency, multi-timezone, multi-PMS

`stay_date` is a calendar date **in the hotel's own local timezone** (`hotels.timezone`), never UTC — a stay is "the night of Aug 15 at this hotel." PMS heterogeneity (`agora` vs `cms`): `src/data.py` normalizes both into one `(hotel_id, room_type_code, stay_date, booking_date, checkout_date, status)` shape at ingestion; `pms_type` stays a feature since cancellation semantics and sync cadence still differ downstream.

I never touch price — `rate_amount` is stripped, and every rate table only covers the test window, so ADR isn't trainable regardless of currency. If it were: never train on raw currency units — `fx_rates(currency, date, usd_rate)`, converting every rate to USD *at the rate's own date* (today's FX rate on a March rate injects a macro trend into the target). `primary_rate_mode` (hotel_C's `average` vs. everyone else's `lowest`) is the same class of gotcha: two hotels' "rate" fields aren't comparable even in the same currency without knowing which aggregation produced them.

## 6.5 Training and serving infrastructure

Feature engineering lives in one module (`src/features.py`), imported by both `train.py` and `predict.py` — the actual guarantee against train/serve skew. Serving is hybrid: nightly batch-precompute every active curve into Postgres (what the pricer reads 99% of the time), plus a synchronous FastAPI path for on-demand refresh when a booking event needs a new `as_of_date` anchor now. Backbone retrains weekly; per-hotel shrinkage recomputes nightly (a residual mean, not a refit). New backbone: shadow 4 weeks, canary 10% of hotels, then promote. Rollback trigger: per-hotel weighted-MAE regression >20% sustained 3 days, or any monotonicity/bound violation (structurally impossible, so even one pages). Monitoring is **per-hotel** dashboards, not fleet aggregates — fleet MAE hides one rotting tenant.

Implemented, not just described: `src/registry.py` versions every run under `artifacts/model/<timestamp>/` and atomically flips a `current.json` pointer — rollback is a file write, `--model-version` pins an older one directly; `src/api.py` loads the model at startup and fails its own readiness check rather than a customer's first request. Postgres, an event bus, and an actual deploy don't exist yet.

Latency was profiled, not assumed: `model.py` was building the level feature matrix twice per request — sharing one build cut single-request latency 549ms → 179ms (byte-identical output, verified). Concurrent p95 barely moved from that alone (stayed 4.4s): FastAPI runs this sync, in a thread pool, and the GIL serializes the remaining pandas work regardless of per-request speed. Worker *processes* fixed that (`WEB_CONCURRENCY`, a Dockerfile knob, only now load-tested): p95 4.4s → 2.1s at 4 workers. For "hundreds of hotels": per-request micro-optimization hits a ceiling fast; horizontal scaling is the actual lever — infrastructure, not modeling.

## 6.6 Evaluation in production

A **switchback**, not a simple A/B: randomize which model (ours vs. `expected_booking_curves`) prices a hotel on a given day, rotating within-hotel — a between-hotel A/B mostly measures which hotels you got, visible in this dataset where hotel_C and hotel_H differ more from each other than any modeling choice would. Outcome is realized RevPAR delta, not curve MAE — a more accurate curve can still price worse if the pricer's response is miscalibrated. Drift: rolling MAE per `(hotel, cp)` vs. a frozen baseline, flagged at 3 consecutive days over its historical p90; a correction is "stale" once its sign disagrees with the rolling residual for 2+ weeks. Report the switchback effect with hotel as a random effect, alongside the per-hotel breakdown — an aggregate "+2% RevPAR" can hide +8% on six hotels and −3% on two.

## 6.7 Uncertainty

P10/P50/P90 via quantile LightGBM on the level target, **conformally calibrated** — each quantile shifted by the empirical quantile of its own out-of-fold residual, because a raw pinball-loss booster on this little, censored-at-1.0 data isn't automatically calibrated (a naive first pass measured 27% empirical PICP; root-caused and fixed rather than shipping a target-width interval that only *looks* like 80%). Post-calibration, in-distribution OOF PICP is 90%; on the true Jul–Sep window it's **62.4%** (`evaluation/interval_metrics.py`; was 39.3% until VALIDATION.md §6.8's damping fix), still one-sided: 34.2% of actuals exceed p90, only 3.4% fall below p10 — sharper for `hotel_H` (0%/44.2%) than `hotel_C` (5.7%/27.4%), consistent with §6.2's diagnosis. Reporting the residual gap rather than tuning to the answer key; the real fix is structural (`season_calendar`) — VALIDATION.md §6.11 shows widening the interval, symmetric or not, can't get there alone. For the pricer: P10/P90 as a confidence-scaled guardrail on yield aggressiveness (tight → trust the point estimate; wide/`low_confidence` → cap it), PICP + pinball loss reported weekly (mean pinball here: 0.088) — pinball rewards honestly-wide intervals over falsely-narrow ones.
