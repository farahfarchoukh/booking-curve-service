"""
Price-demand EDA — the piece that was missing entirely: this is a
"Pricing Intelligence" role, and until now nothing in this repo looked at
`suggested_prices.csv` or `competitor_rates.csv` at all.

The honest headline finding comes before any correlation number: neither
`daily_inventory.csv` nor `suggested_prices.csv` nor `competitor_rates.csv`
has a single row for hotel_H — every price-side table in this dataset
covers only hotel_B, hotel_C, hotel_F. hotel_H (one of the two hotels we
actually have booking-curve ground truth for) has ZERO price information
anywhere. So a price-demand analysis is necessarily hotel_C-only here,
and it does not generalize to hotel_H by construction, not by omission.

For hotel_C, `daily_inventory.csv`'s own `current_price`/`actual_occupancy`
columns are entirely null/zero — unusable. `suggested_prices.csv` has
usable `suggested_price`/`base_rate` and (via available_count/total_rooms)
an occupancy snapshot at pricing time, but `calculated_occupancy` is 91%
null, so this script derives occupancy itself rather than trusting that
column, and joins against our own ground-truth final occupancy
(build_actual_curve_table's cp=0 rows — the same construction used
everywhere else in this repo) rather than the pricer's own snapshot.

Usage: python evaluation/price_demand_eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import build_actual_curve_table, load_reservations, load_static_context  # noqa: E402
from src.train import TRAIN_FLOOR  # noqa: E402


def main():
    data_dir = ROOT / "data"
    static = load_static_context(data_dir)
    reservations = load_reservations(data_dir)

    sp = pd.read_csv(data_dir / "suggested_prices.csv", parse_dates=["stay_date"])
    cr = pd.read_csv(data_dir / "competitor_rates.csv", parse_dates=["stay_date"])
    price_hotels = set(sp.hotel_id) | set(cr.hotel_id)
    modeled_hotels = {"hotel_C", "hotel_H"}
    print(f"\nHotels with ANY price-side data (suggested_prices/competitor_rates): {sorted(price_hotels)}")
    print(f"Hotels this repo has booking-curve ground truth for: {sorted(modeled_hotels)}")
    print(f"-> Overlap: {sorted(price_hotels & modeled_hotels)}  "
          f"(no-overlap hotels get NO price-demand read from this data: "
          f"{sorted(modeled_hotels - price_hotels)})")

    # Ground-truth final occupancy, same construction as everywhere else.
    long_df = build_actual_curve_table(
        reservations, static, TRAIN_FLOOR, "2025-09-30", hotels=["hotel_C"]
    )
    final_occ = long_df[long_df.cp == 0][["hotel_id", "room_type_code", "stay_date", "actual"]].rename(
        columns={"actual": "final_occupancy"}
    )

    spc = sp[sp.hotel_id == "hotel_C"].copy()
    spc["own_price"] = spc["suggested_price"].fillna(spc["base_rate"])

    joined = spc.merge(final_occ, on=["hotel_id", "room_type_code", "stay_date"], how="inner")
    print(f"\nsuggested_prices rows for hotel_C: {len(spc)}")
    print(f"Joined to ground-truth final occupancy: {len(joined)} rows")
    print(f"  by pricer_type: {joined.pricer_type.value_counts().to_dict()}")

    # --- relative price position vs same-night competitor median ---------
    crc = cr[cr.hotel_id == "hotel_C"]
    comp_median = crc.groupby("stay_date")["rate_value"].median().rename("comp_median_rate")
    joined = joined.merge(comp_median, on="stay_date", how="left")
    joined["price_vs_comp"] = joined["own_price"] / joined["comp_median_rate"]
    n_with_comp = joined["comp_median_rate"].notna().sum()
    print(f"  rows with a same-night competitor median available: {n_with_comp}/{len(joined)}")

    results = {}

    def report(x_col, label):
        sub = joined[[x_col, "final_occupancy"]].dropna()
        n = len(sub)
        if n < 10:
            print(f"\n{label}: only {n} usable rows — too few for a defensible correlation read, skipping.")
            results[label] = {"n": n, "skipped": True}
            return
        rho, p = stats.spearmanr(sub[x_col], sub["final_occupancy"])
        print(f"\n{label} (n={n}):")
        print(f"  Spearman rho = {rho:.3f}, p = {p:.4f}")
        results[label] = {"n": n, "spearman_rho": float(rho), "p_value": float(p)}

    report("own_price", "Own price level vs final occupancy")
    report("price_vs_comp", "Price relative to same-night competitor median vs final occupancy")

    # Split by pricer_type: "optimized" (base room, priced first) vs
    # "derived" (priced as a delta off it) — do they show different
    # price-demand behavior? Directly relevant to the hierarchical
    # room-type pooling decision in model.py.
    print("\nBy pricer_type:")
    for pt, sub in joined.groupby("pricer_type"):
        sub2 = sub[["own_price", "final_occupancy"]].dropna()
        if len(sub2) >= 10:
            rho, p = stats.spearmanr(sub2["own_price"], sub2["final_occupancy"])
            print(f"  {pt} (n={len(sub2)}): rho={rho:.3f}, p={p:.4f}")
            results[f"pricer_type={pt}"] = {"n": len(sub2), "spearman_rho": float(rho), "p_value": float(p)}
        else:
            print(f"  {pt} (n={len(sub2)}): too few rows")
            results[f"pricer_type={pt}"] = {"n": len(sub2), "skipped": True}

    print(
        "\nReading: a negative rho (higher price, lower final occupancy) is the"
        " textbook demand-curve direction. A rho near zero or positive with a"
        " large p-value is NOT strong evidence of no relationship at this n —"
        " it's as consistent with 'not enough data to see it' as with 'no"
        " effect', especially split by pricer_type where n drops below 30."
    )

    out_path = ROOT / "evaluation" / "price_demand_eda.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
