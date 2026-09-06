#!/usr/bin/env python
"""
Write a synthetic demo dataset with the same schema as Ampliphi's real
extract, so you can run the full train -> predict -> serve pipeline and
see real output without needing access to the proprietary data.

Usage (run as a module, like every other entrypoint in this repo, so the
package's relative imports resolve):
    python -m scripts.generate_demo_data [--out demo_data]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.demo_data import write_demo_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo_data", help="Output directory (default: demo_data)")
    args = ap.parse_args()

    out = write_demo_dataset(Path(args.out))
    print(f"Wrote a synthetic demo dataset to {out}/")
    print()
    print("Try it:")
    print(f"  python -m src.train --data-dir {out} --out {out}_artifacts")
    print(
        f"  python -m src.predict --data-dir {out} --model-dir {out}_artifacts "
        f"--hotel-id hotel_X --room-type-code rt_x1 --stay-date 2025-08-15"
    )
    print(
        f"  python -m src.predict --data-dir {out} --model-dir {out}_artifacts "
        f"--hotel-id hotel_Z --room-type-code rt_z --stay-date 2025-08-15   "
        f"# hotel_Z doesn't exist anywhere in this dataset — true cold start"
    )
    print()
    print("Or serve it:")
    print(
        f"  BOOKING_CURVE_DATA_DIR=$(pwd)/{out} "
        f"BOOKING_CURVE_MODEL_BASE_DIR=$(pwd)/{out}_artifacts "
        f"uvicorn src.api:app --reload"
    )


if __name__ == "__main__":
    main()
