#!/usr/bin/env python
"""
A real, lightweight load test against a running instance of the API —
concurrent requests, measured p50/p95/p99 latency and throughput. Written
because "no load testing" was a flagged gap; this exists so that claim can
be replaced with actual numbers instead of a script nobody has run.

Usage (against a server already running, e.g. `uvicorn src.api:app`):
    python -m scripts.load_test [--url http://localhost:8000] [--requests 200] [--concurrency 20]

Needs a live server pointed at a real model (see README "How to run" /
scripts/generate_demo_data.py) — this hits the network, it doesn't spin
up its own server.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx


async def _one_request(client: httpx.AsyncClient, url: str, params: dict) -> tuple[int, float]:
    start = time.monotonic()
    try:
        r = await client.get(url, params=params)
        return r.status_code, time.monotonic() - start
    except httpx.RequestError:
        return -1, time.monotonic() - start


async def run(base_url: str, n_requests: int, concurrency: int, hotel_id: str, room_type_code: str):
    url = f"{base_url}/v1/booking-curve"
    params = {"hotel_id": hotel_id, "room_type_code": room_type_code, "stay_date": "2025-08-15"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # confirm the server is actually up and ready before hammering it —
        # a load test against a 503 is a load test of nothing
        ready = await client.get(f"{base_url}/readyz")
        if ready.status_code != 200:
            print(f"Server not ready ({ready.status_code}): {ready.text}")
            print("Point --url at a running, ready instance first.")
            return

        sem = asyncio.Semaphore(concurrency)

        async def bounded():
            async with sem:
                return await _one_request(client, url, params)

        start = time.monotonic()
        results = await asyncio.gather(*(bounded() for _ in range(n_requests)))
        wall_time = time.monotonic() - start

    statuses = [s for s, _ in results]
    latencies = sorted(t for _, t in results)
    ok = sum(1 for s in statuses if s == 200)
    rate_limited = sum(1 for s in statuses if s == 429)
    errors = sum(1 for s in statuses if s not in (200, 429))

    def pct(p):
        idx = min(int(len(latencies) * p), len(latencies) - 1)
        return latencies[idx] * 1000

    print(f"\n{n_requests} requests, concurrency={concurrency}, wall time={wall_time:.2f}s")
    print(f"  throughput:      {n_requests / wall_time:.1f} req/s")
    print(f"  200 OK:          {ok}  ({ok / n_requests:.1%})")
    print(f"  429 rate-limited: {rate_limited}  ({rate_limited / n_requests:.1%})")
    print(f"  other/errors:    {errors}")
    print(f"  latency p50:     {pct(0.50):.1f} ms")
    print(f"  latency p95:     {pct(0.95):.1f} ms")
    print(f"  latency p99:     {pct(0.99):.1f} ms")
    print(f"  latency max:     {latencies[-1] * 1000:.1f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--hotel-id", default="hotel_X")
    ap.add_argument("--room-type-code", default="rt_x1")
    args = ap.parse_args()
    asyncio.run(run(args.url, args.requests, args.concurrency, args.hotel_id, args.room_type_code))


if __name__ == "__main__":
    main()
