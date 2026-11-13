"""Search load test: concurrent authenticated /search requests with p95 reporting.

No new dependency; plain asyncio + httpx (plan §17.1 "Load (manual)" row).
Runs against a live stack (`docker compose up` + seed + worker) and reports
throughput and latency percentiles, so thresholds can be checked against the
runbook (search p95 < 1 s alert-worthy; < 300 ms local target).

Usage:
  uv run python scripts/loadtest.py --tenant acme --email admin@acme.example \
      --password 'correct horse battery staple' --concurrency 10 --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def _worker(
    client: httpx.AsyncClient,
    token: str,
    query: str,
    stop_at: float,
    latencies: list[float],
    statuses: dict[int, int],
) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    while time.monotonic() < stop_at:
        started = time.monotonic()
        try:
            response = await client.post("/api/v1/search", json={"query": query}, headers=headers)
        except httpx.HTTPError:
            statuses[-1] = statuses.get(-1, 0) + 1
            continue
        latencies.append(time.monotonic() - started)
        statuses[response.status_code] = statuses.get(response.status_code, 0) + 1


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


async def run(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "tenant_slug": args.tenant,
                "email": args.email,
                "password": args.password,
            },
        )
        if response.status_code != 200:
            raise SystemExit(f"login failed: {response.status_code} {response.text}")
        token: str = response.json()["access_token"]

        latencies: list[float] = []
        statuses: dict[int, int] = {}
        stop_at = time.monotonic() + args.duration
        started = time.monotonic()
        async with asyncio.TaskGroup() as group:
            for _ in range(args.concurrency):
                group.create_task(_worker(client, token, args.query, stop_at, latencies, statuses))

    total = sum(statuses.values())
    elapsed = time.monotonic() - started
    ordered = sorted(latencies)
    p50 = _percentile(ordered, 0.50) * 1000
    p95 = _percentile(ordered, 0.95) * 1000
    p99 = _percentile(ordered, 0.99) * 1000
    p100 = (ordered[-1] if ordered else 0.0) * 1000
    print(f"requests: {total} over {elapsed:.1f}s -> {total / elapsed:.1f} req/s")
    print(f"latency ms: p50={p50:.0f} p95={p95:.0f} p99={p99:.0f} max={p100:.0f}")
    if ordered:
        print(f"mean ms: {statistics.mean(ordered) * 1000:.0f}")
    for status, count in sorted(statuses.items()):
        print(f"status {status}: {count}")
    if statuses.get(429):
        print("note: 429s present; RL_SEARCH_PER_MINUTE was hit (load test exceeded the limit)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--tenant", default="acme")
    parser.add_argument("--email", default="admin@acme.example")
    parser.add_argument("--password", default="correct horse battery staple")
    parser.add_argument("--query", default="how does ingestion work")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--duration", type=int, default=30, help="seconds")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
