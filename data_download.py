#!/usr/bin/env python3
"""
Downloads EODHD data for the RVOL intraday backtest once, into the backtest's own cache format
(eodhd_cache/), so every later test runs on it without new EODHD calls.

1. Daily bars: one request per US common stock for the whole range (1 call each), turned into the
   per-day files the backtest reads (json/bulk_US_<day>.json.gz).
2. 1-minute bars, regular session only: per stock, per 115-day window, only for stocks that pass the
   backtest's price / average-volume prefilter on at least one day of that window (5 calls each).

Settings below. Already-downloaded files are skipped, so it can be stopped and restarted.
"""
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import rvol_ema_backtest as bt

START = "2025-10-01"            # daily data from here (history for the first window's prefilter)
WINDOWS = [                     # 1-minute data windows (<= 115 calendar days each)
    ("2025-11-03", "2026-02-25"),
    ("2026-02-26", "2026-06-20"),
    ("2026-06-21", "2026-10-13"),
    ("2026-10-14", "2026-11-27"),
]
END = WINDOWS[-1][1]
CALL_CAP = 75_000               # hard stop for this run

bt.MAX_API_CALLS = CALL_CAP


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    days = bt.get_trading_days(START, END)
    universe = sorted(bt.get_universe())
    log(f"{len(days)} trading days {days[0]} -> {days[-1]}, {len(universe):,} US common stocks")

    # 1. Daily bars per stock
    def daily(t):
        try:
            return t, bt.cached_json(f"eod_{t.replace('/', '_')}_{START}_{END}", f"eod/{t}.US",
                                     {"from": START, "to": END}, "eod")
        except bt.CallLimitReached:
            raise
        except Exception as e:
            return t, []
    rows_by_day = {d: [] for d in days}
    with ThreadPoolExecutor(max_workers=bt.DOWNLOAD_THREADS) as pool:
        futs = [pool.submit(daily, t) for t in universe]
        for i, f in enumerate(as_completed(futs), 1):
            t, data = f.result()
            for r in data if isinstance(data, list) else []:
                if r.get("date") in rows_by_day and r.get("close") is not None:
                    rows_by_day[r["date"]].append({"code": t, "close": r["close"], "volume": r.get("volume") or 0})
            if i % 1000 == 0:
                log(f"daily {i:,}/{len(universe):,} (calls used {bt._calls_used:,})")
    for d, rows in rows_by_day.items():
        with gzip.open(bt.cache_path("json", f"bulk_US_{d}.json.gz"), "wt") as f:
            json.dump(rows, f)
    log(f"daily done: per-day files written (calls used {bt._calls_used:,})")

    # 2. Candidates per window: price/volume prefilter on any trading day of the window
    daily_df = pd.DataFrame([dict(r, day=d) for d, rows in rows_by_day.items() for r in rows])
    daily_df = daily_df.sort_values(["code", "day"])
    g = daily_df.groupby("code")
    daily_df["avg_vol"] = g["volume"].transform(lambda s: s.shift(1).rolling(bt.AVG_VOLUME_DAYS, min_periods=bt.MIN_RVOL_HISTORY_DAYS).mean())
    daily_df["prev_close"] = g["close"].shift(1)
    ok = daily_df[(daily_df["avg_vol"] >= bt.MIN_AVG_DAILY_VOLUME) &
                  (daily_df["prev_close"] >= bt.MIN_PRICE * bt.PREFILTER_PRICE_FACTOR)]
    jobs = []
    for s, e in WINDOWS:
        syms = sorted(set(ok[(ok["day"] >= s) & (ok["day"] <= e)]["code"]) | {"SPY"})
        jobs += [(t, s, e) for t in syms]
        log(f"window {s} -> {e}: {len(syms):,} stocks")
    todo = [j for j in jobs if not bt.minutes_file(*j).exists()]
    log(f"1-minute: {len(todo):,} downloads to do = {len(todo) * 5:,} calls (cap {CALL_CAP:,}, used {bt._calls_used:,})")

    # 3. 1-minute bars
    done = failed = 0
    with ThreadPoolExecutor(max_workers=bt.DOWNLOAD_THREADS) as pool:
        futs = {pool.submit(bt.download_minutes, *j): j for j in todo}
        for f in as_completed(futs):
            try:
                f.result()
                done += 1
            except bt.CallLimitReached as e:
                log(f"STOPPED: {e}")
                pool.shutdown(cancel_futures=True)
                break
            except Exception as e:
                failed += 1
            if (done + failed) % 500 == 0:
                log(f"1-minute {done + failed:,}/{len(todo):,} (failed {failed}, calls used {bt._calls_used:,})")
    log(f"DONE: {done:,} downloaded, {failed} failed, EODHD calls used this run: {bt._calls_used:,}")


if __name__ == "__main__":
    sys.exit(main())
