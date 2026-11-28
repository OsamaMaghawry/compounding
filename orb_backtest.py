#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) backtest on saved EODHD 1-minute data (repo market-data). No EODHD calls,
except the news catalyst check, whose results are saved to market-data/eodhd_cache/news/ and reused.

Rules (all settings below):
1. Opening range = high / low of the first OR_MINUTES minutes (09:30-09:45).
2. Stocks in play: relative volume of those first minutes >= MIN_RVOL (vs the same window on the previous
   RVOL_LOOKBACK_DAYS days), price >= MIN_PRICE, 20-day average daily volume >= MIN_AVG_DAILY_VOLUME.
3. Catalyst: company news published between the previous trading day 16:00 ET and NEWS_UNTIL (if NEWS_REQUIRED).
4. Entry: the first 1-minute close beyond the range (above the high = long, below the low = short), on a bar
   whose volume is >= BREAKOUT_VOL_MULT x the average 1-minute volume of the range, before LAST_ENTRY.
5. Stop: the middle of the range. R = range size (high - low).
6. Price reaches entry + BE_AT_R x R -> stop to breakeven.
7. 50% sold at entry + 1.5 R, 25% at entry + 2 R, last 25% runner: trailing stop TRAIL_R x R below the highest
   high since entry, starting once the 2 R target is hit. Everything still open is closed at EXIT_TIME.
Fills: entry at the breakout bar's close; targets at the target price (or the bar's open if it opens beyond);
stops at the stop price (or the open if it gaps through). If a bar touches both the stop and a target, the stop
is assumed first. SLIPPAGE_PCT on every fill.
"""

# ============================================================================
# SETTINGS
# ============================================================================
DATA_DIR = "/home/user/market-data/eodhd_cache"   # saved EODHD data (repo market-data)
TEST_START = "2025-11-17"           # first test day (needs ~10 days of history before it)
TEST_END = "2026-11-27"

OR_MINUTES = 15                     # opening range length
MIN_RVOL = 2.0                      # relative volume of the opening range
RVOL_LOOKBACK_DAYS = 20
MIN_RVOL_HISTORY_DAYS = 10
MIN_PRICE = 5.0
MIN_AVG_DAILY_VOLUME = 500_000
AVG_VOLUME_DAYS = 20
MIN_RANGE_PCT = 0.0                 # e.g. 0.5 = skip ranges narrower than 0.5% of the price
MAX_RANGE_PCT = 0.0                 # e.g. 8 = skip ranges wider than 8% of the price; 0 = off

BREAKOUT_VOL_MULT = 1.0             # breakout bar volume >= this x the average 1-min volume of the range; 0 = off
LAST_ENTRY = "11:30"                # no new entries after this time (ET)
MAX_CHASE_R = 0.0                   # e.g. 0.5 = skip if the breakout close is more than 0.5 R beyond the range; 0 = off
GAP_ALIGNED = False                 # True = longs only on gap-up days, shorts only on gap-down days

STOP_AT = 0.5                       # stop position inside the range: 0.5 = middle (0 = other side of the range)
BE_AT_R = 1.0                       # move the stop to breakeven at entry + 1 R
TARGETS = [(1.5, 0.50), (2.0, 0.25)]   # (R multiple, fraction of the position)
TRAIL_R = 1.0                       # runner: trailing stop this many R below the highest high since entry
TRAIL_START_AFTER = 2               # the trail starts after this many targets are hit
EXIT_TIME = "15:55"

NEWS_REQUIRED = True
NEWS_UNTIL = "09:45"                # news must be published before this time on the trade day
NEWS_CALL_CAP = 20_000              # EODHD calls allowed for news in one run

CAPITAL = 10_000                    # allocated capital ($)
SLIPPAGE_PCT = 0.02                 # % per fill
OUT_DIR = "results/orb"

# ============================================================================
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TZ = "America/New_York"


def hm(s):
    h, m = map(int, s.split(":"))
    return h * 60 + m - 570          # minutes after 09:30


def trading_days():
    f = next(Path(DATA_DIR, "json").glob("SPY_*.json.gz"))
    return sorted(r["date"] for r in json.load(gzip.open(f)))


def daily_info(ticker, days):
    """prev_close and average daily volume (previous AVG_VOLUME_DAYS days) per trading day."""
    f = next(Path(DATA_DIR, "json").glob(f"eod_{ticker}_*.json.gz"), None)
    if not f:
        return {}
    rows = {r["date"]: r for r in json.load(gzip.open(f))}
    out, vols, prev = {}, [], None
    for d in days:
        if prev is not None and len(vols) >= MIN_RVOL_HISTORY_DAYS:
            out[d] = (prev, float(np.mean(vols[-AVG_VOLUME_DAYS:])))
        r = rows.get(d)
        if r and r.get("close"):
            prev = float(r["close"])
            vols.append(float(r.get("volume") or 0))
    return out


def load_minutes(ticker):
    frames = [pd.read_csv(p) for p in sorted(Path(DATA_DIR, "minutes").glob(f"*/{ticker}.csv.gz"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
    df["day"] = ts.dt.strftime("%Y-%m-%d")
    df["m"] = ts.dt.hour * 60 + ts.dt.minute - 570
    return df


def simulate(side, entry_close, orh, orl, bars):
    """One trade from the bar after the breakout. bars: array rows (m, o, h, l, c). Returns dict or None."""
    s = 1 if side == "long" else -1
    slip = SLIPPAGE_PCT / 100
    R = orh - orl
    entry = entry_close * (1 + s * slip)
    stop = orl + R * STOP_AT if s == 1 else orh - R * STOP_AT
    risk = (entry - stop) * s
    if risk <= 0:
        return None
    targets = [(entry + s * k * R, frac) for k, frac in TARGETS]
    remaining, pnl, hit, be, best = 1.0, 0.0, 0, False, entry
    exit_m = None
    exit_at = hm(EXIT_TIME)
    for m, o, h, l, c in bars:
        if m >= exit_at:
            pnl += remaining * (o * (1 - s * slip) - entry) * s
            remaining, exit_m = 0.0, m
            break
        adverse = l if s == 1 else h
        if (adverse - stop) * s <= 0:                      # stop touched
            fill = o if (o - stop) * s <= 0 else stop
            pnl += remaining * (fill * (1 - s * slip) - entry) * s
            remaining, exit_m = 0.0, m
            break
        fav = h if s == 1 else l
        best = max(best, fav) if s == 1 else min(best, fav)
        for i in range(hit, len(targets)):
            price, frac = targets[i]
            if (fav - price) * s >= 0:
                fill = o if (o - price) * s >= 0 else price
                pnl += frac * (fill * (1 - s * slip) - entry) * s
                remaining -= frac
                hit = i + 1
            else:
                break
        if not be and (fav - (entry + s * BE_AT_R * R)) * s >= 0:
            stop = max(stop, entry) if s == 1 else min(stop, entry)
            be = True
        if hit >= TRAIL_START_AFTER and remaining > 1e-9:
            trail = best - s * TRAIL_R * R
            stop = max(stop, trail) if s == 1 else min(stop, trail)
        if remaining <= 1e-9:
            exit_m = m
            break
    if remaining > 1e-9:                                   # data ended before the exit time
        m, o, h, l, c = bars[-1] if len(bars) else (exit_at, entry, entry, entry, entry)
        pnl += remaining * (c * (1 - s * slip) - entry) * s
        exit_m = m
    return {"entry": entry, "risk": risk, "pnl_ps": pnl, "r": pnl / risk, "exit_m": int(exit_m),
            "mfe_r": (best - entry) * s / risk, "targets_hit": hit, "be": be}


def scan_ticker(ticker):
    """All opening-range breakout trades of one ticker (first break either side, and first long / short break)."""
    try:
        df = load_minutes(ticker)
        if df is None:
            return []
        days = trading_days()
        info = daily_info(ticker, days)
        out, or_vol_hist = [], {}
        last_entry, ormin = hm(LAST_ENTRY), OR_MINUTES
        for day, g in df.groupby("day", sort=True):
            a = g[["m", "open", "high", "low", "close", "volume"]].to_numpy(float)
            rng = a[a[:, 0] < ormin]
            if len(rng) < ormin * 0.6 or rng[0, 0] > 2:
                continue
            vol15 = rng[:, 5].sum()
            prior = [or_vol_hist[d] for d in days[max(0, days.index(day) - RVOL_LOOKBACK_DAYS):days.index(day)]
                     if d in or_vol_hist]
            or_vol_hist[day] = vol15
            if not (TEST_START <= day <= TEST_END) or day not in info or len(prior) < MIN_RVOL_HISTORY_DAYS:
                continue
            avg_prior = np.mean(prior)
            rvol = vol15 / avg_prior if avg_prior > 0 else 0
            prev_close, adv = info[day]
            orh, orl, price = rng[:, 2].max(), rng[:, 3].min(), rng[-1, 4]
            if rvol < MIN_RVOL or price < MIN_PRICE or adv < MIN_AVG_DAILY_VOLUME or orh <= orl:
                continue
            range_pct = (orh - orl) / price * 100
            if (MIN_RANGE_PCT and range_pct < MIN_RANGE_PCT) or (MAX_RANGE_PCT and range_pct > MAX_RANGE_PCT):
                continue
            gap = (rng[0, 1] / prev_close - 1) * 100
            avg_bar_vol = vol15 / len(rng)
            after = a[a[:, 0] >= ormin]
            found = {}
            for i, (m, o, h, l, c, v) in enumerate(after):
                if m > last_entry or len(found) == 2:
                    break
                if BREAKOUT_VOL_MULT and v < BREAKOUT_VOL_MULT * avg_bar_vol:
                    continue
                for side, broke, beyond in (("long", c > orh, c - orh), ("short", c < orl, orl - c)):
                    if side in found or not broke:
                        continue
                    if MAX_CHASE_R and beyond > MAX_CHASE_R * (orh - orl):
                        found[side] = None
                        continue
                    if GAP_ALIGNED and ((side == "long" and gap <= 0) or (side == "short" and gap >= 0)):
                        found[side] = None
                        continue
                    t = simulate(side, c, orh, orl, after[i + 1:, :5])
                    found[side] = t and dict(t, ticker=ticker, day=day, side=side, entry_m=int(m), rvol=round(rvol, 2),
                                             gap=round(gap, 2), range_pct=round(range_pct, 2),
                                             chase_r=round(beyond / (orh - orl), 3), price=round(price, 2))
            first = min((t for t in found.values() if t), key=lambda t: t["entry_m"], default=None)
            for side, t in found.items():
                if t:
                    t["first_break"] = t is first
                    out.append(t)
        return out
    except Exception as e:
        print(f"   ! {ticker}: {e}", flush=True)
        return []


# ---------------------------------------------------------------------------- news
_news_calls = 0


def news_times(ticker, key):
    """Publication times (UTC ISO) of all EODHD news for a ticker over the test range, saved in market-data."""
    global _news_calls
    f = Path(DATA_DIR, "news", f"{ticker}.json.gz")
    if f.exists():
        return json.load(gzip.open(f))
    times, offset = [], 0
    while True:
        if _news_calls + 5 > NEWS_CALL_CAP:
            return None
        _news_calls += 5
        for attempt in range(5):
            try:
                r = requests.get("https://eodhd.com/api/news", timeout=60, params={
                    "s": f"{ticker}.US", "from": "2025-10-01", "to": TEST_END, "limit": 1000, "offset": offset,
                    "api_token": key, "fmt": "json"})
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1)); continue
                rows = r.json() if r.status_code == 200 else []
                break
            except Exception:
                time.sleep(3); rows = []
        rows = rows if isinstance(rows, list) else []
        times += [x["date"] for x in rows if x.get("date")]
        if len(rows) < 1000:
            break
        offset += 1000
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".part")                # written completely first, then renamed
    with gzip.open(tmp, "wt") as fh:
        json.dump(sorted(set(times)), fh)
    os.replace(tmp, f)
    return times


def add_news_flags(trades, days):
    key = os.environ.get("EODHD_API_KEY", "")
    tickers = sorted({t["ticker"] for t in trades})
    with ThreadPoolExecutor(max_workers=6) as pool:
        news = dict(zip(tickers, pool.map(lambda t: news_times(t, key), tickers)))
    missing = [t for t, v in news.items() if v is None]
    print(f"📰 news: {len(tickers):,} tickers, {_news_calls:,} EODHD calls this run, {len(missing)} not fetched (cap)")
    idx = {d: i for i, d in enumerate(days)}
    until_h, until_m = map(int, NEWS_UNTIL.split(":"))
    parsed = {t: pd.to_datetime(pd.Series(v or []), utc=True) for t, v in news.items()}
    for t in trades:
        prev_day = days[idx[t["day"]] - 1]
        start = pd.Timestamp(f"{prev_day} 16:00", tz=TZ)
        end = pd.Timestamp(f"{t['day']} {until_h:02d}:{until_m:02d}", tz=TZ)
        s = parsed[t["ticker"]]
        t["news"] = None if news[t["ticker"]] is None else bool(((s >= start) & (s < end)).any())
    return trades


# ---------------------------------------------------------------------------- portfolio
def portfolio(trades, slots, long_only, news_required, sizing="EQUAL", risk_pct=0.5, daily_limit_pct=0.0,
              keep=None):
    """Applies an optional filter (keep), slots (max open positions), sizing and an optional daily loss limit.
    Long+short takes, per stock and day, the first break that passes the filters; long only takes the long break."""
    pool = [t for t in trades if (not news_required or t["news"]) and (keep is None or keep(t))]
    if long_only:
        pool = [t for t in pool if t["side"] == "long"]
    else:
        first = {}
        for t in pool:
            k = (t["ticker"], t["day"])
            if k not in first or t["entry_m"] < first[k]["entry_m"]:
                first[k] = t
        pool = list(first.values())
    by_day = defaultdict(list)
    for t in pool:
        by_day[t["day"]].append(t)
    taken = []
    slot_value = CAPITAL / slots
    for day in sorted(by_day):
        open_exits, closed = [], []          # closed: (exit minute, pnl) of trades taken today
        for t in sorted(by_day[day], key=lambda x: (x["entry_m"], -x["rvol"])):
            open_exits = [e for e in open_exits if e > t["entry_m"]]
            if len(open_exits) >= slots:
                continue
            # daily loss limit: only trades already CLOSED before this entry count (no look-ahead)
            realized = sum(p for e, p in closed if e <= t["entry_m"])
            if daily_limit_pct and realized <= -CAPITAL * daily_limit_pct / 100:
                continue
            if sizing == "RISK":
                shares = min(CAPITAL * risk_pct / 100 / t["risk"], 2 * slot_value / t["entry"])
            else:
                shares = slot_value / t["entry"]
            if t["side"] == "short":
                shares = np.floor(shares)
            if shares <= 0:
                continue
            pnl = shares * t["pnl_ps"]
            taken.append(dict(t, shares=shares, pnl=pnl))
            open_exits.append(t["exit_m"])
            closed.append((t["exit_m"], pnl))
    return taken


def stats(taken, days_in_test):
    if not taken:
        return {"trades": 0}
    df = pd.DataFrame(taken)
    wins, losses = df[df.pnl > 0].pnl.sum(), -df[df.pnl < 0].pnl.sum()
    daily = df.groupby("day").pnl.sum().reindex(days_in_test, fill_value=0.0)
    eq = daily.cumsum()
    best5 = df.pnl.nlargest(5).sum()
    return {
        "trades": len(df), "win_pct": round((df.pnl > 0).mean() * 100, 1),
        "pf": round(wins / losses, 2) if losses else None, "total": round(df.pnl.sum(), 0),
        "avg_r": round(df.r.mean(), 3), "without_best5": round(df.pnl.sum() - best5, 0),
        "long_pnl": round(df[df.side == "long"].pnl.sum(), 0), "short_pnl": round(df[df.side == "short"].pnl.sum(), 0),
        "worst_day": round(daily.min(), 0), "max_dd": round((eq - eq.cummax().clip(lower=0)).min(), 0),
        "days_up": int((daily > 0).sum()), "days_down": int((daily < 0).sum()),
    }


def main():
    days = trading_days()
    test_days = [d for d in days if TEST_START <= d <= TEST_END]
    tickers = sorted({p.name[:-7] for p in Path(DATA_DIR, "minutes").glob("*/*.csv.gz")} - {"SPY"})
    print(f"ORB {OR_MINUTES}m | {len(test_days)} test days {test_days[0]} -> {test_days[-1]} | {len(tickers):,} stocks", flush=True)
    t0 = time.time()
    trades = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for i, res in enumerate(pool.map(scan_ticker, tickers, chunksize=8), 1):
            trades += res
            if i % 500 == 0:
                print(f"   scanned {i:,}/{len(tickers):,} ({time.time() - t0:.0f}s, {len(trades):,} breakouts)", flush=True)
    print(f"✅ {len(trades):,} breakout trades found ({time.time() - t0:.0f}s)", flush=True)
    trades = add_news_flags(trades, days)
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out / "all_breakouts.csv.gz", index=False)
    return trades, test_days


if __name__ == "__main__":
    main()
