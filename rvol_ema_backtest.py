#!/usr/bin/env python3
"""
RVOL + EMA Intraday Backtest (EODHD data)

How a test day is replayed:
1. SCANS (only at the scan times switched on below): find "stocks in play" by relative
   volume - volume from the open to the scan time, compared with the average volume over
   the same window on the previous RVOL_LOOKBACK_DAYS trading days - plus price and
   average daily volume filters. Every stock found is added to the day's watchlist and
   stays there for the rest of the day.
2. MONITORING (every minute, from the first scan until EXIT_ALL_TIME): checks exits on
   open positions and entries on every watchlist stock, on 1-minute bars.
   - LONG entry : close above EMA_SLOW (200) and the mid line - EMA_MID (21) or VWAP (MID_LINE) -
                  and EMA_FAST (9) crosses back above the mid line after dipping below it.
   - LONG exit  : first half when the close crosses below EMA_FAST (9), the rest when it
                  crosses below the mid line; STOP_LOSS_PCT stop; everything closed at
                  EXIT_ALL_TIME.
   - SHORT      : the mirror image (switch with ENABLE_SHORTS).
3. CAPITAL: ACCOUNT_EQUITY x ALLOCATION_PCT is split into NUM_SLOTS equal positions.

Install:  pip install pandas numpy requests matplotlib
Run:      python rvol_ema_backtest.py
"""

# ============================================================================
# SETTINGS - EVERYTHING YOU CAN CHANGE IS HERE
# ============================================================================

# ---------- TEST DAYS ----------
TEST_START_DATE = "2026-05-01"      # first date of the range (YYYY-MM-DD)
TEST_END_DATE = "2026-10-30"        # last date of the range (YYYY-MM-DD)
RANDOM_DAYS = 2                     # 0 = test every trading day in the range; N = pick N random trading days
RANDOM_SEED = None                  # None = different random days each run; a number (e.g. 7) = same days every run

# ---------- SCAN TIMES (Eastern Time) - True = ON, False = OFF ----------
SCAN_TIMES = {
    "09:45": True,
    "11:00": False,
    "13:00": False,
    "15:00": False,
}

# ---------- STOCKS IN PLAY FILTERS ----------
MIN_RELATIVE_VOLUME = 2.0           # volume so far today vs the average for the same window
RVOL_LOOKBACK_DAYS = 20             # previous trading days used for the average
MIN_RVOL_HISTORY_DAYS = 10          # a stock needs at least this many of those days to be scanned
MIN_PRICE = 5.0                     # $ price at scan time
MIN_AVG_DAILY_VOLUME = 500_000      # average daily shares over AVG_VOLUME_DAYS
AVG_VOLUME_DAYS = 20

# ---------- STRATEGY ----------
EMA_FAST = 9
EMA_MID = 21
MID_LINE = "EMA"                    # "EMA" = the EMA_MID (21) line; "VWAP" = the day's VWAP replaces it
                                    # (entry: EMA_FAST crosses back over the line; second exit: close crosses the line)
EMA_SLOW = 200
USE_SLOW_EMA_FILTER = False         # True = longs only above EMA_SLOW (200), shorts only below it; False = ignored
ENABLE_LONGS = True
ENABLE_SHORTS = True
FIRST_EXIT_FRACTION = 0.5           # part sold at the EMA_FAST cross (0.5 = half)
STOP_LOSS_PCT = 5.0                 # % against the entry price; 0 = no stop
LAST_ENTRY_TIME = "15:30"           # no new entries after this time
EXIT_ALL_TIME = "15:45"             # every open position is closed at this time
ALLOW_REENTRY = True                # a stock can be traded again the same day after it is closed
EMA_WARMUP_DAYS = 2                 # previous trading days loaded so the 200 EMA is ready at the open
                                    # (raised automatically for bigger bars, e.g. 5-minute bars need 4)
EMA_INCLUDE_EXTENDED_HOURS = False  # True = pre/after-market bars count in the EMAs

# ---------- FILTERS AGAINST WHIPSAWS (0 = off) ----------
BAR_MINUTES = 1                     # chart timeframe: 1, 2, 5 ... minute bars (EMAs and signals use these bars)
MAX_TRADES_PER_STOCK_PER_DAY = 0    # e.g. 2 = a stock can be entered at most twice a day; 0 = no limit
MIN_TREND_MINUTES = 0               # e.g. 15 = before the dip, EMA_FAST must have been on the trade's side of
                                    # EMA_MID for at least 15 minutes (skips sideways stocks)
EXIT_BUFFER_PCT = 0.0               # e.g. 0.2 = the EMA_MID exit needs a close 0.2% beyond EMA_MID, not just a touch
MAX_WATCHLIST_PER_SCAN = 0          # e.g. 20 = only the 20 highest relative volume stocks of each scan; 0 = all

# ---------- CAPITAL ----------
ACCOUNT_EQUITY = 100_000            # account equity ($)
ALLOCATION_PCT = 10.0               # % of equity this strategy trades (10 -> $10,000)
NUM_SLOTS = 10                      # max positions at once; each gets allocated capital / NUM_SLOTS
COMPOUND_DAILY = False              # True = each test day starts with the previous day's ending capital
ALLOW_FRACTIONAL_SHARES = False     # longs only; shorts always use whole shares

# ---------- FILLS & COSTS ----------
SLIPPAGE_PCT = 0.02                 # % worse than the signal bar's close, on every fill
COMMISSION_PER_SHARE = 0.0          # $ per share, each side

# ---------- EODHD DATA ----------
EODHD_API_KEY = "PUT_YOUR_EODHD_API_KEY_HERE"   # or set the EODHD_API_KEY environment variable
EXCLUDED_EXCHANGES = ["OTC", "PINK", "NMFQS"]   # venues whose name contains any of these are left out (OTC, pink sheets, mutual funds)
INCLUDE_ETFS = False                # False = stocks only; True = ETFs can be stocks in play too
INCLUDE_DELISTED = False            # False = only stocks listed today; True = also stocks delisted since then
PREFILTER_PRICE_FACTOR = 0.7        # 1-minute data is only downloaded for stocks whose previous close was
                                    # at least MIN_PRICE x this (the exact price check happens at scan time)
CACHE_DIR = "eodhd_cache"           # downloaded data is kept here, so re-running the same days costs nothing
DOWNLOAD_THREADS = 8                # parallel downloads (EODHD allows about 1,000 requests a minute)
MAX_API_CALLS = 60_000              # stop before a run would use more EODHD calls than this (your plan: 100,000/day)
ESTIMATE_CALLS_ONLY = False         # True = only print how many EODHD calls the run needs, download nothing

# ---------- OUTPUT ----------
RESULTS_DIR = "backtest_results"

# ============================================================================
# END OF SETTINGS
# ============================================================================


import gzip
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TZ = "America/New_York"
SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
EODHD_URL = "https://eodhd.com/api"
CALL_COST = {"eod": 1, "list": 1, "bulk": 100, "intraday": 5}   # EODHD API calls per request
MAX_DAYS_PER_REQUEST = 115           # EODHD gives at most 120 days of 1-minute bars per request

_calls_used = 0
_calls_lock = threading.Lock()


class CallLimitReached(Exception):
    pass


# ============================================================================
# EODHD DATA (with local cache)
# ============================================================================

def api_key():
    key = os.environ.get("EODHD_API_KEY") or EODHD_API_KEY
    if not key or key.startswith("PUT_YOUR"):
        raise SystemExit("Set EODHD_API_KEY at the top of the file (or as an environment variable).")
    return key


def eodhd_get(path, params, kind):
    """One EODHD request, counted against MAX_API_CALLS, retried when rate-limited."""
    global _calls_used
    with _calls_lock:
        if _calls_used + CALL_COST[kind] > MAX_API_CALLS:
            raise CallLimitReached(f"This run reached MAX_API_CALLS ({MAX_API_CALLS:,} EODHD calls).")
        _calls_used += CALL_COST[kind]
    for attempt in range(6):
        try:
            r = requests.get(f"{EODHD_URL}/{path}", params={**params, "api_token": api_key(), "fmt": "json"}, timeout=90)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 404:
            return []
        if r.status_code == 402:
            raise CallLimitReached("EODHD says today's API limit is used up.")
        if r.status_code in (401, 403):
            raise SystemExit(f"EODHD refused the request ({r.status_code}): check the API key and that your plan includes {path.split('/')[0]}.")
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"EODHD request kept failing: {path}")


def cache_path(*parts):
    p = Path(CACHE_DIR).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cached_json(name, path, params, kind):
    file = cache_path("json", f"{name}.json.gz")
    if file.exists():
        with gzip.open(file, "rt") as f:
            return json.load(f)
    data = eodhd_get(path, params, kind)
    with gzip.open(file, "wt") as f:
        json.dump(data, f)
    return data


def get_trading_days(start_date, end_date):
    """Trading days in a range, taken from SPY daily bars."""
    data = cached_json(f"SPY_{start_date}_{end_date}", "eod/SPY.US", {"from": start_date, "to": end_date}, "eod")
    return sorted(row["date"] for row in data)


def get_universe():
    """US stock tickers, without OTC/pink-sheet venues (plus delisted ones if switched on)."""
    month = datetime.now().strftime("%Y-%m")
    rows = cached_json(f"symbols_US_{month}", "exchange-symbol-list/US", {}, "list")
    if INCLUDE_DELISTED:
        rows = rows + cached_json(f"symbols_US_delisted_{month}", "exchange-symbol-list/US", {"delisted": 1}, "list")
    types = {"Common Stock"} | ({"ETF"} if INCLUDE_ETFS else set())
    def listed(exchange):
        exchange = str(exchange or "").upper()
        return not any(x.upper() in exchange for x in EXCLUDED_EXCHANGES)
    return {r["Code"] for r in rows if r.get("Type") in types and listed(r.get("Exchange"))}


def get_bulk_day(day):
    """Daily bars for every US ticker on one day (close and volume)."""
    rows = cached_json(f"bulk_US_{day}", "eod-bulk-last-day/US", {"date": day}, "bulk")
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["code", "close", "volume"])
    return df[["code", "close", "volume"]]


def minutes_file(ticker, start_day, end_day):
    return cache_path("minutes", f"{start_day}_{end_day}", f"{ticker.replace('/', '_')}.csv.gz")


def download_minutes(ticker, start_day, end_day):
    """1-minute bars (incl. pre/after-market) for one ticker over one date range, saved to the cache."""
    file = minutes_file(ticker, start_day, end_day)
    if file.exists():
        return
    start = int(pd.Timestamp(start_day, tz=TZ).timestamp())
    end = int((pd.Timestamp(end_day, tz=TZ) + timedelta(days=1)).timestamp())
    rows = eodhd_get(f"intraday/{ticker}.US", {"interval": "1m", "from": start, "to": end}, "intraday")
    df = pd.DataFrame(rows if isinstance(rows, list) else [])
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[cols].dropna() if not df.empty else pd.DataFrame(columns=cols)
    if not EMA_INCLUDE_EXTENDED_HOURS and not df.empty:
        # Only regular-session bars are used, so only those are kept on disk
        t = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(TZ).dt.time
        df = df[(t >= SESSION_OPEN) & (t < SESSION_CLOSE)]
    df.to_csv(file, index=False, compression="gzip")


def load_minutes(ticker, chunks, days):
    """Cached 1-minute bars of one ticker for the given trading days (Eastern Time)."""
    frames = []
    for start_day, end_day in chunks:
        file = minutes_file(ticker, start_day, end_day)
        if file.exists() and any(start_day <= d <= end_day for d in days):
            frames.append(pd.read_csv(file))
    if not frames:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(TZ)
    df = df[df["ts"].dt.strftime("%Y-%m-%d").isin(set(days))]
    df["symbol"] = ticker
    return df[["ts", "symbol", "open", "high", "low", "close", "volume"]].sort_values("ts")


def date_chunks(test_days, all_days, days_back):
    """Date ranges of 1-minute data needed: each test day plus the days before it, merged, <= 115 days each."""
    ranges = []
    for day in test_days:
        idx = all_days.index(day)
        ranges.append([all_days[max(0, idx - days_back)], day])
    ranges.sort()
    merged = []
    for start, end in ranges:
        if merged and pd.Timestamp(start) <= pd.Timestamp(merged[-1][1]) + timedelta(days=7):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    chunks = []
    for start, end in merged:
        s = pd.Timestamp(start)
        while s <= pd.Timestamp(end):
            e = min(s + timedelta(days=MAX_DAYS_PER_REQUEST - 1), pd.Timestamp(end))
            chunks.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
            s = e + timedelta(days=1)
    return chunks


# ============================================================================
# SCANNING - STOCKS IN PLAY
# ============================================================================

def mid_label():
    return "VWAP" if MID_LINE.upper() == "VWAP" else f"EMA {EMA_MID}"


def warmup_days():
    """Previous trading days needed so the slow EMA is ready at the open on the chosen bar size."""
    return max(EMA_WARMUP_DAYS, int(np.ceil(EMA_SLOW * BAR_MINUTES / 390)) + 1)


def enabled_scan_times():
    return sorted(t for t, on in SCAN_TIMES.items() if on)


def daily_filters(day, all_days, universe):
    """Average daily volume and previous close before a test day, for the stocks in the universe."""
    idx = all_days.index(day)
    days = all_days[max(0, idx - AVG_VOLUME_DAYS):idx]
    frames = [get_bulk_day(d).assign(day=d) for d in days]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["code"].isin(universe)].sort_values("day")
    g = df.groupby("code")
    return pd.DataFrame({"avg_daily_volume": g["volume"].mean(), "prev_close": g["close"].last()})


def scan_windows(ticker, chunks, days):
    """For one ticker: volume from the open to each scan time, and the last price before it, per day."""
    df = load_minutes(ticker, chunks, days)
    if df.empty:
        return []
    t = df["ts"].dt.time
    df = df[(t >= SESSION_OPEN) & (t < SESSION_CLOSE)]
    df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
    rows = []
    for scan in enabled_scan_times():
        upto = df[df["ts"].dt.time < dtime.fromisoformat(scan)]
        if upto.empty:
            continue
        g = upto.groupby("day")
        for day, vol, price in zip(g.size().index, g["volume"].sum(), g["close"].last()):
            rows.append((ticker, day, scan, float(vol), float(price)))
    return rows


def scan_day(day, prior_days, windows, filters):
    """Run every enabled scan for one day. Returns the watchlist (first scan that found each stock)."""
    found = []
    for scan in enabled_scan_times():
        w = windows[windows["scan_time"] == scan]
        today = w[w["date"] == day].set_index("symbol")
        hist = w[w["date"].isin(prior_days)].groupby("symbol")["volume"]
        df = today.join(pd.DataFrame({"avg_volume": hist.mean(), "history_days": hist.count()}), how="inner")
        df = df.join(filters, how="inner")
        df = df[(df["history_days"] >= MIN_RVOL_HISTORY_DAYS) & (df["avg_volume"] > 0)]
        df["rvol"] = df["volume"] / df["avg_volume"]
        df = df[(df["rvol"] >= MIN_RELATIVE_VOLUME) & (df["price"] >= MIN_PRICE)
                & (df["avg_daily_volume"] >= MIN_AVG_DAILY_VOLUME)]
        if MAX_WATCHLIST_PER_SCAN:
            df = df.sort_values("rvol", ascending=False).head(MAX_WATCHLIST_PER_SCAN)
        print(f"   🔎 {day} {scan}: {len(df)} stocks in play")
        found.append(df.reset_index().rename(columns={"volume": "today_volume"})
                     [["symbol", "scan_time", "rvol", "price", "today_volume", "avg_volume", "avg_daily_volume"]])
    watchlist = pd.concat(found, ignore_index=True)
    return watchlist.sort_values("scan_time").drop_duplicates("symbol", keep="first").reset_index(drop=True)


# ============================================================================
# INDICATORS
# ============================================================================

def prepare_bars(minutes, day):
    """EMAs per symbol over warm-up days + the test day; returns only the test day's session bars."""
    if minutes.empty:
        return {}
    t = minutes["ts"].dt.time
    if not EMA_INCLUDE_EXTENDED_HOURS:
        minutes = minutes[(t >= SESSION_OPEN) & (t < SESSION_CLOSE)]
    out = {}
    for symbol, g in minutes.sort_values("ts").groupby("symbol"):
        g = g.drop_duplicates("ts").set_index("ts")
        if BAR_MINUTES > 1:
            # N-minute bars aligned to the 9:30 open, built day by day
            g = (g.groupby(g.index.date, group_keys=False)
                  .apply(lambda d: d.resample(f"{BAR_MINUTES}min", origin=d.index[0].normalize() + pd.Timedelta(hours=9, minutes=30))
                         .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}))
                  .dropna(subset=["close"]))
        for name, span in (("ema_fast", EMA_FAST), ("ema_mid", EMA_MID), ("ema_slow", EMA_SLOW)):
            g[name] = g["close"].ewm(span=span, adjust=False).mean()
        if MID_LINE.upper() == "VWAP":
            # Session VWAP, restarting every day at the open (typical price x volume)
            typical = (g["high"] + g["low"] + g["close"]) / 3
            day_key = g.index.date
            g["ema_mid"] = ((typical * g["volume"]).groupby(day_key).cumsum()
                            / g["volume"].groupby(day_key).cumsum().replace(0, np.nan)).ffill()
        g["prev_close"] = g["close"].shift(1)
        g["prev_fast"] = g["ema_fast"].shift(1)
        g["prev_mid"] = g["ema_mid"].shift(1)
        g["bars_seen"] = np.arange(1, len(g) + 1)
        # Length (in bars) of the EMA_FAST-vs-EMA_MID run two runs back: the trend before the dip
        above = g["ema_fast"] > g["ema_mid"]
        run_id = (above != above.shift()).cumsum()
        run_len = run_id.map(run_id.value_counts())
        lengths = run_len.groupby(run_id).first()
        g["trend_bars"] = run_id.map(lambda r: lengths.get(r - 2, 0))
        g = g[(g.index.strftime("%Y-%m-%d") == day)
              & (g.index.time >= SESSION_OPEN) & (g.index.time < SESSION_CLOSE)]
        if not g.empty:
            out[symbol] = g
    return out


def entry_signal(bar):
    """'long', 'short' or None for one closed bar."""
    warmup = EMA_SLOW if USE_SLOW_EMA_FILTER else EMA_MID
    if bar["bars_seen"] < warmup or pd.isna(bar["prev_fast"]):
        return None  # EMAs not ready yet
    crossed_up = bar["ema_fast"] > bar["ema_mid"] and bar["prev_fast"] <= bar["prev_mid"]
    crossed_down = bar["ema_fast"] < bar["ema_mid"] and bar["prev_fast"] >= bar["prev_mid"]
    if MIN_TREND_MINUTES and bar["trend_bars"] * BAR_MINUTES < MIN_TREND_MINUTES:
        return None  # no established trend before the dip
    above_slow = bar["close"] > bar["ema_slow"] or not USE_SLOW_EMA_FILTER
    below_slow = bar["close"] < bar["ema_slow"] or not USE_SLOW_EMA_FILTER
    if ENABLE_LONGS and crossed_up and bar["close"] > bar["ema_mid"] and above_slow:
        return "long"
    if ENABLE_SHORTS and crossed_down and bar["close"] < bar["ema_mid"] and below_slow:
        return "short"
    return None


# ============================================================================
# SIMULATION
# ============================================================================

def fill_price(price, side, action):
    """Apply slippage: buying fills higher, selling fills lower."""
    buying = (side == "long" and action == "entry") or (side == "short" and action == "exit")
    return price * (1 + SLIPPAGE_PCT / 100) if buying else price * (1 - SLIPPAGE_PCT / 100)


def close_part(pos, shares, price, ts, reason):
    exit_px = fill_price(price, pos["side"], "exit")
    direction = 1 if pos["side"] == "long" else -1
    pnl = (exit_px - pos["entry_price"]) * shares * direction - COMMISSION_PER_SHARE * shares
    pos["exits"].append({"time": ts + timedelta(minutes=BAR_MINUTES), "shares": shares, "price": exit_px, "reason": reason, "pnl": pnl})
    pos["shares_left"] -= shares


def finish_trade(pos, trades):
    exited = sum(e["shares"] for e in pos["exits"])
    pnl = sum(e["pnl"] for e in pos["exits"]) - COMMISSION_PER_SHARE * pos["shares"]
    avg_exit = sum(e["price"] * e["shares"] for e in pos["exits"]) / exited
    trades.append({
        "date": pos["date"], "symbol": pos["symbol"], "side": pos["side"], "scan_time": pos["scan_time"],
        "rvol": round(pos["rvol"], 2), "entry_time": pos["entry_time"].strftime("%H:%M"),
        "entry_price": round(pos["entry_price"], 4), "shares": pos["shares"],
        "position_value": round(pos["entry_price"] * pos["shares"], 2),
        "exit_1_time": pos["exits"][0]["time"].strftime("%H:%M"), "exit_1_reason": pos["exits"][0]["reason"],
        "exit_2_time": pos["exits"][-1]["time"].strftime("%H:%M"), "exit_2_reason": pos["exits"][-1]["reason"],
        "avg_exit_price": round(avg_exit, 4), "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / (pos["entry_price"] * pos["shares"]) * 100, 3),
    })


def simulate_day(day, watchlist, bars, capital):
    """Replay one day minute by minute. Returns the closed trades."""
    slot_value = capital / NUM_SLOTS
    first_scan = dtime.fromisoformat(enabled_scan_times()[0])
    last_entry = dtime.fromisoformat(LAST_ENTRY_TIME)
    exit_all = dtime.fromisoformat(EXIT_ALL_TIME)

    added_at = {r.symbol: dtime.fromisoformat(r.scan_time) for r in watchlist.itertuples()}
    rvol = dict(zip(watchlist["symbol"], watchlist["rvol"]))
    scan_of = dict(zip(watchlist["symbol"], watchlist["scan_time"]))
    by_symbol_time = {s: g for s, g in bars.items() if s in added_at}
    all_times = sorted(set().union(*[set(g.index) for g in by_symbol_time.values()])) if by_symbol_time else []

    open_positions, trades, closed_at = {}, [], {}
    entries_today = {}
    for ts in all_times:
        bar_close_time = (ts + timedelta(minutes=BAR_MINUTES)).time()   # decisions are made when the bar closes
        if bar_close_time < first_scan:
            continue

        # 1. Exits
        for symbol in list(open_positions):
            g = by_symbol_time[symbol]
            if ts not in g.index:
                continue
            bar, pos = g.loc[ts], open_positions[symbol]
            long = pos["side"] == "long"

            if STOP_LOSS_PCT > 0:
                stop = pos["entry_price"] * (1 - STOP_LOSS_PCT / 100 if long else 1 + STOP_LOSS_PCT / 100)
                hit = bar["low"] <= stop if long else bar["high"] >= stop
                if hit:
                    gap_px = min(bar["open"], stop) if long else max(bar["open"], stop)
                    close_part(pos, pos["shares_left"], gap_px, ts, "stop loss")
            if pos["shares_left"] > 0 and bar_close_time >= exit_all:
                close_part(pos, pos["shares_left"], bar["close"], ts, f"end of day {EXIT_ALL_TIME}")
            if pos["shares_left"] > 0:
                buffer = EXIT_BUFFER_PCT / 100
                below_mid = (bar["close"] < bar["ema_mid"] * (1 - buffer)) if long else (bar["close"] > bar["ema_mid"] * (1 + buffer))
                # the close crosses the fast EMA (it was on the trade's side of it one bar earlier)
                if long:
                    below_fast = bar["close"] < bar["ema_fast"] and bar["prev_close"] >= bar["prev_fast"]
                else:
                    below_fast = bar["close"] > bar["ema_fast"] and bar["prev_close"] <= bar["prev_fast"]
                if below_mid:
                    close_part(pos, pos["shares_left"], bar["close"], ts, f"crossed {mid_label()}")
                elif below_fast and not pos["first_exit_done"]:
                    part = pos["shares"] * FIRST_EXIT_FRACTION
                    part = part if (long and ALLOW_FRACTIONAL_SHARES) else max(1, int(part))
                    part = min(part, pos["shares_left"])
                    close_part(pos, part, bar["close"], ts, f"crossed EMA {EMA_FAST}")
                    pos["first_exit_done"] = True
            if pos["shares_left"] <= 1e-9:
                finish_trade(pos, trades)
                closed_at[symbol] = ts
                del open_positions[symbol]

        # 2. Entries - highest relative volume first
        if bar_close_time > last_entry or len(open_positions) >= NUM_SLOTS:
            continue
        candidates = []
        for symbol, g in by_symbol_time.items():
            if symbol in open_positions or ts not in g.index or bar_close_time < added_at[symbol]:
                continue
            if not ALLOW_REENTRY and entries_today.get(symbol, 0) > 0:
                continue
            if MAX_TRADES_PER_STOCK_PER_DAY and entries_today.get(symbol, 0) >= MAX_TRADES_PER_STOCK_PER_DAY:
                continue
            if closed_at.get(symbol) == ts:
                continue  # never re-enter on the bar that closed the trade
            bar = g.loc[ts]
            side = entry_signal(bar)
            if side:
                candidates.append((rvol[symbol], symbol, side, bar))
        for _, symbol, side, bar in sorted(candidates, key=lambda c: c[0], reverse=True):
            if len(open_positions) >= NUM_SLOTS:
                break
            price = fill_price(bar["close"], side, "entry")
            shares = slot_value / price
            shares = shares if (side == "long" and ALLOW_FRACTIONAL_SHARES) else int(shares)
            if shares <= 0:
                continue
            open_positions[symbol] = {
                "date": day, "symbol": symbol, "side": side, "scan_time": scan_of[symbol], "rvol": rvol[symbol],
                "entry_time": ts + timedelta(minutes=BAR_MINUTES), "entry_price": price, "shares": shares,
                "shares_left": shares, "first_exit_done": False, "exits": [],
            }
            entries_today[symbol] = entries_today.get(symbol, 0) + 1

    # Anything still open when the data ends (should not happen before EXIT_ALL_TIME)
    for symbol, pos in open_positions.items():
        last = by_symbol_time[symbol].iloc[-1]
        close_part(pos, pos["shares_left"], last["close"], by_symbol_time[symbol].index[-1], "end of data")
        finish_trade(pos, trades)
    return trades


# ============================================================================
# REPORT
# ============================================================================

def max_drawdown(equity):
    peak = equity.cummax()
    return float(((equity - peak) / peak).min() * 100) if len(equity) else 0.0


def write_report(trades, daily, out_dir, start_capital):
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily)
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    daily_df.to_csv(out_dir / "daily_summary.csv", index=False)

    lines = ["=" * 64, "BACKTEST SUMMARY", "=" * 64,
             f"Days tested       : {len(daily_df)}  ({', '.join(daily_df['date']) if len(daily_df) else '-'})",
             f"Scan times ON     : {', '.join(enabled_scan_times())}",
             f"Longs / Shorts    : {'ON' if ENABLE_LONGS else 'OFF'} / {'ON' if ENABLE_SHORTS else 'OFF'}",
             f"Allocated capital : ${start_capital:,.2f} in {NUM_SLOTS} slots of ${start_capital / NUM_SLOTS:,.2f}"]
    if trades_df.empty:
        lines.append("No trades.")
    else:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]
        total = trades_df["pnl"].sum()
        equity = start_capital + daily_df["pnl"].cumsum()
        lines += [
            f"Trades            : {len(trades_df)}  (long {sum(trades_df['side'] == 'long')}, short {sum(trades_df['side'] == 'short')})",
            f"Win rate          : {len(wins) / len(trades_df) * 100:.1f}%",
            f"Total P&L         : ${total:,.2f}  ({total / start_capital * 100:+.2f}% of allocated capital)",
            f"Average win/loss  : ${wins['pnl'].mean() if len(wins) else 0:,.2f} / ${losses['pnl'].mean() if len(losses) else 0:,.2f}",
            f"Profit factor     : {wins['pnl'].sum() / abs(losses['pnl'].sum()) if losses['pnl'].sum() else float('inf'):.2f}",
            f"Max drawdown      : {max_drawdown(pd.concat([pd.Series([start_capital]), equity])):.2f}% (end of day)",
            "", "By scan time:"]
        for scan, g in trades_df.groupby("scan_time"):
            lines.append(f"  {scan}: {len(g)} trades, win rate {(g['pnl'] > 0).mean() * 100:.1f}%, P&L ${g['pnl'].sum():,.2f}")
        lines += ["", "By side:"]
        for side, g in trades_df.groupby("side"):
            lines.append(f"  {side}: {len(g)} trades, win rate {(g['pnl'] > 0).mean() * 100:.1f}%, P&L ${g['pnl'].sum():,.2f}")
        lines += ["", "By exit:"]
        for reason, g in trades_df.groupby("exit_2_reason"):
            lines.append(f"  {reason}: {len(g)} trades, P&L ${g['pnl'].sum():,.2f}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(range(len(equity) + 1), [start_capital] + list(equity), marker="o", color="#1a5d7a")
            ax.set_xticks(range(len(equity) + 1))
            ax.set_xticklabels(["start"] + list(daily_df["date"]), rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Allocated capital ($)")
            ax.set_title("Equity curve (end of each test day)")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "equity_curve.png", dpi=120)
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not draw the equity curve: {e}")

    lines += ["", f"Files: {out_dir}/trades.csv, daily_summary.csv, watchlist.csv, summary.txt, equity_curve.png"]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not enabled_scan_times():
        raise SystemExit("Switch on at least one scan time in SCAN_TIMES.")

    capital = ACCOUNT_EQUITY * ALLOCATION_PCT / 100
    start_capital = capital
    print("=" * 64)
    print("RVOL + EMA INTRADAY BACKTEST (EODHD data)")
    print(f"Allocated: ${capital:,.0f} ({ALLOCATION_PCT}% of ${ACCOUNT_EQUITY:,.0f}) in {NUM_SLOTS} slots")
    print(f"Scans: {', '.join(enabled_scan_times())} | RVOL >= {MIN_RELATIVE_VOLUME} | EMA {EMA_FAST} / {mid_label()} / EMA {EMA_SLOW} | {BAR_MINUTES}-min bars")
    print(f"Longs {'ON' if ENABLE_LONGS else 'OFF'} | Shorts {'ON' if ENABLE_SHORTS else 'OFF'} | Stop {STOP_LOSS_PCT}% | Exit all {EXIT_ALL_TIME}")
    print("=" * 64)

    out_dir = Path(RESULTS_DIR) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    trades, daily, watchlists = [], [], []
    try:
        # 1. Trading days and the test days
        lookback_start = (pd.Timestamp(TEST_START_DATE) - timedelta(days=70)).strftime("%Y-%m-%d")
        all_days = get_trading_days(lookback_start, TEST_END_DATE)
        in_range = [d for d in all_days if TEST_START_DATE <= d <= TEST_END_DATE]
        test_days = sorted(random.Random(RANDOM_SEED).sample(in_range, RANDOM_DAYS)) \
            if RANDOM_DAYS and len(in_range) > RANDOM_DAYS else in_range
        print(f"\nTest days: {', '.join(test_days) if test_days else '(none)'}")
        if not test_days:
            return

        # 2. Stocks worth downloading: price and average daily volume before each test day
        print("\n📋 Building the stock list from daily data...")
        universe = get_universe()
        filters, candidates = {}, set()
        for day in test_days:
            f = daily_filters(day, all_days, universe)
            filters[day] = f
            ok = f[(f["avg_daily_volume"] >= MIN_AVG_DAILY_VOLUME) & (f["prev_close"] >= MIN_PRICE * PREFILTER_PRICE_FACTOR)]
            candidates |= set(ok.index)
            print(f"   {day}: {len(ok):,} of {len(universe):,} stocks pass price/volume before the open")

        # 3. 1-minute data needed (cached files are reused)
        days_back = max(RVOL_LOOKBACK_DAYS, warmup_days())
        chunks = date_chunks(test_days, all_days, days_back)
        jobs = [(t, s, e) for s, e in chunks for t in sorted(candidates)]
        todo = [j for j in jobs if not minutes_file(*j).exists()]
        print(f"\n⬇ 1-minute data: {len(jobs):,} ticker-ranges, {len(todo):,} not cached yet "
              f"= {len(todo) * CALL_COST['intraday']:,} EODHD calls (plus {_calls_used:,} used for daily data)")
        if ESTIMATE_CALLS_ONLY:
            print("ESTIMATE_CALLS_ONLY is on - nothing else downloaded.")
            return

        done = 0
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
            futures = [pool.submit(download_minutes, *j) for j in todo]
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 250 == 0 or done == len(todo):
                    print(f"   {done:,}/{len(todo):,} downloaded")

        # 4. Opening-window volume per stock and day, for the relative volume scans
        print("\n📊 Measuring volume at the scan times...")
        needed_days = sorted({d for day in test_days
                              for d in all_days[max(0, all_days.index(day) - RVOL_LOOKBACK_DAYS):all_days.index(day) + 1]})
        rows = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
            for result in pool.map(lambda t: scan_windows(t, chunks, needed_days), sorted(candidates)):
                rows += result
        windows = pd.DataFrame(rows, columns=["symbol", "date", "scan_time", "volume", "price"])

        # 5. Replay each test day
        for day in test_days:
            print(f"\n📅 {day}")
            idx = all_days.index(day)
            prior_days = all_days[max(0, idx - RVOL_LOOKBACK_DAYS):idx]
            watchlist = scan_day(day, prior_days, windows, filters[day])
            if watchlist.empty:
                daily.append({"date": day, "watchlist": 0, "trades": 0, "pnl": 0.0, "capital": round(capital, 2)})
                continue
            watchlists.append(watchlist.assign(date=day))

            bar_days = all_days[max(0, idx - warmup_days()):idx + 1]
            minutes = pd.concat([load_minutes(s, chunks, bar_days) for s in watchlist["symbol"]], ignore_index=True)
            bars = prepare_bars(minutes, day)
            day_trades = simulate_day(day, watchlist, bars, capital)
            day_pnl = sum(t["pnl"] for t in day_trades)
            trades += day_trades
            print(f"   💰 {len(day_trades)} trades, P&L ${day_pnl:,.2f}")
            daily.append({"date": day, "watchlist": len(watchlist), "trades": len(day_trades),
                          "pnl": round(day_pnl, 2), "capital": round(capital + day_pnl, 2)})
            if COMPOUND_DAILY:
                capital += day_pnl
    except CallLimitReached as e:
        print(f"\n⛔ {e}\n   Raise MAX_API_CALLS or test fewer days; downloads so far are cached. Results so far are below.")

    if watchlists:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(watchlists).to_csv(out_dir / "watchlist.csv", index=False)
    write_report(trades, daily, out_dir, start_capital)
    print(f"\nEODHD calls used this run: {_calls_used:,}")


if __name__ == "__main__":
    main()
