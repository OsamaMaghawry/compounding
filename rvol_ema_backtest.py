#!/usr/bin/env python3
"""
RVOL + EMA Intraday Backtest (Databento data)

How a test day is replayed:
1. SCANS (only at the scan times switched on below): find "stocks in play" by relative
   volume - volume from the open to the scan time, compared with the average volume over
   the same window on the previous RVOL_LOOKBACK_DAYS trading days - plus price and
   average daily volume filters. Every stock found is added to the day's watchlist and
   stays there for the rest of the day.
2. MONITORING (every minute, from the first scan until EXIT_ALL_TIME): checks exits on
   open positions and entries on every watchlist stock, on 1-minute bars.
   - LONG entry : close above EMA_SLOW (200) and EMA_MID (21), and EMA_FAST (9) crosses
                  back above EMA_MID (21) after dipping below it.
   - LONG exit  : first half when the close crosses below EMA_FAST (9), the rest when it
                  crosses below EMA_MID (21); STOP_LOSS_PCT stop; everything closed at
                  EXIT_ALL_TIME.
   - SHORT      : the mirror image (switch with ENABLE_SHORTS).
3. CAPITAL: ACCOUNT_EQUITY x ALLOCATION_PCT is split into NUM_SLOTS equal positions.

Install:  pip install databento pandas numpy matplotlib
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
EMA_SLOW = 200
ENABLE_LONGS = True
ENABLE_SHORTS = True
FIRST_EXIT_FRACTION = 0.5           # part sold at the EMA_FAST cross (0.5 = half)
STOP_LOSS_PCT = 5.0                 # % against the entry price; 0 = no stop
LAST_ENTRY_TIME = "15:30"           # no new entries after this time
EXIT_ALL_TIME = "15:45"             # every open position is closed at this time
ALLOW_REENTRY = True                # a stock can be traded again the same day after it is closed
EMA_WARMUP_DAYS = 2                 # previous trading days loaded so the 200 EMA is ready at the open
EMA_INCLUDE_EXTENDED_HOURS = False  # True = pre/after-market bars count in the EMAs

# ---------- CAPITAL ----------
ACCOUNT_EQUITY = 100_000            # account equity ($)
ALLOCATION_PCT = 10.0               # % of equity this strategy trades (10 -> $10,000)
NUM_SLOTS = 10                      # max positions at once; each gets allocated capital / NUM_SLOTS
COMPOUND_DAILY = False              # True = each test day starts with the previous day's ending capital
ALLOW_FRACTIONAL_SHARES = False     # longs only; shorts always use whole shares

# ---------- FILLS & COSTS ----------
SLIPPAGE_PCT = 0.02                 # % worse than the signal bar's close, on every fill
COMMISSION_PER_SHARE = 0.0          # $ per share, each side

# ---------- DATABENTO ----------
DATABENTO_API_KEY = "PUT_YOUR_DATABENTO_API_KEY_HERE"   # or set the DATABENTO_API_KEY environment variable
MINUTE_DATASET = "EQUS.MINI"        # 1-minute bars for the whole US market
DAILY_DATASET = "EQUS.SUMMARY"      # daily bars (consolidated volume) for the average daily volume filter
CACHE_DIR = "databento_cache"       # downloaded data is kept here, so re-running the same days costs nothing
MAX_COST_USD = 25.0                 # stop before any download that would push this run's Databento cost above this
ESTIMATE_COST_ONLY = False          # True = only print what the market-wide downloads would cost, download nothing

# ---------- OUTPUT ----------
RESULTS_DIR = "backtest_results"

# ============================================================================
# END OF SETTINGS
# ============================================================================

import os
import random
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"
SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)

_client = None
_spent_usd = 0.0
_estimated_usd = 0.0


class CostLimitReached(Exception):
    pass


# ============================================================================
# DATABENTO DATA (with local cache)
# ============================================================================

def get_client():
    global _client
    if _client is None:
        import databento as db
        key = os.environ.get("DATABENTO_API_KEY") or DATABENTO_API_KEY
        if not key or key.startswith("PUT_YOUR"):
            raise SystemExit("Set DATABENTO_API_KEY at the top of the file (or as an environment variable).")
        _client = db.Historical(key)
    return _client


def to_utc_iso(day, hhmm):
    ts = pd.Timestamp(f"{day} {hhmm}", tz=TZ)
    return ts.tz_convert("UTC").isoformat()


def fetch(dataset, schema, start, end, symbols, tag):
    """Download (or load from cache) one Databento request and return it as a DataFrame."""
    global _spent_usd, _estimated_usd
    import databento as db

    Path(CACHE_DIR).mkdir(exist_ok=True)
    cache_file = Path(CACHE_DIR) / f"{dataset}_{schema}_{tag}.dbn.zst"
    if not cache_file.exists():
        client = get_client()
        cost = client.metadata.get_cost(dataset=dataset, schema=schema, start=start, end=end,
                                        symbols=symbols, stype_in="raw_symbol")
        if ESTIMATE_COST_ONLY:
            _estimated_usd += cost
            print(f"   [estimate] {tag}: ${cost:.2f}")
            return None
        if _spent_usd + cost > MAX_COST_USD:
            raise CostLimitReached(f"Downloading {tag} (${cost:.2f}) would take this run over "
                                   f"MAX_COST_USD (${MAX_COST_USD:.2f}); ${_spent_usd:.2f} spent so far.")
        print(f"   ⬇ {tag} (${cost:.2f})")
        client.timeseries.get_range(dataset=dataset, schema=schema, start=start, end=end,
                                    symbols=symbols, stype_in="raw_symbol", path=str(cache_file))
        _spent_usd += cost

    store = db.DBNStore.from_file(str(cache_file))
    df = store.to_df(tz=TZ)
    if df.empty:
        return df
    df = df.reset_index()
    ts_col = "ts_event" if "ts_event" in df.columns else df.columns[0]
    return df.rename(columns={ts_col: "ts"})[["ts", "symbol", "open", "high", "low", "close", "volume"]]


def get_trading_days(start_date, end_date):
    """Trading days in a range, taken from SPY daily bars."""
    end = (pd.Timestamp(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = fetch(DAILY_DATASET, "ohlcv-1d", start_date, end, ["SPY"], f"SPY_{start_date}_{end_date}")
    if df is None:
        return []
    return sorted({d.strftime("%Y-%m-%d") for d in df["ts"]})


def get_daily_bars(before_day):
    """Daily bars for every stock over the ~2 months before a test day."""
    start = (pd.Timestamp(before_day) - timedelta(days=60)).strftime("%Y-%m-%d")
    return fetch(DAILY_DATASET, "ohlcv-1d", start, before_day, "ALL_SYMBOLS", f"ALL_{start}_{before_day}")


def get_market_minutes(day, start_hhmm, end_hhmm):
    """1-minute bars for every stock between two times of one day."""
    tag = f"ALL_{day}_{start_hhmm.replace(':', '')}-{end_hhmm.replace(':', '')}"
    return fetch(MINUTE_DATASET, "ohlcv-1m", to_utc_iso(day, start_hhmm), to_utc_iso(day, end_hhmm),
                 "ALL_SYMBOLS", tag)


def get_symbol_minutes(symbols, start_day, end_day):
    """1-minute bars (including extended hours) for a few symbols over several days."""
    if not symbols:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    frames = []
    symbols = sorted(symbols)
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        tag = f"SYM_{start_day}_{end_day}_{abs(hash(tuple(chunk))) % 10**10}"
        end = (pd.Timestamp(end_day) + timedelta(days=1)).strftime("%Y-%m-%d")
        df = fetch(MINUTE_DATASET, "ohlcv-1m", start_day, end, chunk, tag)
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================================================================
# SCANNING - STOCKS IN PLAY
# ============================================================================

def enabled_scan_times():
    return sorted(t for t, on in SCAN_TIMES.items() if on)


def window_volume(minutes, scan_hhmm):
    """Volume per symbol from the open up to (not including) the scan time."""
    t = minutes["ts"].dt.time
    mask = (t >= SESSION_OPEN) & (t < dtime.fromisoformat(scan_hhmm))
    return minutes[mask].groupby("symbol")["volume"].sum()


def scan_day(day, prior_days, daily_bars):
    """Run every enabled scan for one day. Returns today's market minutes and the watchlist."""
    scans = enabled_scan_times()
    last_scan = scans[-1]
    today = get_market_minutes(day, "09:30", last_scan)
    prior = {d: get_market_minutes(d, "09:30", last_scan) for d in prior_days}
    if ESTIMATE_COST_ONLY or today is None or today.empty:
        return pd.DataFrame()

    # Average daily volume before the test day
    daily = daily_bars[daily_bars["ts"].dt.strftime("%Y-%m-%d") < day]
    daily = daily.sort_values("ts").groupby("symbol").tail(AVG_VOLUME_DAYS)
    avg_daily_volume = daily.groupby("symbol")["volume"].mean()

    rows = []
    for scan in scans:
        today_vol = window_volume(today, scan)
        hist = pd.DataFrame({d: window_volume(m, scan) for d, m in prior.items() if m is not None and not m.empty})
        hist_days = hist.notna().sum(axis=1)
        hist_avg = hist.mean(axis=1)

        upto = today[today["ts"].dt.time < dtime.fromisoformat(scan)]
        last_price = upto.sort_values("ts").groupby("symbol")["close"].last()

        df = pd.DataFrame({"today_volume": today_vol, "avg_volume": hist_avg, "history_days": hist_days,
                           "price": last_price, "avg_daily_volume": avg_daily_volume}).dropna()
        df = df[(df["history_days"] >= MIN_RVOL_HISTORY_DAYS) & (df["avg_volume"] > 0)]
        df["rvol"] = df["today_volume"] / df["avg_volume"]
        df = df[(df["rvol"] >= MIN_RELATIVE_VOLUME) & (df["price"] >= MIN_PRICE)
                & (df["avg_daily_volume"] >= MIN_AVG_DAILY_VOLUME)]
        df["scan_time"] = scan
        rows.append(df.reset_index().rename(columns={"index": "symbol"}))
        print(f"   🔎 {day} {scan}: {len(df)} stocks in play")

    found = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if found.empty:
        return found
    # A stock stays on the list from the first scan that found it
    return found.sort_values("scan_time").drop_duplicates("symbol", keep="first").reset_index(drop=True)


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
        for name, span in (("ema_fast", EMA_FAST), ("ema_mid", EMA_MID), ("ema_slow", EMA_SLOW)):
            g[name] = g["close"].ewm(span=span, adjust=False).mean()
        g["prev_close"] = g["close"].shift(1)
        g["prev_fast"] = g["ema_fast"].shift(1)
        g["prev_mid"] = g["ema_mid"].shift(1)
        g["bars_seen"] = np.arange(1, len(g) + 1)
        g = g[(g.index.strftime("%Y-%m-%d") == day)
              & (g.index.time >= SESSION_OPEN) & (g.index.time < SESSION_CLOSE)]
        if not g.empty:
            out[symbol] = g
    return out


def entry_signal(bar):
    """'long', 'short' or None for one closed 1-minute bar."""
    if bar["bars_seen"] < EMA_SLOW or pd.isna(bar["prev_fast"]):
        return None  # 200 EMA not ready yet
    crossed_up = bar["ema_fast"] > bar["ema_mid"] and bar["prev_fast"] <= bar["prev_mid"]
    crossed_down = bar["ema_fast"] < bar["ema_mid"] and bar["prev_fast"] >= bar["prev_mid"]
    if ENABLE_LONGS and crossed_up and bar["close"] > bar["ema_mid"] and bar["close"] > bar["ema_slow"]:
        return "long"
    if ENABLE_SHORTS and crossed_down and bar["close"] < bar["ema_mid"] and bar["close"] < bar["ema_slow"]:
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
    pos["exits"].append({"time": ts + timedelta(minutes=1), "shares": shares, "price": exit_px, "reason": reason, "pnl": pnl})
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

    open_positions, trades, traded_today, closed_at = {}, [], set(), {}
    for ts in all_times:
        bar_close_time = (ts + timedelta(minutes=1)).time()   # decisions are made when the bar closes
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
                below_mid = bar["close"] < bar["ema_mid"] if long else bar["close"] > bar["ema_mid"]
                # the close crosses the fast EMA (it was on the trade's side of it one bar earlier)
                if long:
                    below_fast = bar["close"] < bar["ema_fast"] and bar["prev_close"] >= bar["prev_fast"]
                else:
                    below_fast = bar["close"] > bar["ema_fast"] and bar["prev_close"] <= bar["prev_fast"]
                if below_mid:
                    close_part(pos, pos["shares_left"], bar["close"], ts, f"crossed EMA {EMA_MID}")
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
            if not ALLOW_REENTRY and symbol in traded_today:
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
                "entry_time": ts + timedelta(minutes=1), "entry_price": price, "shares": shares,
                "shares_left": shares, "first_exit_done": False, "exits": [],
            }
            traded_today.add(symbol)

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

def pick_test_days():
    days = get_trading_days(TEST_START_DATE, TEST_END_DATE)
    if RANDOM_DAYS and len(days) > RANDOM_DAYS:
        days = sorted(random.Random(RANDOM_SEED).sample(days, RANDOM_DAYS))
    return days


def main():
    if not enabled_scan_times():
        raise SystemExit("Switch on at least one scan time in SCAN_TIMES.")

    capital = ACCOUNT_EQUITY * ALLOCATION_PCT / 100
    start_capital = capital
    print("=" * 64)
    print("RVOL + EMA INTRADAY BACKTEST")
    print(f"Allocated: ${capital:,.0f} ({ALLOCATION_PCT}% of ${ACCOUNT_EQUITY:,.0f}) in {NUM_SLOTS} slots")
    print(f"Scans: {', '.join(enabled_scan_times())} | RVOL >= {MIN_RELATIVE_VOLUME} | EMAs {EMA_FAST}/{EMA_MID}/{EMA_SLOW}")
    print(f"Longs {'ON' if ENABLE_LONGS else 'OFF'} | Shorts {'ON' if ENABLE_SHORTS else 'OFF'} | Stop {STOP_LOSS_PCT}% | Exit all {EXIT_ALL_TIME}")
    print("=" * 64)

    all_days = get_trading_days(
        (pd.Timestamp(TEST_START_DATE) - timedelta(days=45)).strftime("%Y-%m-%d"), TEST_END_DATE)
    test_days = pick_test_days()
    print(f"\nTest days: {', '.join(test_days) if test_days else '(none)'}")

    out_dir = Path(RESULTS_DIR) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    trades, daily, watchlists = [], [], []
    try:
        for day in test_days:
            print(f"\n📅 {day}")
            idx = all_days.index(day)
            prior_days = all_days[max(0, idx - RVOL_LOOKBACK_DAYS):idx]
            warmup_days = all_days[max(0, idx - EMA_WARMUP_DAYS):idx]

            daily_bars = get_daily_bars(day)
            watchlist = scan_day(day, prior_days, daily_bars)
            if ESTIMATE_COST_ONLY:
                continue
            if watchlist.empty:
                daily.append({"date": day, "watchlist": 0, "trades": 0, "pnl": 0.0, "capital": round(capital, 2)})
                continue
            watchlists.append(watchlist.assign(date=day))

            minutes = get_symbol_minutes(list(watchlist["symbol"]), warmup_days[0] if warmup_days else day, day)
            bars = prepare_bars(minutes, day)
            day_trades = simulate_day(day, watchlist, bars, capital)
            day_pnl = sum(t["pnl"] for t in day_trades)
            trades += day_trades
            print(f"   💰 {len(day_trades)} trades, P&L ${day_pnl:,.2f}")
            daily.append({"date": day, "watchlist": len(watchlist), "trades": len(day_trades),
                          "pnl": round(day_pnl, 2), "capital": round(capital + day_pnl, 2)})
            if COMPOUND_DAILY:
                capital += day_pnl
    except CostLimitReached as e:
        print(f"\n⛔ {e}\n   Raise MAX_COST_USD or test fewer days. Results so far are below.")

    if ESTIMATE_COST_ONLY:
        print(f"\nEstimated cost of the market-wide downloads: ${_estimated_usd:.2f} "
              f"(plus a little for the watchlist stocks' full-day bars). Nothing was downloaded.")
        return

    if watchlists:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(watchlists).to_csv(out_dir / "watchlist.csv", index=False)
    write_report(trades, daily, out_dir, start_capital)
    print(f"\nDatabento spend this run: ${_spent_usd:.2f}")


if __name__ == "__main__":
    main()
