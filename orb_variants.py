#!/usr/bin/env python3
"""
ORB rule variants on one month: different entries and exits on the same stocks in play.
Uses the saved data and news (market-data repo) and the settings of orb_backtest.py (dates, price, RVOL, news).
Writes results/orb/variants_<start>_<end>.md and .csv.
"""
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import orb_backtest as o

OR_LENGTHS = (5, 15, 30)
CACHE = Path(f"/tmp/claude-0/orb_days2_{o.TEST_START}_{o.TEST_END}.pkl")   # derived from saved data, rebuilt for free
SLIP = o.SLIPPAGE_PCT / 100
EXIT_AT = o.hm(o.EXIT_TIME)
LAST_ENTRY = o.hm(o.LAST_ENTRY)


# ---------------------------------------------------------------------------- stocks in play per day
def candidate_days(ticker):
    """Days on which the stock is in play (RVOL of the first 15 minutes >= MIN_RVOL, price, volume)."""
    df = o.load_minutes(ticker)
    if df is None:
        return []
    days = o.trading_days()
    info = o.daily_info(ticker, days)
    hist, out = {}, []
    for day, g in df.groupby("day", sort=True):
        a = g[["m", "open", "high", "low", "close", "volume"]].to_numpy(float)
        first = a[a[:, 0] < 15]
        if len(first) < 9 or first[0, 0] > 2:
            continue
        v15, v5 = first[:, 5].sum(), a[a[:, 0] < 5][:, 5].sum()
        i = days.index(day)
        prior = [hist[d] for d in days[max(0, i - o.RVOL_LOOKBACK_DAYS):i] if d in hist]
        hist[day] = (v5, v15)
        if not (o.TEST_START <= day <= o.TEST_END) or day not in info or len(prior) < o.MIN_RVOL_HISTORY_DAYS:
            continue
        m5, m15 = np.mean([p[0] for p in prior]), np.mean([p[1] for p in prior])
        rvol = v15 / m15 if m15 > 0 else 0          # first 15 minutes (known at 09:45)
        rvol5 = v5 / m5 if m5 > 0 else 0            # first 5 minutes (known at 09:35)
        prev_close, adv = info[day]
        price5 = a[a[:, 0] < 5][-1, 4] if (a[:, 0] < 5).any() else first[-1, 4]
        if max(rvol, rvol5) < o.MIN_RVOL or min(first[-1, 4], price5) < o.MIN_PRICE or adv < o.MIN_AVG_DAILY_VOLUME:
            continue
        out.append({"ticker": ticker, "day": day, "rvol": round(rvol, 2), "rvol5": round(rvol5, 2), "price5": price5,
                    "gap": round((a[0, 1] / prev_close - 1) * 100, 2), "bars": a})
    return out


def load_candidates():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    days = o.trading_days()
    tickers = sorted({p.name[:-7] for p in Path(o.DATA_DIR, "minutes").glob("*/*.csv.gz")
                      if p.parent.name.split("_")[0] <= o.TEST_END and p.parent.name.split("_")[1] >= o.TEST_START} - {"SPY"})
    t0, out = time.time(), []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for res in pool.map(candidate_days, tickers, chunksize=8):
            out += res
    print(f"stocks in play: {len(out):,} stock-days ({time.time() - t0:.0f}s)", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(CACHE, "wb"))
    return out


# ---------------------------------------------------------------------------- one trade
def manage(s, entry, R, risk_unit, bars, x):
    """Exit rules x on the bars after entry. R = range size; levels are multiples of x['unit'] (range or risk)."""
    unit = R if x["unit"] == "range" else risk_unit
    stop = x["stop"]
    targets = [(entry + s * k * unit, frac) for k, frac in x["targets"]]
    remaining, pnl, hit, be, best, exit_m = 1.0, 0.0, 0, False, entry, None
    for m, op, h, l, c in bars:
        if m >= EXIT_AT:
            pnl += remaining * (op * (1 - s * SLIP) - entry) * s; remaining = 0; exit_m = m; break
        if x.get("time_stop") and not be and m - bars[0][0] >= x["time_stop"] and (c - entry) * s < 0.25 * unit:
            pnl += remaining * (c * (1 - s * SLIP) - entry) * s; remaining = 0; exit_m = m; break
        adverse, fav = (l, h) if s == 1 else (h, l)
        if (adverse - stop) * s <= 0:
            fill = op if (op - stop) * s <= 0 else stop
            pnl += remaining * (fill * (1 - s * SLIP) - entry) * s; remaining = 0; exit_m = m; break
        best = max(best, fav) if s == 1 else min(best, fav)
        for i in range(hit, len(targets)):
            price, frac = targets[i]
            if (fav - price) * s < 0:
                break
            fill = op if (op - price) * s >= 0 else price
            pnl += frac * (fill * (1 - s * SLIP) - entry) * s; remaining -= frac; hit = i + 1
        if x.get("be") and not be and (fav - (entry + s * x["be"] * unit)) * s >= 0:
            stop = max(stop, entry) if s == 1 else min(stop, entry); be = True
        if x.get("trail") and hit >= x.get("trail_after", 0) and remaining > 1e-9 and \
                (x.get("trail_after", 0) > 0 or (best - entry) * s >= x.get("trail_from", 0) * unit):
            t = best - s * x["trail"] * unit
            stop = max(stop, t) if s == 1 else min(stop, t)
        if remaining <= 1e-9:
            exit_m = m; break
    if remaining > 1e-9:
        m, op, h, l, c = bars[-1]
        pnl += remaining * (c * (1 - s * SLIP) - entry) * s; exit_m = m
    return pnl, exit_m


def find_entries(cand, e):
    """Entries of rule e on one stock-day: {side: (entry_m, entry_price, orh, orl, index of next bar)}."""
    a = cand["bars"]
    n = e["or"]
    rng = a[a[:, 0] < n]
    if len(rng) < n * 0.6 or rng[0, 0] > 2:
        return {}
    orh, orl = rng[:, 2].max(), rng[:, 3].min()
    if orh <= orl:
        return {}
    avg_vol = rng[:, 5].mean()
    after_idx = np.where(a[:, 0] >= n)[0]
    if not len(after_idx):
        return {}
    start = after_idx[0]
    out = {}
    if e.get("type") == "stop_order":                      # buy stop at the range high / sell stop at the low
        for j in range(start, len(a)):
            m, op, h, l, c, v = a[j]
            if m > LAST_ENTRY:
                break
            if "long" not in out and h > orh:
                out["long"] = (m, max(op, orh) * (1 + SLIP), orh, orl, j + 1)
            if "short" not in out and l < orl:
                out["short"] = (m, min(op, orl) * (1 - SLIP), orh, orl, j + 1)
            if len(out) == 2:
                break
        return out
    k = e.get("bar", 1)                                # confirmation bar size (1 or 5 minutes)
    j = start
    while j < len(a):
        if k == 1:
            m, op, h, l, c, v = a[j]; last = j
        else:                                          # 5-minute bar from 1-minute bars
            bucket = (a[j, 0] - n) // k
            last = j
            while last + 1 < len(a) and (a[last + 1, 0] - n) // k == bucket:
                last += 1
            m, c, v = a[last, 0], a[last, 4], a[j:last + 1, 5].sum()
        if m > LAST_ENTRY or len(out) == 2:
            break
        if e.get("only_first") and j > start:
            break                                      # only the first candle after the range counts
        vol_ok = not e.get("vol") or v >= e["vol"] * avg_vol * k
        for side, broke in (("long", c > orh), ("short", c < orl)):
            if side in out or not broke or not vol_ok:
                continue
            if e.get("retest"):                        # wait for a pullback to the range edge: limit order there
                level = orh if side == "long" else orl
                for r in range(last + 1, len(a)):
                    rm, rop, rh, rl, rc, rv = a[r]
                    if rm > m + e["retest"] or rm > LAST_ENTRY:
                        break
                    if (side == "long" and rl <= level) or (side == "short" and rh >= level):
                        fill = min(rop, level) if side == "long" else max(rop, level)
                        out[side] = (rm, fill * (1 + (SLIP if side == "long" else -SLIP)), orh, orl, r + 1)
                        break
                else:
                    pass
                if side not in out:
                    out[side] = None
            else:
                s = 1 if side == "long" else -1
                out[side] = (m, c * (1 + s * SLIP), orh, orl, last + 1)
        j = last + 1
    return {k2: v2 for k2, v2 in out.items() if v2}


def trades_for(cand, e, x):
    out = []
    for side, (m, entry, orh, orl, nxt) in find_entries(cand, e).items():
        s = 1 if side == "long" else -1
        if e.get("gap_aligned") and cand["gap"] * s <= 0:
            continue
        R = orh - orl
        stop = (orl + R * x["stop_at"]) if s == 1 else (orh - R * x["stop_at"])
        risk = (entry - stop) * s
        if risk <= 0 or nxt >= len(cand["bars"]):
            continue
        xx = dict(x, stop=stop)
        pnl, exit_m = manage(s, entry, R, risk, cand["bars"][nxt:, :5], xx)
        out.append({"ticker": cand["ticker"], "day": cand["day"], "side": side, "entry_m": int(m), "entry": entry,
                    "risk": risk, "pnl_ps": pnl, "r": pnl / risk, "exit_m": int(exit_m), "rvol": cand["rvol"],
                    "gap": cand["gap"]})
    return out


# ---------------------------------------------------------------------------- versions
BASE_E = {"name": "Base: 15-min range, 1-min close beyond it, volume >= range average", "or": 15, "bar": 1, "vol": 1.0}
BASE_X = {"name": "Base: stop mid-range, BE at 1x, 50% at 1.5x, 25% at 2x, 25% trail 1x (x = range)",
          "unit": "range", "stop_at": 0.5, "be": 1.0, "targets": [(1.5, 0.5), (2.0, 0.25)], "trail": 1.0, "trail_after": 2}

ENTRIES = [
    BASE_E,
    {"name": "5-min opening range", "or": 5, "bar": 1, "vol": 1.0},
    {"name": "30-min opening range", "or": 30, "bar": 1, "vol": 1.0},
    {"name": "5-min candle close beyond the range", "or": 15, "bar": 5, "vol": 1.0},
    {"name": "Stop order at the range high/low (no wait, no volume check)", "or": 15, "type": "stop_order"},
    {"name": "No breakout-volume check", "or": 15, "bar": 1, "vol": 0},
    {"name": "Breakout volume >= 2x range average", "or": 15, "bar": 1, "vol": 2.0},
    {"name": "Retest: limit at the range edge after the break (30 min)", "or": 15, "bar": 1, "vol": 1.0, "retest": 30},
    {"name": "Only in the direction of the gap", "or": 15, "bar": 1, "vol": 1.0, "gap_aligned": True},
]
EXITS = [
    BASE_X,
    {"name": "Stop at the other side of the range", **{k: v for k, v in BASE_X.items() if k != "name"}, "stop_at": 0.0},
    {"name": "Targets in risk units (1.5x / 2x the stop distance, BE at 1x)",
     **{k: v for k, v in BASE_X.items() if k != "name"}, "unit": "risk"},
    {"name": "All out at 2x range (no partials, BE at 1x)", "unit": "range", "stop_at": 0.5, "be": 1.0,
     "targets": [(2.0, 1.0)]},
    {"name": "All out at 1x range (BE at 0.5x)", "unit": "range", "stop_at": 0.5, "be": 0.5, "targets": [(1.0, 1.0)]},
    {"name": "No targets: trail 1x range once +1x, BE at 1x, else end of day", "unit": "range", "stop_at": 0.5,
     "be": 1.0, "targets": [], "trail": 1.0, "trail_after": 0, "trail_from": 1.0},
    {"name": "Base + tighter runner trail (0.5x range)", **{k: v for k, v in BASE_X.items() if k != "name"}, "trail": 0.5},
    {"name": "Base + time stop (out after 30 min if not +0.25x)", **{k: v for k, v in BASE_X.items() if k != "name"},
     "time_stop": 30},
]


def run_version(cands, e, x, news, slots=5):
    trades = [t for c in cands for t in trades_for(c, e, x)]
    for t in trades:
        t["news"] = news.get((t["ticker"], t["day"]))
    res = {}
    for long_only in (False, True):
        taken = o.portfolio(trades, slots, long_only, news_required=True)
        st = o.stats(taken, DAYS)
        res["LO" if long_only else "LS"] = st
    return res, trades


DAYS = [d for d in o.trading_days() if o.TEST_START <= d <= o.TEST_END]


def news_lookup(cands):
    """News flag per stock-day from the saved news (fetches only stocks not saved yet, capped)."""
    o.NEWS_CALL_CAP = 1_000
    fake = [{"ticker": c["ticker"], "day": c["day"]} for c in cands]
    flagged = o.add_news_flags(fake, o.trading_days())
    return {(t["ticker"], t["day"]): t["news"] for t in flagged}


def main():
    cands = load_candidates()
    news = news_lookup(cands)
    rows = []
    combos = [(e, BASE_X) for e in ENTRIES] + [(BASE_E, x) for x in EXITS[1:]]
    for e, x in combos:
        res, _ = run_version(cands, e, x, news)
        for k, label in (("LS", "Long+short"), ("LO", "Long only")):
            st = res[k]
            rows.append({"entry": e["name"], "exit": x["name"], "strategy": label, **st})
        print(f"  {e['name'][:45]:45s} | {x['name'][:45]:45s} | LS {res['LS'].get('total')} | LO {res['LO'].get('total')}", flush=True)
    df = pd.DataFrame(rows)
    # best entries x best exits
    def score(name_col, name):
        sub = df[df[name_col] == name]
        return sub.total.sum()
    best_e = sorted({r["entry"] for r in rows}, key=lambda n: -score("entry", n))[:3]
    best_x = sorted({r["exit"] for r in rows}, key=lambda n: -score("exit", n))[:3]
    for en in best_e:
        for xn in best_x:
            e = next(i for i in ENTRIES if i["name"] == en); x = next(i for i in EXITS if i["name"] == xn)
            if (e is BASE_E) or (x is BASE_X):
                continue
            res, _ = run_version(cands, e, x, news)
            for k, label in (("LS", "Long+short"), ("LO", "Long only")):
                rows.append({"entry": e["name"], "exit": x["name"], "strategy": label, "combo": True, **res[k]})
    df = pd.DataFrame(rows)
    out = Path(o.OUT_DIR)
    df.to_csv(out / f"variants_{o.TEST_START}_{o.TEST_END}.csv", index=False)
    print(df[["entry", "exit", "strategy", "trades", "win_pct", "pf", "total", "without_best5", "max_dd"]].to_string(index=False))


if __name__ == "__main__":
    main()
