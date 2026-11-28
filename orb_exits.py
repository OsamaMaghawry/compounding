#!/usr/bin/env python3
"""
ORB exit variants with the entry fixed: 5-minute opening range, 1-minute close beyond it (volume >= range average),
stop at the other side of the range. Same stocks, dates and news as orb_variants.py; saved data only (no EODHD calls).
Writes results/orb/exits_<start>_<end>.md and .csv.
"""
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import orb_backtest as o
import orb_variants as v

ENTRY = {"name": "5-min range, 1-min close beyond, volume >= range average", "or": 5, "bar": 1, "vol": 1.0}
SLIP = o.SLIPPAGE_PCT / 100
EXIT_AT = o.hm(o.EXIT_TIME)
ADR_DAYS = 14


def ema(x, n):
    out, k = np.empty(len(x)), 2 / (n + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def adr_table():
    """Average daily range ($) of the previous ADR_DAYS days, per ticker and day."""
    out = {}
    for f in Path(o.DATA_DIR, "json").glob("eod_*_*.json.gz"):
        t = f.name[4:].rsplit("_", 2)[0]
        rows = sorted(json.load(gzip.open(f)), key=lambda r: r["date"])
        rng = [(r["date"], (r["high"] or 0) - (r["low"] or 0)) for r in rows if r.get("high")]
        for i in range(ADR_DAYS, len(rng)):
            out[(t, rng[i][0])] = float(np.mean([x for _, x in rng[i - ADR_DAYS:i]]))
    return out


def atr_table():
    """Daily ATR (14-day average true range, $) known before each day, per ticker."""
    out = {}
    for f in Path(o.DATA_DIR, "json").glob("eod_*_*.json.gz"):
        t = f.name[4:].rsplit("_", 2)[0]
        rows = [r for r in sorted(json.load(gzip.open(f)), key=lambda r: r["date"]) if r.get("high") and r.get("close")]
        trs = [max(r["high"] - r["low"], abs(r["high"] - p["close"]), abs(r["low"] - p["close"]))
               for p, r in zip(rows[:-1], rows[1:])]
        for i in range(ADR_DAYS, len(trs)):
            out[(t, rows[i + 1]["date"])] = float(np.mean(trs[i - ADR_DAYS:i]))
    return out


def series(bars):
    c, h, l, vol = bars[:, 4], bars[:, 2], bars[:, 3], bars[:, 5]
    tp = (h + l + c) / 3
    vwap = np.cumsum(tp * vol) / np.maximum(np.cumsum(vol), 1)
    # 5-minute closes and their EMA 9, known only at the close of each 5-minute bar
    bucket = (bars[:, 0] // 5).astype(int)
    last_in_bucket = np.r_[bucket[1:] != bucket[:-1], True]
    c5 = c[last_in_bucket]
    e5 = ema(c5, 9)
    ema9_5 = np.full(len(c), np.nan)
    ema9_5[np.where(last_in_bucket)[0]] = e5
    tr = np.maximum.reduce([h - l, np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])])
    atr = ema(tr, 14)
    return {"ema9": ema(c, 9), "ema20": ema(c, 20), "vwap": vwap, "ema9_5": ema9_5, "atr": atr}


def manage(s, entry, R, adr, bars, j0, ser, x, datr=None):
    """Runs exit rules x from bar j0. Units: 'R' = range size, 'ADR' = average daily range."""
    unit = {"R": R, "ADR": adr}
    stop = x["stop"]
    tgts = [(entry + s * k * unit[u], frac) for k, frac, u in x.get("targets", [])]
    remaining, pnl, hit, be, best, exit_m = 1.0, 0.0, 0, False, entry, None
    day_hi, day_lo = bars[:j0, 2].max(), bars[:j0, 3].min()

    def close_all(price, m):
        nonlocal pnl, remaining, exit_m
        pnl += remaining * (price * (1 - s * SLIP) - entry) * s
        remaining, exit_m = 0.0, m

    def active(rule):
        start = rule.get("start", "entry")
        if start == "entry":
            return True
        if start == "be":
            return be
        if start.startswith("t"):
            return hit >= int(start[1:])
        if start.startswith("R"):
            return (best - entry) * s >= float(start[1:]) * R
        return False

    for j in range(j0, len(bars)):
        m, op, h, l, c = bars[j, :5]
        if m >= EXIT_AT:
            close_all(op, m); break
        adverse, fav = (l, h) if s == 1 else (h, l)
        if (adverse - stop) * s <= 0:
            close_all(op if (op - stop) * s <= 0 else stop, m); break
        best = max(best, fav) if s == 1 else min(best, fav)
        day_hi, day_lo = max(day_hi, h), min(day_lo, l)
        for i in range(hit, len(tgts)):
            price, frac = tgts[i]
            if (fav - price) * s < 0:
                break
            fill = op if (op - price) * s >= 0 else price
            pnl += frac * (fill * (1 - s * SLIP) - entry) * s
            remaining -= frac
            hit = i + 1
            if x.get("step"):                    # stop to the previous target (entry after the first)
                lvl = entry if i == 0 else tgts[i - 1][0]
                stop = max(stop, lvl) if s == 1 else min(stop, lvl)
        if remaining <= 1e-9:
            exit_m = m; break
        be_move = x["be_atr"] * datr if x.get("be_atr") else (x["be"] * R if x.get("be") is not None else None)
        if be_move is not None and not be and (fav - (entry + s * be_move)) * s >= 0:
            lvl = entry + s * x.get("be_lock", 0) * R
            stop = max(stop, lvl) if s == 1 else min(stop, lvl); be = True
        for rule in x.get("trail", []):
            if not active(rule):
                continue
            kind = rule["type"]
            if kind in ("R", "ADR", "ATR"):      # price trail from the best price
                dist = rule["k"] * (ser["atr"][j] if kind == "ATR" else unit[kind])
                t = best - s * dist
                stop = max(stop, t) if s == 1 else min(stop, t)
            else:                                # close-based: EMA / VWAP cross
                line = ser[kind][j]
                if not np.isnan(line) and (c - line) * s < 0:
                    close_all(c, m); break
        if remaining <= 1e-9:
            break
        if x.get("adr_exhaust") and (day_hi - day_lo) >= x["adr_exhaust"] * adr:
            close_all(c, m); break
    if remaining > 1e-9:
        close_all(bars[-1, 4], bars[-1, 0])
    return pnl, exit_m


ENTRY_MODE = os.environ.get("ORB_ENTRY", "breakout")   # breakout | pb1 | pb2


def pullback_entries(cand, mode):
    """Breakout of the 5-min range (1-min close beyond it, volume >= range average), then WAIT for a pullback, then
    enter on a break of the most recent 5-min high (low for shorts).
    pb1: pullback = a 5-min candle that closes against the move (red for longs) or makes a lower high; the entry level is
         the high of the latest 5-min candle (it moves down while the pullback goes on); stop at the other side of the range;
         cancelled if price closes back inside the range's other side.
    pb2: like pb1, but the pullback must come back to within 0.25 x range of the breakout level (a retest), and the
         stop goes just below the pullback's low."""
    a = cand["bars"]
    rng = a[a[:, 0] < 5]
    if len(rng) < 3 or rng[0, 0] > 2:
        return {}
    orh, orl = rng[:, 2].max(), rng[:, 3].min()
    R = orh - orl
    if R <= 0:
        return {}
    avg_vol = rng[:, 5].mean()
    last_entry = o.hm(o.LAST_ENTRY)
    out = {}
    for side in ("long", "short"):
        s = 1 if side == "long" else -1
        level = orh if s == 1 else orl
        far = orl if s == 1 else orh
        broke_at = None
        for j in range(len(a)):
            m, op, h, l, c, vol = a[j]
            if m < 5:
                continue
            if m > last_entry:
                break
            if (c - level) * s > 0 and vol >= avg_vol:
                broke_at = j
                break
        if broke_at is None:
            continue
        cur, prev5, pulled, trigger, pb_ext = None, None, False, None, None
        for j in range(broke_at + 1, len(a)):
            m, op, h, l, c, vol = a[j]
            if m > last_entry:
                break
            if (c - far) * s < 0:                       # back through the other side of the range: setup failed
                break
            if trigger is not None and (h - trigger if s == 1 else trigger - l) > 0:
                ok = True
                if mode == "pb2":
                    ok = (pb_ext - (level + s * 0.25 * R)) * s <= 0
                if ok:
                    fill = (max(op, trigger) if s == 1 else min(op, trigger)) * (1 + s * SLIP)
                    stop = far if mode == "pb1" else pb_ext - s * 0.0005 * pb_ext
                    out[side] = (m, fill, orh, orl, j + 1, stop)
                    break
            b = int(m // 5)                             # 5-minute candles from 1-minute bars
            if cur is None or cur[0] != b:
                cur = [b, op, h, l, c]
            else:
                cur[2], cur[3], cur[4] = max(cur[2], h), min(cur[3], l), c
            last_of_bucket = j + 1 >= len(a) or int(a[j + 1, 0] // 5) != b
            if last_of_bucket:
                _, co, ch, cl, cc = cur
                against = (cc < co) if s == 1 else (cc > co)
                lower_high = prev5 is not None and ((ch < prev5[2]) if s == 1 else (cl > prev5[3]))
                if against or lower_high:
                    pulled = True
                if pulled:
                    trigger = ch if s == 1 else cl         # the most recent 5-min high (low for shorts)
                    ext = cl if s == 1 else ch
                    pb_ext = ext if pb_ext is None else (min(pb_ext, ext) if s == 1 else max(pb_ext, ext))
                prev5 = cur[:]
    return out


def trades(cands, x, adr, vwap_filter=False, datr_tab=None):
    out = []
    for cand in cands:
        a_ = adr.get((cand["ticker"], cand["day"]))
        if not a_:
            continue
        if ENTRY["or"] == 5 and (cand["rvol5"] < o.MIN_RVOL or cand["price5"] < o.MIN_PRICE):
            continue                             # stocks in play judged on what is known at 09:35
        if ENTRY_MODE == "breakout":
            ents = {k_: tuple(v_) + (None,) for k_, v_ in v.find_entries(cand, ENTRY).items()}
        else:
            ents = pullback_entries(cand, ENTRY_MODE)
        if not ents:
            continue
        bars = cand["bars"]
        ser = series(bars)
        for side, (m, entry, orh, orl, nxt, own_stop) in ents.items():
            s = 1 if side == "long" else -1
            if nxt >= len(bars):
                continue
            if vwap_filter and (bars[nxt - 1, 4] - ser["vwap"][nxt - 1]) * s < 0:
                continue
            R = orh - orl
            stop = own_stop if own_stop is not None else (orl if s == 1 else orh)
            risk = (entry - stop) * s
            if risk <= 0:
                continue
            d_atr = (datr_tab or {}).get((cand["ticker"], cand["day"]))
            if x.get("be_atr") and not d_atr:
                continue
            pnl, exit_m = manage(s, entry, R, a_, bars, nxt, ser, dict(x, stop=stop), d_atr)
            out.append({"ticker": cand["ticker"], "day": cand["day"], "side": side, "entry_m": int(m),
                        "entry": entry, "risk": risk, "pnl_ps": pnl, "r": pnl / risk, "exit_m": int(exit_m),
                        "rvol": cand["rvol5"] if ENTRY["or"] == 5 else cand["rvol"], "gap": cand["gap"]})
    return out


T = lambda k, f, u="R": (k, f, u)
BASE = {"be": 1.0, "targets": [T(1.5, .5), T(2.0, .25)], "trail": [{"type": "R", "k": 1.0, "start": "t2"}]}
EXITS = [
    ("Base (current): BE 1R, 50% 1.5R, 25% 2R, runner trail 1R", BASE, {}),
    ("BE at 0.25 x daily ATR", dict(BASE, be=None, be_atr=0.25), {}),
    # breakeven timing
    ("BE at 0.5R", dict(BASE, be=0.5), {}),
    ("BE at 0.75R", dict(BASE, be=0.75), {}),
    ("BE at 1.5R", dict(BASE, be=1.5), {}),
    ("No breakeven", dict(BASE, be=None), {}),
    ("BE at 1R, lock +0.25R", dict(BASE, be_lock=0.25), {}),
    # ADR
    ("ADR targets: 50% at 25% ADR, 25% at 50% ADR, runner trail 15% ADR", {"be": 1.0, "targets": [T(.25, .5, "ADR"), T(.5, .25, "ADR")],
     "trail": [{"type": "ADR", "k": .15, "start": "t2"}]}, {}),
    ("ADR trail only: trail 25% ADR once at BE", {"be": 1.0, "trail": [{"type": "ADR", "k": .25, "start": "be"}]}, {}),
    ("Base + out when the day's range reaches 100% ADR", dict(BASE, adr_exhaust=1.0), {}),
    ("Base + out when the day's range reaches 130% ADR", dict(BASE, adr_exhaust=1.3), {}),
    # moving averages
    ("Base, runner trails 9 EMA (1-min close)", dict(BASE, trail=[{"type": "ema9", "start": "t2"}]), {}),
    ("Base, runner trails 20 EMA (1-min close)", dict(BASE, trail=[{"type": "ema20", "start": "t2"}]), {}),
    ("Base, runner trails 9 EMA on 5-min bars", dict(BASE, trail=[{"type": "ema9_5", "start": "t2"}]), {}),
    ("50% at 1.5R, rest trails 9 EMA 5-min", {"be": 1.0, "targets": [T(1.5, .5)], "trail": [{"type": "ema9_5", "start": "t1"}]}, {}),
    ("No targets: all trails 9 EMA 5-min once at BE", {"be": 1.0, "trail": [{"type": "ema9_5", "start": "be"}]}, {}),
    ("No targets: all trails 20 EMA 1-min once at BE", {"be": 1.0, "trail": [{"type": "ema20", "start": "be"}]}, {}),
    # portions / steps
    ("Thirds at 1R, 2R, 3R, stop steps up", {"be": 1.0, "targets": [T(1, 1/3), T(2, 1/3), T(3, 1/3)], "step": True}, {}),
    ("Quarters at 1R, 2R, 3R, 4R, stop steps up", {"be": 1.0, "targets": [T(1, .25), T(2, .25), T(3, .25), T(4, .25)], "step": True}, {}),
    ("50% 1.5R, 25% 3R, runner trail 1.5R", {"be": 1.0, "targets": [T(1.5, .5), T(3, .25)], "trail": [{"type": "R", "k": 1.5, "start": "t2"}]}, {}),
    ("33% 1R, 33% 2R, runner trail 9 EMA 5-min, steps", {"be": 1.0, "targets": [T(1, 1/3), T(2, 1/3)], "step": True,
     "trail": [{"type": "ema9_5", "start": "t2"}]}, {}),
    # VWAP
    ("Base + out on a close back through VWAP", dict(BASE, trail=BASE["trail"] + [{"type": "vwap", "start": "entry"}]), {}),
    ("Base, runner trails VWAP", dict(BASE, trail=[{"type": "vwap", "start": "t2"}]), {}),
    ("Base + entry only on the right side of VWAP", BASE, {"vwap_filter": True}),
    # ATR
    ("Base, runner chandelier 3x 1-min ATR", dict(BASE, trail=[{"type": "ATR", "k": 3, "start": "t2"}]), {}),
    ("No targets: chandelier 3x 1-min ATR once at BE", {"be": 1.0, "trail": [{"type": "ATR", "k": 3, "start": "be"}]}, {}),
]


SETUPS = [  # (name, positions per day, how they are chosen)
    ("5 slots", 5, "first"),
    ("1 stock/day: first breakout", 1, "first"),
    ("2 stocks/day: first breakouts", 2, "first"),
]


def select(ts, k, pick, need_news):
    """Keeps only trades in the top-k stocks of each day, ranked at 09:35 (RVOL of the first 5 minutes or gap).
    The ranking uses every stock in play that day, whether or not it breaks out later."""
    ranked = {}
    for t in CANDS:
        if need_news and not NEWS.get((t["ticker"], t["day"])):
            continue
        if t["rvol5"] < o.MIN_RVOL or t["price5"] < o.MIN_PRICE:
            continue
        ranked.setdefault(t["day"], []).append(t)
    top = set()
    for day, lst in ranked.items():
        key = (lambda c: c["rvol5"]) if pick == "rvol" else (lambda c: abs(c["gap"]))
        top |= {(c["ticker"], day) for c in sorted(lst, key=key, reverse=True)[:k]}
    return [t for t in ts if (t["ticker"], t["day"]) in top]


CANDS, NEWS = [], {}
COMMISSION_PCT = 0.5   # $5 per $1,000 traded, on the entry and on the exit


def construction(all_trades, taken, need_news, long_only):
    """How the portfolio was invested: signals, positions per day, slots full, capital in use."""
    pool = [t for t in all_trades if (not need_news or t["news"]) and (t["side"] == "long" or not long_only)]
    sig = pd.DataFrame(pool)
    df = pd.DataFrame(taken)
    if df.empty:
        return {}
    per_day = df.groupby("day").size().reindex(v.DAYS, fill_value=0)
    # most positions open at the same time, per day, and capital in use at that moment
    peak, peak_val = [], []
    for day, g in df.groupby("day"):
        ev = sorted([(r.entry_m, 1, r.value) for r in g.itertuples()] + [(r.exit_m, -1, -r.value) for r in g.itertuples()],
                    key=lambda e_: (e_[0], e_[1]))
        n = val = mx = mxv = 0
        for _, d, val_ in ev:
            n += d; val += val_; mx = max(mx, n); mxv = max(mxv, val)
        peak.append(mx); peak_val.append(mxv)
    sig_days = sig.groupby("day").size() if len(sig) else pd.Series(dtype=int)
    return {
        "signals": len(sig), "trades_taken": len(df), "signals_skipped_slots_full": len(sig) - len(df) if not long_only else None,
        "days_traded": int((per_day > 0).sum()), "trades_per_day_avg": round(per_day.mean(), 1),
        "trades_per_day_max": int(per_day.max()), "max_open_at_once_avg": round(np.mean(peak), 1),
        "days_all_5_slots_full": int(sum(1 for p in peak if p >= 5)),
        "peak_capital_in_use_avg": round(np.mean(peak_val), 0),
        "avg_hold_min": round((df.exit_m - df.entry_m).mean(), 0),
        "avg_position": round(df.value.mean(), 0), "longs": int((df.side == "long").sum()), "shorts": int((df.side == "short").sum()),
    }


FROZEN = ["Base (current): BE 1R, 50% 1.5R, 25% 2R, runner trail 1R",
          "No targets: all trails 9 EMA 5-min once at BE",
          "ADR trail only: trail 25% ADR once at BE",
          "Thirds at 1R, 2R, 3R, stop steps up",
          "BE at 0.5R",
          "BE at 0.25 x daily ATR"]


def main():
    global EXITS
    if os.environ.get("ORB_SET") == "frozen":
        EXITS = [x for x in EXITS if x[0] in FROZEN]
    cands = v.load_candidates()
    o.NEWS_CALL_CAP = int(os.environ.get("ORB_NEWS_CAP", "0"))   # 0 = saved news only, no EODHD calls
    flagged = o.add_news_flags([{"ticker": c["ticker"], "day": c["day"]} for c in cands], o.trading_days())
    news = {(t["ticker"], t["day"]): t["news"] for t in flagged}
    CANDS[:] = cands
    NEWS.update(news)
    adr = adr_table()
    datr = atr_table()
    rows, build = [], []
    for name, x, kw in EXITS:
        ts = trades(cands, x, adr, datr_tab=datr, **kw)
        for t in ts:
            t["news"] = news.get((t["ticker"], t["day"]))
        for need_news in (True, False):
          for setup, k, pick in SETUPS:
            for long_only, label in ((False, "Long+short"), (True, "Long only")):
                for comm in (0.0, COMMISSION_PCT):
                    pool = select(ts, k, pick, need_news) if pick != "first" else ts
                    taken = o.portfolio(pool, k, long_only, news_required=need_news, commission_pct=comm,
                                        max_per_day=k if setup != "5 slots" else 0)
                    st = o.stats(taken, v.DAYS)
                    rows.append({"exit": name, "setup": setup, "news": "news" if need_news else "no news", "strategy": label,
                                 "commission": "$5 per $1,000" if comm else "none", **st})
                    if not comm:
                        build.append({"exit": name, "setup": setup, "news": "news" if need_news else "no news", "strategy": label,
                                      **construction(ts, taken, need_news, long_only)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{o.OUT_DIR}/exits_{ENTRY_MODE}_{o.TEST_START}_{o.TEST_END}.csv", index=False)
    pd.DataFrame(build).to_csv(f"{o.OUT_DIR}/construction_{ENTRY_MODE}_{o.TEST_START}_{o.TEST_END}.csv", index=False)
    cols = ["exit", "setup", "news", "commission", "strategy", "trades", "win_pct", "pf", "total", "without_best5", "avg_r", "long_pnl", "short_pnl",
            "worst_day", "max_dd"]
    with open(f"{o.OUT_DIR}/exits_{ENTRY_MODE}_{o.TEST_START}_{o.TEST_END}.md", "w") as f:
        f.write(f"# ORB exit variants - {o.TEST_START} to {o.TEST_END}\n\nEntry fixed: {ENTRY['name']}; stop at the other "
                f"side of the range. Stocks above ${o.MIN_PRICE:.0f}, RVOL >= {o.MIN_RVOL}, news, 5 slots, ${o.CAPITAL:,}.\n\n")
        f.write(df[cols].to_markdown(index=False))


if __name__ == "__main__":
    main()
