#!/usr/bin/env python3
"""Risk work on ONE setup: 5-min range, stop order at the range edge, 1 stock a day (first trigger, highest RVOL).
Tests sizing by risk, tighter stops, a wide-range filter and SPY filters, on the best exits, over 3 months.
Saved data only, Trade The Pool commission. Writes results/orb/risk_5min_stop_1stock.csv"""
import itertools
import pickle
import numpy as np
import pandas as pd
import orb_backtest as o
import orb_exits as ox

MONTHS = [("2025-11-17", "2025-12-16"), ("2025-12-17", "2026-01-16"), ("2026-01-17", "2026-02-16")]
EXIT_NAMES = ["ADR trail only: trail 25% ADR once at BE", "No targets: all trails 20 EMA 1-min once at BE",
              "BE at 0.75R", "Thirds at 1R, 2R, 3R, stop steps up",
              "ADR targets: 50% at 25% ADR, 25% at 50% ADR, runner trail 15% ADR"]
STOPS = [(0.0, "stop: other side of range"), (0.25, "stop: 1/4 into the range"), (0.5, "stop: middle of range")]
FILTERS = ["none", "range <= 50% ADR", "SPY 1st candle", "SPY vs VWAP", "range <= 50% ADR + SPY 1st candle",
           "range <= 50% ADR + SPY vs VWAP"]
SIZING = [("EQUAL", 0, "$10,000 every trade")]


def spy_state():
    df = o.load_minutes("SPY")
    out = {}
    for day, g in df.groupby("day"):
        a = g[["m", "open", "high", "low", "close", "volume"]].to_numpy(float)
        a = a[a[:, 0] >= 0]
        if len(a) < 10:
            continue
        tp = (a[:, 2] + a[:, 3] + a[:, 4]) / 3
        vwap = np.cumsum(tp * a[:, 5]) / np.maximum(np.cumsum(a[:, 5]), 1)
        first = a[a[:, 0] < 5]
        out[day] = {"first_up": first[-1, 4] > first[0, 1], "m": a[:, 0], "c": a[:, 4], "vwap": vwap}
    return out


def keep(t, f, spy):
    s = 1 if t["side"] == "long" else -1
    if "range" in f and t["R"] > 0.5 * t["adr"]:
        return False
    st = spy.get(t["day"])
    if "SPY 1st" in f:
        up = bool(st["first_up"]) if st is not None else None
        if up is None or (up if s == 1 else not up) is False:
            return False
    if "VWAP" in f:
        if st is None:
            return False
        i = np.searchsorted(st["m"], t["entry_m"]) - 1          # last SPY bar finished before the entry minute
        if i < 0 or (st["c"][i] - st["vwap"][i]) * s <= 0:
            return False
    return True


def main():
    spy = spy_state()
    adr, datr = ox.adr_table(), ox.atr_table()
    o.MIN_PRICE = 20.0
    ox.ENTRY = {"name": "5-min stop", "or": 5, "type": "stop_order"}
    ox.ENTRY_MODE = "breakout"
    exits = {n: (x, kw) for n, x, kw in ox.EXITS if n in EXIT_NAMES}
    trades = {}   # (exit, stop) -> list over all months
    for s, e in MONTHS:
        cands = pickle.load(open(f"/tmp/claude-0/orb_days2_{s}_{e}.pkl", "rb"))
        for (xn, (x, kw)), (sf, _) in itertools.product(exits.items(), STOPS):
            ts = ox.trades(cands, dict(x, stop_frac=sf), adr, datr_tab=datr, **kw)
            for t in ts:
                t["news"] = None
            trades.setdefault((xn, sf), []).extend(ts)
        print(f"trades built: {s}", flush=True)
    days = [d for d in o.trading_days() if MONTHS[0][0] <= d <= MONTHS[-1][1]]
    rows = []
    for (xn, sf), ts in trades.items():
        for f in FILTERS:
            pool = [t for t in ts if keep(t, f, spy)]
            for (sz, rp, szname), (lo, label) in itertools.product(SIZING, ((False, "Long+short"), (True, "Long only"))):
                taken = o.portfolio(pool, 1, lo, news_required=False, commission_pct="TTP", max_per_day=1,
                                    sizing=sz, risk_pct=rp)
                st = o.stats(taken, days)
                df = pd.DataFrame(taken)
                r = {"exit": xn, "stop": dict(STOPS)[sf], "filter": f, "sizing": szname, "strategy": label, **st}
                for i, (ms, me) in enumerate(MONTHS, 1):
                    r[f"m{i}"] = round(df[(df.day >= ms) & (df.day <= me)].pnl.sum(), 0) if len(df) else 0
                r["avg_position"] = round(df.value.mean(), 0) if len(df) else 0
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{o.OUT_DIR}/risk_5min_stop_1stock.csv", index=False)
    print("DONE", len(rows))


if __name__ == "__main__":
    main()
