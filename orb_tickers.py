#!/usr/bin/env python3
"""ORB on fixed tickers (Mag 7, QQQ, TQQQ, SPY): each ticker alone with $10,000 and at most 1 trade a day, plus a
Mag 7 basket (1 stock a day, highest relative volume first). No relative-volume or news filter. Saved data only.
Trade The Pool commission. Writes results/orb/tickers_<start>_<end>.csv"""
import os
import numpy as np
import pandas as pd
import orb_backtest as o
import orb_variants as v
import orb_exits as ox

o.MIN_RVOL = 0.0
o.MIN_PRICE = 0.0
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
TICKERS = MAG7 + ["QQQ", "TQQQ", "SPY"]
ENTRIES = [
    ("5-min range, stop order", {"or": 5, "type": "stop_order"}),
    ("5-min range, 1-min close beyond", {"or": 5, "bar": 1, "vol": 0}),
    ("15-min range, stop order", {"or": 15, "type": "stop_order"}),
    ("15-min range, 1-min close beyond", {"or": 15, "bar": 1, "vol": 0}),
]
MONTHS = [("2025-11-17", "2025-12-16"), ("2025-12-17", "2026-01-16"), ("2026-01-17", "2026-02-16")]


def main():
    cands = {}
    for t in TICKERS:
        cands[t] = v.candidate_days(t)
        print(f"{t}: {len(cands[t])} days", flush=True)
    adr, datr = ox.adr_table(), ox.atr_table()
    rows = []
    groups = [(t, [t]) for t in TICKERS] + [("Mag 7 basket (1 stock/day, highest RVOL)", MAG7)]
    for ename, e in ENTRIES:
        ox.ENTRY = dict(e, name=ename)
        ox.ENTRY_MODE = "breakout"
        for xname, x, kw in ox.EXITS:
            per = {t: ox.trades(cands[t], x, adr, datr_tab=datr, **kw) for t in TICKERS}
            for gname, members in groups:
                ts = [tr for t in members for tr in per[t]]
                for long_only, label in ((False, "Long+short"), (True, "Long only")):
                    r = {"ticker": gname, "entry": ename, "exit": xname, "strategy": label}
                    for comm, tag in ((0.0, "before"), (ox.COMMISSION_PCT, "after")):
                        taken = o.portfolio(ts, 1, long_only, news_required=False, commission_pct=comm, max_per_day=1)
                        df = pd.DataFrame(taken)
                        st = o.stats(taken, v.DAYS)
                        r[f"total_{tag}"] = st.get("total", 0)
                        for i, (s, en) in enumerate(MONTHS, 1):
                            r[f"m{i}_{tag}"] = round(df[(df.day >= s) & (df.day <= en)].pnl.sum(), 0) if len(df) else 0
                        if tag == "after":
                            r.update({"trades": st.get("trades", 0), "win_pct": st.get("win_pct"), "pf": st.get("pf"),
                                      "max_dd": st.get("max_dd"), "worst_day": st.get("worst_day")})
                    rows.append(r)
        print(f"done: {ename}", flush=True)
    pd.DataFrame(rows).to_csv(f"{o.OUT_DIR}/tickers_{o.TEST_START}_{o.TEST_END}.csv", index=False)


if __name__ == "__main__":
    main()
