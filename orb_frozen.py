#!/usr/bin/env python3
"""The two frozen 5-min versions (from the tuning on Nov 17 - Feb 16) on the full scan, TQQQ, NVDA and the Mag 7 basket,
for the 3 tuning months and the next, unseen month (Feb 17 - Mar 16, 2026). $10,000, 1 trade a day, Trade The Pool
commission, saved data only. Writes results/orb/frozen_5min_versions.csv and frozen_5min_trades.csv"""
import pickle
from pathlib import Path

import pandas as pd

import orb_backtest as o
import orb_variants as v
import orb_exits as ox
import orb_tune as tu

MONTHS = tu.MONTHS + [("2026-02-17", "2026-03-16")]
MONTH_NAMES = ["Nov 17 - Dec 16", "Dec 17 - Jan 16", "Jan 17 - Feb 16", "Feb 17 - Mar 16 (unseen)"]
DAYS = [d for d in o.trading_days() if MONTHS[0][0] <= d <= MONTHS[-1][1]]
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

VERSIONS = {
    "A: long+short": ({"buffer": ("none", {}), "stop": ("other side of range", {}),
                       "exit": ("ADR targets 25%/50%, trail 10%", (0.25, 0.50, 0.10)), "time_exit": ("12:00", "12:00")},
                      dict(tu.BASE_FILT, skip_gap_thru=True, range_max_adr=None), False),
    "B: long only": ({"buffer": ("none", {}), "stop": ("other side of range", {}),
                      "exit": ("ADR targets 20%/40%, trail 25%", (0.20, 0.40, 0.25)), "time_exit": ("12:00", "12:00")},
                     dict(tu.BASE_FILT, range_max_adr=0.5), True),
}
VERSIONS["A + SPY: long+short"] = (VERSIONS["A: long+short"][0], dict(VERSIONS["A: long+short"][1], spy="SPY vs VWAP"), False)
VERSIONS["B + SPY: long only"] = (VERSIONS["B: long only"][0], dict(VERSIONS["B: long only"][1], spy="SPY vs VWAP"), True)
import os
if os.environ.get("ORB_VERSIONS"):              # e.g. "SPY" = only the versions with the SPY filter
    VERSIONS = {k: val for k, val in VERSIONS.items() if (os.environ["ORB_VERSIONS"] == "SPY") == ("SPY" in k)}
OUT_SUFFIX = os.environ.get("ORB_OUT_SUFFIX", "")


def scan_candidates():
    out = []
    for s, e in MONTHS:
        f = Path(f"/tmp/claude-0/orb_days_p10_{s}_{e}.pkl")
        if not f.exists():
            o.TEST_START, o.TEST_END, o.MIN_PRICE, o.MIN_RVOL = s, e, 10.0, 2.0
            v.CACHE = f
            v.load_candidates()
        out += pickle.load(open(f, "rb"))
    return out


def ticker_candidates():
    o.TEST_START, o.TEST_END, o.MIN_PRICE, o.MIN_RVOL = MONTHS[0][0], MONTHS[-1][1], 0.0, 0.0
    return {t: v.candidate_days(t) for t in MAG7 + ["TQQQ"]}


def run(cands, sim, filt, long_only, adr, datr, rvol_price_filters=True, spy=None):
    o.MIN_PRICE, o.MIN_RVOL = (10.0, 2.0) if rvol_price_filters else (0.0, 0.0)
    ox.ENTRY = {"name": "stop", "or": 5, "type": "stop_order"}
    ox.ENTRY_MODE = "breakout"
    for c in cands:
        c.pop("gap_thru", None)
    ts = ox.trades(cands, tu.exit_rule(sim), adr, datr_tab=datr)
    f = dict(filt) if rvol_price_filters else dict(filt, rvol_min=0.0, price_min=0.0)
    ts = [dict(t, news=None, t1_dist=sim["exit"][1][0] * t["adr"]) for t in ts]
    ts = [t for t in ts if tu.keep(t, f, spy or {})]
    return o.portfolio(ts, 1, long_only, news_required=False, commission_pct="TTP", max_per_day=1)


def main():
    adr, datr = ox.adr_table(), ox.atr_table()
    o.TEST_START, o.TEST_END = MONTHS[0][0], MONTHS[-1][1]   # load SPY bars for the whole period (all saved windows)
    spy = tu.spy_state()
    missing = [d for d in DAYS if d not in spy]
    print(f"SPY days missing: {missing}", flush=True)
    scan = scan_candidates()
    tick = ticker_candidates()
    groups = {"Full scan (1 stock/day)": (scan, True), "TQQQ": (tick["TQQQ"], False), "NVDA": (tick["NVDA"], False),
              "Mag 7 basket (1 stock/day, highest RVOL)": ([c for t in MAG7 for c in tick[t]], False)}
    rows, all_trades = [], []
    for vname, (sim, filt, lo) in VERSIONS.items():
        for gname, (cands, rp) in groups.items():
            taken = run(cands, sim, filt, lo, adr, datr, rvol_price_filters=rp, spy=spy)
            df = pd.DataFrame(taken)
            r = {"version": vname, "group": gname}
            for (s, e), mn in zip(MONTHS, MONTH_NAMES):
                m = df[(df.day >= s) & (df.day <= e)] if len(df) else df
                st = o.stats(m.to_dict("records"), [d for d in DAYS if s <= d <= e])
                r[f"{mn} total"], r[f"{mn} trades"], r[f"{mn} win%"], r[f"{mn} max_dd"] = \
                    st.get("total", 0), st.get("trades", 0), st.get("win_pct"), st.get("max_dd")
            first3 = df[df.day <= MONTHS[2][1]] if len(df) else df
            st3 = o.stats(first3.to_dict("records"), [d for d in DAYS if d <= MONTHS[2][1]])
            r["3 tuning months total"], r["3 tuning months max_dd"] = st3.get("total", 0), st3.get("max_dd")
            rows.append(r)
            all_trades += [dict(t, version=vname, group=gname) for t in taken]
            print(f"{vname} | {gname}: done", flush=True)
    pd.DataFrame(rows).to_csv(f"{o.OUT_DIR}/frozen_5min_versions{OUT_SUFFIX}.csv", index=False)
    keep_cols = ["version", "group", "day", "ticker", "side", "entry_m", "entry", "shares", "value", "exit_m", "pnl_ps", "pnl"]
    pd.DataFrame(all_trades)[keep_cols].to_csv(f"{o.OUT_DIR}/frozen_5min_trades{OUT_SUFFIX}.csv", index=False)
    print("ALL DONE")


if __name__ == "__main__":
    main()
