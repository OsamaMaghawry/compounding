#!/usr/bin/env python3
"""
Tuning ONE setup: opening range (5 or 15 min), stop order at the range edge, 1 stock a day, $10,000 per trade.
Starts from a base version and changes one thing at a time (entry, exit, stop, reward/risk, filters), then combines
the best settings. 3 months, saved data only, Trade The Pool commission. Writes results/orb/tune_<step>.csv
"""
import itertools
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import orb_backtest as o
import orb_variants as v
import orb_exits as ox
from orb_risk import spy_state

MONTHS = [("2025-11-17", "2025-12-16"), ("2025-12-17", "2026-01-16"), ("2026-01-17", "2026-02-16")]
DAYS = [d for d in o.trading_days() if MONTHS[0][0] <= d <= MONTHS[-1][1]]

BASE_SIM = {"buffer": ("none", {}), "stop": ("other side of range", {}),
            "exit": ("ADR targets 25%/50%, trail 15%", (0.25, 0.50, 0.15)), "time_exit": ("15:55", None)}
BASE_FILT = {"last_entry": "11:30", "skip_gap_thru": False, "rr_min": 0.0, "range_max_adr": 0.5,
             "rvol_min": 2.0, "spy": "none", "price_min": 20.0}

SIM_OPTIONS = {
    "buffer": [("none", {}), ("+0.05% beyond the edge", {"buffer_pct": 0.05}), ("+10% of the range", {"buffer_r": 0.10})],
    "stop": [("other side of range", {}), ("1/4 into the range", {"stop_frac": 0.25}),
             ("0.10 x daily ATR", {"stop_atr": 0.10}), ("0.15 x daily ATR", {"stop_atr": 0.15}),
             ("0.20 x daily ATR", {"stop_atr": 0.20})],
    "exit": [(f"ADR targets {int(a*100)}%/{int(b*100)}%, trail {int(t*100)}%", (a, b, t))
             for (a, b), t in itertools.product([(0.20, 0.40), (0.25, 0.50), (0.30, 0.60)], [0.10, 0.15, 0.25])],
    "time_exit": [("15:55", None), ("12:00", "12:00")],
}
FILT_OPTIONS = {
    "last_entry": ["10:00", "10:30", "11:30"],
    "skip_gap_thru": [False, True],
    "rr_min": [0.0, 1.0, 1.5, 2.0],
    "range_max_adr": [None, 0.33, 0.50, 0.75],
    "rvol_min": [2.0, 3.0, 5.0],
    "spy": ["none", "SPY vs VWAP"],
    "price_min": [20.0, 10.0],
}


def exit_rule(sim):
    a, b, t = sim["exit"][1]
    x = {"be": 1.0, "targets": [(a, .5, "ADR"), (b, .25, "ADR")], "trail": [{"type": "ADR", "k": t, "start": "t2"}]}
    x.update(sim["stop"][1])
    if sim["time_exit"][1]:
        x["exit_at_m"] = o.hm(sim["time_exit"][1])
    return x


CANDS = {}


def candidates(month):
    if month not in CANDS:
        s, e = month
        f = Path(f"/tmp/claude-0/orb_days_p10_{s}_{e}.pkl")
        if not f.exists():
            o.TEST_START, o.TEST_END, o.MIN_PRICE = s, e, 10.0
            v.CACHE = f
        CANDS[month] = v.load_candidates() if not f.exists() else pickle.load(open(f, "rb"))
    return CANDS[month]


SIM_CACHE = {}


def simulate(or_len, sim, adr, datr):
    key = (or_len, sim["buffer"][0], sim["stop"][0], sim["exit"][0], sim["time_exit"][0])
    if key not in SIM_CACHE:
        o.MIN_PRICE, o.MIN_RVOL = 10.0, 2.0
        ox.ENTRY = {"name": "stop", "or": or_len, "type": "stop_order", **sim["buffer"][1]}
        ox.ENTRY_MODE = "breakout"
        x = exit_rule(sim)
        ts = []
        for mth in MONTHS:
            for c in candidates(mth):
                c.pop("gap_thru", None)
            ts += ox.trades(candidates(mth), x, adr, datr_tab=datr)
        for t in ts:
            t["news"] = None
            t["t1_dist"] = sim["exit"][1][0] * t["adr"]
        SIM_CACHE[key] = ts
    return SIM_CACHE[key]


def keep(t, f, spy):
    s = 1 if t["side"] == "long" else -1
    if t["entry_m"] > o.hm(f["last_entry"]):
        return False
    if f["skip_gap_thru"] and t["gap_thru"]:
        return False
    if f["rr_min"] and t["t1_dist"] < f["rr_min"] * t["risk"]:
        return False
    if f["range_max_adr"] and t["R"] > f["range_max_adr"] * t["adr"]:
        return False
    if t["rvol"] < f["rvol_min"]:
        return False
    if (t.get("price5") or t["entry"]) < f["price_min"]:
        return False
    if f["spy"] != "none":
        st = spy.get(t["day"])
        if st is None:
            return False
        i = np.searchsorted(st["m"], t["entry_m"]) - 1
        if i < 0 or (st["c"][i] - st["vwap"][i]) * s <= 0:
            return False
    return True


def evaluate(or_len, sim, filt, adr, datr, spy, tag):
    ts = [t for t in simulate(or_len, sim, adr, datr) if keep(t, filt, spy)]
    rows = []
    for lo, label in ((False, "Long+short"), (True, "Long only")):
        taken = o.portfolio(ts, 1, lo, news_required=False, commission_pct="TTP", max_per_day=1)
        st = o.stats(taken, DAYS)
        df = pd.DataFrame(taken)
        r = {"range": f"{or_len}-min", "strategy": label, "changed": tag,
             **{k: val[0] for k, val in sim.items()}, **filt, **st}
        for i, (ms, me) in enumerate(MONTHS, 1):
            r[f"m{i}"] = round(df[(df.day >= ms) & (df.day <= me)].pnl.sum(), 0) if len(df) else 0
        rows.append(r)
    return rows


def main():
    spy = spy_state()
    adr, datr = ox.adr_table(), ox.atr_table()
    for mth in MONTHS:
        print(f"candidates {mth}: {len(candidates(mth))}", flush=True)
    rows = []
    step1_file = Path(f"{o.OUT_DIR}/tune_step1_one_change.csv")
    # step 1: one change at a time from the base (skipped if already saved)
    for or_len in (() if step1_file.exists() else (5, 15)):
        rows += evaluate(or_len, BASE_SIM, BASE_FILT, adr, datr, spy, "base")
        for k, opts in SIM_OPTIONS.items():
            for opt in opts:
                if opt[0] == BASE_SIM[k][0]:
                    continue
                rows += evaluate(or_len, dict(BASE_SIM, **{k: opt}), BASE_FILT, adr, datr, spy, f"{k}: {opt[0]}")
        for k, opts in FILT_OPTIONS.items():
            for opt in opts:
                if opt == BASE_FILT[k]:
                    continue
                rows += evaluate(or_len, BASE_SIM, dict(BASE_FILT, **{k: opt}), adr, datr, spy, f"{k}: {opt}")
        print(f"step 1 done: {or_len}-min", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(step1_file, index=False)
    step1 = pd.read_csv(step1_file)

    # step 2: combine the 2 best options of each setting (by 3-month total / drawdown), per range and strategy
    for (rng, strat), g in step1.groupby(["range", "strategy"]):
        or_len = int(rng.split("-")[0])
        part = Path(f"{o.OUT_DIR}/tune_step2_{rng}_{strat.replace('+', '_').replace(' ', '_')}.csv")
        if part.exists():                       # saved part: not run again
            continue
        rows2 = []
        base = g[g.changed == "base"].iloc[0]
        score = lambda r: r["total"] / max(1.0, -r["max_dd"])
        best = {}
        for k in list(SIM_OPTIONS) + list(FILT_OPTIONS):
            cand = [(score(base), BASE_SIM[k][0] if k in SIM_OPTIONS else BASE_FILT[k])]
            for _, r in g[g.changed.str.startswith(k + ":")].iterrows():
                cand.append((score(r), r[k]))
            cand.sort(key=lambda z: -z[0])
            best[k] = list(dict.fromkeys(c[1] for c in cand))[:2]
        lo = strat == "Long only"
        sim_keys = list(SIM_OPTIONS)
        filt_keys = list(FILT_OPTIONS)
        for combo in itertools.product(*[best[k] for k in sim_keys + filt_keys]):
            sim = {k: next(o_ for o_ in SIM_OPTIONS[k] if o_[0] == combo[i]) for i, k in enumerate(sim_keys)}
            filt = {k: combo[len(sim_keys) + i] for i, k in enumerate(filt_keys)}
            for r in evaluate(or_len, sim, filt, adr, datr, spy, "combination"):
                if (r["strategy"] == "Long only") == lo:
                    rows2.append(r)
        pd.DataFrame(rows2).to_csv(part, index=False)
        os.system(f'git add "{part}" && git commit -qm "ORB tuning step 2: {rng} {strat}" '
                  f'-m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" '
                  f'-m "Claude-Session: https://claude.ai/code/session_01UMT8os9rEpqA4E8Br1QE99" && git push -q')
        print(f"step 2 done and saved: {rng} {strat} ({len(rows2)} rows)", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
