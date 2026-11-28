#!/usr/bin/env python3
"""Version A with 5 stocks a day, $25,000: one improvement at a time, month by month. TTP commission, saved data."""
import pickle
import pandas as pd
import orb_backtest as o
import orb_exits as ox
import orb_frozen as fz
import orb_tune as tu

MONTHS = [("2025-11-17", "2025-12-16"), ("2025-12-17", "2026-01-16"), ("2026-01-17", "2026-02-16"),
          ("2026-02-17", "2026-03-16"), ("2026-03-17", "2026-04-16"), ("2026-04-17", "2026-05-16")]
DAYS = [d for d in o.trading_days() if MONTHS[0][0] <= d <= MONTHS[-1][1]]
o.CAPITAL = 25_000
ADR, DATR = ox.adr_table(), ox.atr_table()
SIM_A, FILT_A, _ = fz.VERSIONS["A: long+short"]
CANDS = {m: pickle.load(open(f"/tmp/claude-0/orb_days_p10_{m[0]}_{m[1]}.pkl", "rb")) for m in MONTHS}


def trades(sim, filt):
    out = []
    for m in MONTHS:
        o.MIN_PRICE, o.MIN_RVOL = 10.0, 2.0
        ox.ENTRY = {"name": "stop", "or": 5, "type": "stop_order"}; ox.ENTRY_MODE = "breakout"
        for c in CANDS[m]:
            c.pop("gap_thru", None)
        ts = [dict(t, news=None, t1_dist=sim["exit"][1][0] * t["adr"]) for t in ox.trades(CANDS[m], tu.exit_rule(sim), ADR, datr_tab=DATR)]
        out += [t for t in ts if tu.keep(t, filt, {})]
    return out


def row(name, ts, **kw):
    kw.setdefault("max_per_day", 5)
    taken = o.portfolio(ts, 5, False, news_required=False, commission_pct="TTP", **kw)
    d = pd.DataFrame(taken)
    r = {"version": name}
    for i, (s, e) in enumerate(MONTHS, 1):
        r[f"m{i}"] = round(d[(d.day >= s) & (d.day <= e)].pnl.sum()) if len(d) else 0
    dl = d.groupby("day").pnl.sum().reindex(DAYS, fill_value=0); eq = dl.cumsum()
    r.update(total=round(d.pnl.sum()), pct=round(d.pnl.sum() / o.CAPITAL * 100, 1), worst_day=round(dl.min()),
             max_dd=round((eq - eq.cummax().clip(lower=0)).min()), trades=len(d), max_position=round(d.value.max()))
    r["dd_pct"] = round(r["max_dd"] / o.CAPITAL * 100, 1)
    print(r, flush=True)
    return r


base = trades(SIM_A, FILT_A)
rows = [row("Base: version A, 5 stocks", base),
        row("1. Reuse money (up to 10 trades/day)", base, max_per_day=10),
        row("1b. Reuse money (no daily limit)", base, max_per_day=0),
        row("2. #2-5 only if RVOL >= 5", base, min_rvol_after_first=5.0)]
for name, ex in [("3a. Exit: ADR 20%/40%, trail 10%", ("ADR targets 20%/40%, trail 10%", (0.20, 0.40, 0.10))),
                 ("3b. Exit: ADR 30%/60%, trail 10%", ("ADR targets 30%/60%, trail 10%", (0.30, 0.60, 0.10))),
                 ("3c. Exit: ADR 25%/50%, trail 25%", ("ADR targets 25%/50%, trail 25%", (0.25, 0.50, 0.25)))]:
    rows.append(row(name, trades(dict(SIM_A, exit=ex), FILT_A)))
rows.append(row("3d. Exit held to 15:55 (not 12:00)", trades(dict(SIM_A, time_exit=("15:55", None)), FILT_A)))
rows.append(row("4a. Filter: range <= 50% ADR", trades(SIM_A, dict(FILT_A, range_max_adr=0.5))))
rows.append(row("4b. Filter: don't skip stocks opening past the entry", trades(SIM_A, dict(FILT_A, skip_gap_thru=False))))
rows.append(row("5. (exceeds equal limit) $10,000 in #1, $3,750 in #2-5", base, weights=[0.40, 0.15, 0.15, 0.15, 0.15]))
pd.DataFrame(rows).to_csv(f"{o.OUT_DIR}/five_stocks_25k_improvements.csv", index=False)
print("ALL DONE")
