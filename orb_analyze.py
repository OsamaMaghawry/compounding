#!/usr/bin/env python3
"""Portfolio results for the ORB backtest: every version as long+short and long only, by period and month.
Reads results/orb/all_breakouts.csv.gz (written by orb_backtest.py) and writes results/orb/summary.md + .csv."""
import pandas as pd

import orb_backtest as o

PERIODS = [(o.TEST_START, o.TEST_END)]

trades = pd.read_csv(f"{o.OUT_DIR}/all_breakouts.csv.gz").to_dict("records")
days = [d for d in o.trading_days() if o.TEST_START <= d <= o.TEST_END]

VERSIONS = [  # (name, slots, kwargs)
    ("Your rules, 3 slots", 3, {}),
    ("Your rules, 5 slots", 5, {}),
    ("Your rules, 8 slots", 8, {}),
    ("5 slots, NO news filter", 5, {"news_required": False}),
    ("5 slots + gap in trade direction", 5, {"keep": lambda t: (t["gap"] > 0) if t["side"] == "long" else (t["gap"] < 0)}),
    ("5 slots + no chase (<= 0.25 R past range)", 5, {"keep": lambda t: t["chase_r"] <= 0.25}),
    ("5 slots + entries by 10:30", 5, {"keep": lambda t: t["entry_m"] <= 60}),
    ("5 slots + range >= 0.5% of price", 5, {"keep": lambda t: t["range_pct"] >= 0.5}),
    ("5 slots + risk sizing 0.5%", 5, {"sizing": "RISK"}),
    ("5 slots + 1% daily loss limit", 5, {"daily_limit_pct": 1.0}),
]

rows = []
for name, slots, kw in VERSIONS:
    for long_only in (False, True):
        args = {"news_required": True, **kw}
        taken = o.portfolio(trades, slots, long_only, **args)
        st = o.stats(taken, days)
        df = pd.DataFrame(taken)
        for i, (s, e) in enumerate(PERIODS, 1):
            st[f"P{i}"] = round(df[(df.day >= s) & (df.day <= e)].pnl.sum(), 0) if len(df) else 0
        if len(df):
            for mth, v in df.groupby(df.day.str[:7]).pnl.sum().items():
                st[mth] = round(v, 0)
        rows.append({"version": name, "strategy": "Long only" if long_only else "Long+short", **st})

res = pd.DataFrame(rows)
res.to_csv(f"{o.OUT_DIR}/summary.csv", index=False)
cols = ["version", "strategy", "trades", "win_pct", "pf", "total", "without_best5", "avg_r", "long_pnl", "short_pnl",
        "worst_day", "max_dd", "days_up", "days_down"]
months = sorted(c for c in res.columns if c[:2] == "20" and len(c) == 7)
with open(f"{o.OUT_DIR}/summary.md", "w") as f:
    f.write(f"# ORB backtest - {days[0]} to {days[-1]} ({len(days)} days), ${o.CAPITAL:,} capital\n\n")
    f.write(res[cols].to_markdown(index=False) + "\n\n## Monthly P&L\n\n")
    f.write(res[["version", "strategy"] + months].to_markdown(index=False) + "\n")
print(res[cols].to_string(index=False))
