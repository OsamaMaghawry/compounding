#!/usr/bin/env python3
"""Version A (frozen) per month with 1 stock ($10,000) or 5 stocks a day ($2,000 each, never more per stock).
The first stocks to trigger take the slots (ties: highest RVOL); at most K trades a day. TTP commission."""
import pickle
from pathlib import Path
import pandas as pd
import orb_backtest as o
import orb_variants as v
import orb_exits as ox
import orb_frozen as fz
import orb_tune as tu

MONTHS = [("2025-11-17", "2025-12-16", "tuned on"), ("2025-12-17", "2026-01-16", "tuned on"),
          ("2026-01-17", "2026-02-16", "tuned on"), ("2026-02-17", "2026-03-16", "unseen"),
          ("2026-03-17", "2026-04-16", "unseen"), ("2026-04-17", "2026-05-16", "unseen (new)")]
adr, datr = ox.adr_table(), ox.atr_table()
sim, filt, _ = fz.VERSIONS["A: long+short"]
rows, trades = [], []
for s, e, kind in MONTHS:
    f = Path(f"/tmp/claude-0/orb_days_p10_{s}_{e}.pkl")
    if not f.exists():
        o.TEST_START, o.TEST_END, o.MIN_PRICE, o.MIN_RVOL = s, e, 10.0, 2.0
        v.CACHE = f
        v.load_candidates()
    cands = pickle.load(open(f, "rb"))
    o.MIN_PRICE, o.MIN_RVOL = 10.0, 2.0
    ox.ENTRY = {"name": "stop", "or": 5, "type": "stop_order"}; ox.ENTRY_MODE = "breakout"
    for c in cands:
        c.pop("gap_thru", None)
    ts = [dict(t, news=None, t1_dist=sim["exit"][1][0] * t["adr"]) for t in ox.trades(cands, tu.exit_rule(sim), adr, datr_tab=datr)]
    ts = [t for t in ts if tu.keep(t, filt, {})]
    days = [d for d in o.trading_days() if s <= d <= e]
    for k in (1, 5):
        taken = o.portfolio(ts, k, False, news_required=False, commission_pct="TTP", max_per_day=k)
        st = o.stats(taken, days)
        df = pd.DataFrame(taken)
        rows.append({"month": f"{s} - {e}", "kind": kind, "stocks_per_day": k, "position": 10000 // k, **st,
                     "trades_per_day": round(len(df) / len(days), 2), "max_position_value": round(df.value.max(), 0),
                     "commission": round(sum(max(0.75, 0.005 * t["shares"]) * (1 + len(t["fills"])) for t in taken), 2)})
        trades += [dict(t, stocks_per_day=k) for t in taken]
    print(f"done {s}", flush=True)
pd.DataFrame(rows).to_csv(f"{o.OUT_DIR}/version_a_1_vs_5_stocks.csv", index=False)
cols = ["stocks_per_day", "day", "ticker", "side", "entry_m", "entry", "shares", "value", "exit_m", "pnl"]
pd.DataFrame(trades)[cols].to_csv(f"{o.OUT_DIR}/version_a_1_vs_5_stocks_trades.csv", index=False)
print("ALL DONE")
