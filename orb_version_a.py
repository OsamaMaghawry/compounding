#!/usr/bin/env python3
"""Version A (frozen), full scan, on any month: python orb_version_a.py 2026-03-17 2026-04-16
5-min range, stop order at the range edge, stop at the other side, ADR targets 25%/50%, trail 10%, out at 12:00,
skip stocks whose entry bar opens past the entry level, RVOL >= 2, price >= $20, 1 stock a day, $10,000, TTP commission."""
import pickle
import sys
from pathlib import Path
import pandas as pd
import orb_backtest as o
import orb_variants as v
import orb_exits as ox
import orb_frozen as fz

s, e = sys.argv[1], sys.argv[2]
f = Path(f"/tmp/claude-0/orb_days_p10_{s}_{e}.pkl")
if not f.exists():
    o.TEST_START, o.TEST_END, o.MIN_PRICE, o.MIN_RVOL = s, e, 10.0, 2.0
    v.CACHE = f
    v.load_candidates()
cands = pickle.load(open(f, "rb"))
sim, filt, lo = fz.VERSIONS["A: long+short"]
taken = fz.run(cands, sim, filt, lo, ox.adr_table(), ox.atr_table())
days = [d for d in o.trading_days() if s <= d <= e]
st = o.stats(taken, days)
df = pd.DataFrame(taken)
df[["day", "ticker", "side", "entry_m", "entry", "shares", "value", "exit_m", "pnl"]].to_csv(
    f"{o.OUT_DIR}/version_a_trades_{s}_{e}.csv", index=False)
print(f"VERSION A {s} -> {e}: {st}")
print(f"longs {df[df.side=='long'].pnl.sum():+.0f} ({(df.side=='long').sum()}) | shorts {df[df.side=='short'].pnl.sum():+.0f} ({(df.side=='short').sum()})")
