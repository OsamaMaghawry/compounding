"""Adds the news flag to results/orb/all_breakouts.csv.gz (EODHD news, saved in market-data and reused)."""
import pandas as pd
import orb_backtest as o
df = pd.read_csv(f"{o.OUT_DIR}/all_breakouts.csv.gz")
trades = o.add_news_flags(df.to_dict("records"), o.trading_days())
pd.DataFrame(trades).to_csv(f"{o.OUT_DIR}/all_breakouts.csv.gz", index=False)
print("NEWS STEP DONE", sum(1 for t in trades if t["news"]), "of", len(trades), "breakouts had news")
