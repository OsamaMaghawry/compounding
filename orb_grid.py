#!/usr/bin/env python3
"""1 stock per day (the first entry of the day), every entry version x every exit version, month set in orb_backtest.
Stop at the other side of the range unless the entry sets its own. Saved data and news only (no EODHD calls).
Writes results/orb/grid_1stock_<start>_<end>.csv"""
import os
import pandas as pd
import orb_backtest as o
import orb_variants as v
import orb_exits as ox

ENTRIES = [  # (name, entry rule for orb_variants.find_entries, or a pullback mode)
    ("5-min range, 1-min close beyond (volume >= range avg)", {"or": 5, "bar": 1, "vol": 1.0}, None),
    ("15-min range, 1-min close beyond", {"or": 15, "bar": 1, "vol": 1.0}, None),
    ("30-min range, 1-min close beyond", {"or": 30, "bar": 1, "vol": 1.0}, None),
    ("5-min range, 5-min candle close beyond", {"or": 5, "bar": 5, "vol": 1.0}, None),
    ("5-min range, stop order at the range edge", {"or": 5, "type": "stop_order"}, None),
    ("5-min range, breakout volume >= 2x range avg", {"or": 5, "bar": 1, "vol": 2.0}, None),
    ("5-min range, no volume check", {"or": 5, "bar": 1, "vol": 0}, None),
    ("5-min range, retest of the range edge (limit)", {"or": 5, "bar": 1, "vol": 1.0, "retest": 30}, None),
    ("5-min range, only in the gap direction", {"or": 5, "bar": 1, "vol": 1.0, "gap": True}, None),
    ("Pullback PB1: after the break, a pullback, then a break of the latest 5-min high", {"or": 5}, "pb1"),
]


def main():
    cands = v.load_candidates()
    o.NEWS_CALL_CAP = int(os.environ.get("ORB_NEWS_CAP", "0"))
    flagged = o.add_news_flags([{"ticker": c["ticker"], "day": c["day"]} for c in cands], o.trading_days())
    news = {(t["ticker"], t["day"]): t["news"] for t in flagged}
    adr, datr = ox.adr_table(), ox.atr_table()
    rows = []
    for ename, e, mode in ENTRIES:
        ox.ENTRY = dict(e, name=ename)
        ox.ENTRY_MODE = mode or "breakout"
        for xname, x, kw in ox.EXITS:
            ts = ox.trades(cands, x, adr, datr_tab=datr, **kw)
            if e.get("gap"):
                ts = [t for t in ts if (t["gap"] > 0) == (t["side"] == "long") and t["gap"] != 0]
            for t in ts:
                t["news"] = news.get((t["ticker"], t["day"]))
            for need_news in (True, False):
                for long_only, label in ((False, "Long+short"), (True, "Long only")):
                    r = {"entry": ename, "exit": xname, "news": "news" if need_news else "no news", "strategy": label}
                    for comm, tag in ((0.0, "before"), (ox.COMMISSION_PCT, "after")):
                        taken = o.portfolio(ts, 1, long_only, news_required=need_news, commission_pct=comm, max_per_day=1)
                        st = o.stats(taken, v.DAYS)
                        r[f"total_{tag}"] = st.get("total", 0)
                        if tag == "before":
                            r.update({k: st.get(k) for k in ("trades", "win_pct", "pf", "without_best5", "max_dd", "worst_day")})
                        else:
                            r["pf_after"], r["max_dd_after"] = st.get("pf"), st.get("max_dd")
                            r["long_after"], r["short_after"] = st.get("long_pnl"), st.get("short_pnl")
                    rows.append(r)
        print(f"done: {ename}", flush=True)
    pd.DataFrame(rows).to_csv(f"{o.OUT_DIR}/grid_1stock_{o.TEST_START}_{o.TEST_END}.csv", index=False)


if __name__ == "__main__":
    main()
