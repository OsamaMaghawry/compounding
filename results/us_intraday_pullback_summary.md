# US intraday RVOL pullback strategy: backtest summary

Code: `rvol_ema_backtest.py` (EODHD 1-minute data). Capital: $10,000 (10% of a $100k account).

## Strategy as last tested (rules frozen)
- Scan: stocks in play at 9:35 ET. Relative volume ≥ 2× the 20-day average for the same window,
  price > $5, average volume > 500k shares, minimum opening gap (2%, 5% or 10%). ETFs and OTC excluded.
- Entry (PULLBACK, 5-minute bars): EMA 3 > EMA 9 (> EMA 200), buy the close back above EMA 3 after a dip.
  Confirmation buy stop 0.1 × ATR above the signal bar. Shorts are the mirror image (news catalyst required).
- Bracket at entry: target entry + 2.5 × ATR, stop entry − 0.75 × ATR. Also exits on an EMA 3/9 cross or at 15:45.
- Slots: 3, 5 or 8 equal positions. 1% daily loss limit on the 3/5-slot versions.

## Tuning period: 71 days (around Jul–Oct 2026) + Apr 21 – May 5, 11 days (the rules were tuned on the 71 days only)
| Gap | Slots | Strategy | 71 days | April | All 82 days |
|---|---|---|---|---|---|
| 2% | 3, 1% limit | Long+short | +$2,420 (PF 1.52) | +$295 | **+$2,715** |
| 2% | 3, 1% limit | Long only | +$1,293 (PF 1.34) | +$347 | +$1,640 |
| 5% | 3, 1% limit | Long+short | +$1,623 (PF 1.44) | +$705 | +$2,328 |
| 5% | 3, 1% limit | Long only | +$620 | +$722 | +$1,342 |
| 10% | 3, 1% limit | Long+short | +$1,103 (PF 1.60) | +$871 | +$1,974 |
| 10% | 3, 1% limit | Long only | +$391 | +$905 | +$1,296 |
| 2% | 8 | Long+short | +$934 | +$189 | +$1,123 |
| 2% | 8 | Long only | +$580 | +$226 | +$806 |
| 5% | 8 | Long+short | +$650 | +$322 | +$972 |
| 5% | 8 | Long only | +$394 | +$342 | +$736 |
| 10% | 8 | Long+short | +$553 | +$314 | +$867 |
| 10% | 8 | Long only | +$245 | +$326 | +$571 |

Caveats:
- A few trades made most of the profit. Without its best 10 trades, the 2%/3-slot version made +$179.
- April depended on one trade: AKAN on May 1, +$598 at its 2.5 ATR target of +18%. Without AKAN, the 2% versions lost money.
- 50+ variations were tried on the same days.

## Untested periods
| Period | Result |
|---|---|
| Jul 16–29 | −$221 (original 8-slot version) |
| Apr 21 – May 5 | positive only because of AKAN |
| **Dec 10, 2025 – Mar 18, 2026 (67 days, 18 versions, frozen rules)** | **all 18 lost** |

Dec 2025 – Mar 2026:
| Gap | Slots | Long+short | Long only |
|---|---|---|---|
| 2% | 3 (1% limit) | −$465 (PF 0.92) | −$1,125 (PF 0.78) |
| 2% | 5 (1% limit) | −$580 (PF 0.88) | −$905 (PF 0.76) |
| 2% | 8 | −$607 (PF 0.84) | −$621 (PF 0.76) |
| 5% | 3 (1% limit) | −$1,291 (PF 0.75) | −$1,197 (PF 0.70) |
| 5% | 5 (1% limit) | −$849 (PF 0.78) | −$779 (PF 0.70) |
| 5% | 8 | −$683 (PF 0.74) | −$494 (PF 0.71) |
| 10% | 3 (1% limit) | −$418 (PF 0.86) | −$620 (PF 0.70) |
| 10% | 5 (1% limit) | −$377 (PF 0.82) | −$270 (PF 0.79) |
| 10% | 8 | −$231 (PF 0.82) | −$168 (PF 0.79) |

- The win rate fell from 34% to 25%, while average wins and losses stayed the same (+2.5% / −1.05%).
- January was the worst month. Longs carried the losses.

## Conclusion
The tuning-period profits did not hold out of sample: they came from fitting those days plus a strong August.
Not tradeable with real money in this form. Paper trading only, or a new setup tested out of sample from the start.
