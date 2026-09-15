# Market data

Your own price history, ingested from files you drop in, queried locally, and turned
into **numbers a script can cite**.

## The rule this is built around

The writing model never computes a return, a CAGR or a drawdown. It receives a fact
sheet computed here, in Python, from your data — and its only job is to arrange and
narrate those numbers.

This is not fussiness. A model that invents a percentage on a finance channel costs you
credibility you cannot buy back, and language models are particularly bad at multi-step
arithmetic over long series. So the split is hard: **arithmetic in Python, language in
the model.**

## Adding data

Export from wherever you already get it — TradingView, Yahoo Finance, investing.com, your
broker — and add the file:

```bash
reelforge market add spx.csv --symbol SPX --name "S&P 500" --currency USD
reelforge market add gold.csv --symbol GOLD --name "Gold" --currency USD
reelforge market list
```

Re-adding a file updates rather than duplicates, so refreshing is just running it again.

The reader handles what real exports actually contain, not an idealised CSV:

- comma, semicolon, tab or pipe separated
- newest-first or oldest-first rows
- `1,234.56` and `1.234,56` thousands conventions
- `(45.2)` accounting negatives, `2.3M` volume suffixes, currency symbols
- Arabic-Indic digits (`٤٥٫٥`) and Arabic headers (`التاريخ`, `الاغلاق`)
- `15/03/2024` vs `03/15/2024`, disambiguated by scanning the whole column rather
  than guessing per row

If it cannot find a date and a close column it tells you which columns it did see.

## Getting facts

```bash
reelforge market facts SPX --since 2015-01-01 --monthly 100
```

```
  S&P 500 (SPX)  2015-01-01 to 2024-12-31  ·  2609 trading days

  S&P 500 returned 173.5% between 2015-01-01 and 2024-12-31.
  That is 10.6% a year, compounded, over 10.0 years.
  1,000 USD invested at the start would be 2,735 USD at the end (2.7x).
  100 USD a month for 120 months means 12,000 USD invested, worth 21,529 USD.
  The worst fall was -33.9%, from 2020-02-19 to 2020-03-23.

  ready to say, in Arabic:
      ...
```

Every line comes in English and in Arabic, phrased to be said out loud rather than read.
`--json` gives the raw numbers for scripting.

What it computes: total return, CAGR, annualised volatility, maximum drawdown with its
peak and trough dates, per-calendar-year returns, best and worst year, a lump-sum
projection, and a monthly dollar-cost-averaging plan.

**Part-years are never quoted as a best or worst year.** If your range starts in April or
ends in July, those years are excluded from that comparison — quoting a part-year against
full years is exactly the kind of number that must not reach a script.

## Comparing

```bash
reelforge market compare SPX GOLD BTC --since 2018-01-01
```

Comparison always aligns every symbol to the window they *all* cover, so the numbers are
actually comparable. If two symbols do not overlap in time, it says so instead of
producing a meaningless answer.

## Where it lives

`.reelforge/market.db`, a plain SQLite file next to where you run the command. Your
original CSVs stay wherever you put them. Nothing is uploaded.

## What comes next

This is the first piece of the content studio. The fact sheet is the input to script
writing: you pick a topic and a framework, the script is written around *these* numbers,
and the finished script then feeds the editor — where it doubles as a transcription prior,
because knowing roughly what you said makes Arabic captions markedly more accurate.
