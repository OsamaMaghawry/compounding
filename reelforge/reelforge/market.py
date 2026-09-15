"""Your own market data, ingested from files you drop in and queried locally.

The point of this module is not charts. It is to produce **verified numbers** that a
script can cite. The writing model never computes a return, a CAGR or a drawdown -
it receives a fact sheet built here and writes prose around it. A model that invents
a percentage on a finance channel costs you credibility you cannot buy back, so the
split is deliberate: arithmetic happens in Python, language happens in the model.

Everything is stdlib. Exports from TradingView, Yahoo Finance, investing.com and
plain broker CSVs all land in the same normalised store.
"""

from __future__ import annotations

import csv
import math
import re
import sqlite3
import statistics
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    currency TEXT,
    source TEXT,
    added TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    day TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL NOT NULL, volume REAL,
    PRIMARY KEY (symbol, day)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_day ON bars(symbol, day);
"""

# Column aliases seen in real exports, English and Arabic.
COLUMN_ALIASES = {
    "day": ["date", "day", "time", "timestamp", "datetime", "تاريخ", "التاريخ"],
    "close": ["close", "close/last", "adj close", "adj_close", "adjclose", "price",
              "last", "closing price", "اغلاق", "الاغلاق", "السعر", "سعر الاغلاق"],
    "open": ["open", "opening price", "افتتاح", "الافتتاح"],
    "high": ["high", "max", "اعلى", "الاعلى"],
    "low": ["low", "min", "ادنى", "الادنى"],
    "volume": ["volume", "vol", "vol.", "الحجم", "حجم"],
}

DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
                "%d.%m.%Y", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%Y%m%d"]
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
SUFFIXES = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}
TRADING_DAYS = 252


class MarketError(RuntimeError):
    pass


# ------------------------------------------------------------------ parsing

def parse_number(raw: str | float | int | None) -> float | None:
    """Parse a number out of whatever a spreadsheet export threw at us."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().translate(ARABIC_DIGITS)
    if not text or text in {"-", "--", "n/a", "N/A", "null", "."}:
        return None

    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = text.replace("٫", ".").replace("٬", ",")   # Arabic decimal/thousands
    text = re.sub(r"[^\d.,\-+eE]", "", text.replace("%", ""))
    if not text:
        return None

    multiplier = 1.0
    tail = str(raw).strip()[-1:].lower()
    if tail in SUFFIXES:
        multiplier = SUFFIXES[tail]

    # Decide which separator is the decimal point.
    if "," in text and "." in text:
        text = text.replace(",", "") if text.rfind(".") > text.rfind(",") \
            else text.replace(".", "").replace(",", ".")
    elif "," in text:
        parts = text.split(",")
        # "1,234" is thousands; "1,23" is a decimal comma.
        text = text.replace(",", "") if len(parts[-1]) == 3 and len(parts) > 1 \
            else text.replace(",", ".")
    try:
        value = float(text) * multiplier
    except ValueError:
        return None
    return -value if negative else value


def _day_first(samples: list[str]) -> bool:
    """Tell 12/03/2024 apart from 03/12/2024 by looking at the whole column."""
    for sample in samples:
        parts = re.split(r"[/\-.]", sample.strip())
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            first, second = int(parts[0]), int(parts[1])
            if first > 12 and second <= 12:
                return True
            if second > 12 and first <= 12:
                return False
    return False   # ambiguous everywhere - assume US-style, the commoner export


def parse_date(raw: str, *, day_first: bool = False) -> date | None:
    text = (raw or "").strip().translate(ARABIC_DIGITS)
    if not text:
        return None
    text = text.split("T")[0].split(" ")[0] if re.match(r"^\d{4}-\d{2}-\d{2}", text) else text
    formats = list(DATE_FORMATS)
    if day_first:
        formats.remove("%d/%m/%Y")
        formats.remove("%d-%m-%Y")
        formats.insert(0, "%d/%m/%Y")
        formats.insert(1, "%d-%m-%Y")
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _match_column(header: list[str]) -> dict[str, int]:
    """Map our field names onto the columns actually present."""
    normalised = [(h or "").strip().strip('"').lower() for h in header]
    mapping: dict[str, int] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for index, name in enumerate(normalised):
            if name in aliases:
                mapping[field] = index
                break
        if field not in mapping:                      # fall back to a loose match
            for index, name in enumerate(normalised):
                if any(alias in name for alias in aliases if len(alias) > 3):
                    mapping[field] = index
                    break
    return mapping


@dataclass
class Bar:
    day: date
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None


def read_csv(path: str | Path) -> list[Bar]:
    """Read a price export into bars, tolerating the usual export quirks."""
    path = Path(path)
    if not path.exists():
        raise MarketError(f"file not found: {path}")

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if len(rows) < 2:
        raise MarketError(f"{path.name}: not enough rows to read")

    mapping = _match_column(rows[0])
    if "day" not in mapping or "close" not in mapping:
        raise MarketError(
            f"{path.name}: could not find a date and a close column.\n"
            f"  saw: {', '.join(c for c in rows[0] if c)[:160]}\n"
            f"  expected a header with something like Date and Close."
        )

    body = rows[1:]
    date_samples = [r[mapping["day"]] for r in body[:200] if len(r) > mapping["day"]]
    day_first = _day_first(date_samples)

    bars: list[Bar] = []
    for row in body:
        if len(row) <= max(mapping.values()):
            continue
        when = parse_date(row[mapping["day"]], day_first=day_first)
        close = parse_number(row[mapping["close"]])
        if when is None or close is None or close <= 0:
            continue
        bars.append(Bar(
            day=when, close=close,
            open=parse_number(row[mapping["open"]]) if "open" in mapping else None,
            high=parse_number(row[mapping["high"]]) if "high" in mapping else None,
            low=parse_number(row[mapping["low"]]) if "low" in mapping else None,
            volume=parse_number(row[mapping["volume"]]) if "volume" in mapping else None,
        ))

    if not bars:
        raise MarketError(f"{path.name}: no usable rows (check the date and price columns)")
    bars.sort(key=lambda b: b.day)          # exports are often newest-first
    return bars


# -------------------------------------------------------------------- store

class MarketStore:
    """Local price history you own, in one SQLite file."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "market.db"
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- ingest ----------------------------------------------------------
    def add_csv(self, path: str | Path, symbol: str, *, name: str = "",
                currency: str = "", note: str = "") -> dict:
        bars = read_csv(path)
        symbol = symbol.strip().upper()
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO series (symbol, name, currency, source, added, note) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET "
                "name=COALESCE(NULLIF(excluded.name,''), name), "
                "currency=COALESCE(NULLIF(excluded.currency,''), currency), "
                "source=excluded.source, added=excluded.added",
                (symbol, name or symbol, currency, str(Path(path).name),
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), note),
            )
            conn.executemany(
                "INSERT INTO bars (symbol, day, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(symbol, day) DO UPDATE SET "
                "open=excluded.open, high=excluded.high, low=excluded.low, "
                "close=excluded.close, volume=excluded.volume",
                [(symbol, b.day.isoformat(), b.open, b.high, b.low, b.close, b.volume)
                 for b in bars],
            )
            conn.commit()
        return {"symbol": symbol, "rows": len(bars),
                "start": bars[0].day.isoformat(), "end": bars[-1].day.isoformat()}

    # -- read ------------------------------------------------------------
    def symbols(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT s.symbol, s.name, s.currency, s.source, COUNT(b.day) AS rows, "
                "MIN(b.day) AS start, MAX(b.day) AS end "
                "FROM series s LEFT JOIN bars b ON b.symbol = s.symbol "
                "GROUP BY s.symbol ORDER BY s.symbol"
            ).fetchall()
        return [dict(r) for r in rows]

    def bars(self, symbol: str, start: str | None = None,
             end: str | None = None) -> list[Bar]:
        query = "SELECT * FROM bars WHERE symbol = ?"
        params: list = [symbol.upper()]
        if start:
            query += " AND day >= ?"
            params.append(start)
        if end:
            query += " AND day <= ?"
            params.append(end)
        query += " ORDER BY day"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [Bar(day=date.fromisoformat(r["day"]), close=r["close"], open=r["open"],
                    high=r["high"], low=r["low"], volume=r["volume"]) for r in rows]

    def has(self, symbol: str) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT 1 FROM series WHERE symbol = ?",
                               (symbol.upper(),)).fetchone()
        return row is not None

    def remove(self, symbol: str) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute("DELETE FROM bars WHERE symbol = ?", (symbol.upper(),))
            conn.execute("DELETE FROM series WHERE symbol = ?", (symbol.upper(),))
            conn.commit()
            return cursor.rowcount

    def price_on(self, symbol: str, when: str) -> Bar | None:
        """Close on that day, or the last trading day before it."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM bars WHERE symbol = ? AND day <= ? ORDER BY day DESC LIMIT 1",
                (symbol.upper(), when),
            ).fetchone()
        if not row:
            return None
        return Bar(day=date.fromisoformat(row["day"]), close=row["close"], open=row["open"],
                   high=row["high"], low=row["low"], volume=row["volume"])


# ---------------------------------------------------------------- analytics

def total_return(bars: list[Bar]) -> float | None:
    """Simple return over the whole range, as a fraction."""
    if len(bars) < 2 or bars[0].close <= 0:
        return None
    return bars[-1].close / bars[0].close - 1.0


def years_between(first: date, last: date) -> float:
    return max(0.0, (last - first).days / 365.25)


def cagr(bars: list[Bar]) -> float | None:
    """Compound annual growth rate - the number this whole channel is about."""
    if len(bars) < 2 or bars[0].close <= 0:
        return None
    span = years_between(bars[0].day, bars[-1].day)
    if span < 0.08:                      # under a month, annualising is meaningless
        return None
    return (bars[-1].close / bars[0].close) ** (1.0 / span) - 1.0


def max_drawdown(bars: list[Bar]) -> dict | None:
    """Worst peak-to-trough fall, and when it bottomed."""
    if len(bars) < 2:
        return None
    peak = bars[0].close
    peak_day = bars[0].day
    worst = 0.0
    result = None
    for bar in bars:
        if bar.close > peak:
            peak, peak_day = bar.close, bar.day
        if peak > 0:
            fall = bar.close / peak - 1.0
            if fall < worst:
                worst = fall
                result = {"drawdown": fall, "peak_day": peak_day.isoformat(),
                          "trough_day": bar.day.isoformat(),
                          "peak": peak, "trough": bar.close}
    return result


def volatility(bars: list[Bar]) -> float | None:
    """Annualised standard deviation of daily returns."""
    if len(bars) < 20:
        return None
    returns = []
    for previous, current in zip(bars, bars[1:]):
        if previous.close > 0:
            returns.append(math.log(current.close / previous.close))
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * math.sqrt(TRADING_DAYS)


def calendar_years(bars: list[Bar]) -> dict[int, float]:
    """Return per calendar year, using the first and last close in each."""
    by_year: dict[int, list[Bar]] = {}
    for bar in bars:
        by_year.setdefault(bar.day.year, []).append(bar)
    out: dict[int, float] = {}
    previous_close: float | None = None
    for year in sorted(by_year):
        rows = by_year[year]
        opening = previous_close if previous_close is not None else rows[0].close
        if opening > 0:
            out[year] = rows[-1].close / opening - 1.0
        previous_close = rows[-1].close
    return out


def invest_lump(bars: list[Bar], amount: float) -> dict | None:
    """What a single investment at the start would be worth at the end."""
    if len(bars) < 2 or bars[0].close <= 0:
        return None
    units = amount / bars[0].close
    final = units * bars[-1].close
    return {"invested": amount, "value": final, "profit": final - amount,
            "multiple": final / amount if amount else None,
            "return": final / amount - 1.0 if amount else None}


def invest_monthly(bars: list[Bar], amount: float) -> dict | None:
    """Dollar-cost averaging: buy `amount` on the first trading day of each month."""
    if len(bars) < 2:
        return None
    units = 0.0
    invested = 0.0
    seen: set[tuple[int, int]] = set()
    for bar in bars:
        key = (bar.day.year, bar.day.month)
        if key in seen or bar.close <= 0:
            continue
        seen.add(key)
        units += amount / bar.close
        invested += amount
    if invested <= 0:
        return None
    final = units * bars[-1].close
    return {"invested": invested, "value": final, "profit": final - invested,
            "months": len(seen), "monthly": amount,
            "multiple": final / invested, "return": final / invested - 1.0}


def _money(value: float, currency: str = "") -> str:
    unit = f" {currency}" if currency else ""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M{unit}"
    return f"{value:,.0f}{unit}" if abs(value) >= 100 else f"{value:,.2f}{unit}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:,.1f}%"


def fact_sheet(store: "MarketStore", symbol: str, *, start: str | None = None,
               end: str | None = None, amount: float = 1000.0,
               monthly: float | None = None) -> dict:
    """Every number a script might cite, computed here so the model never guesses one."""
    symbol = symbol.upper()
    if not store.has(symbol):
        raise MarketError(f"no data for {symbol}. Add some with: reelforge market add file.csv "
                          f"--symbol {symbol}")
    bars = store.bars(symbol, start, end)
    if len(bars) < 2:
        raise MarketError(f"{symbol}: not enough data in that range")

    meta = next((s for s in store.symbols() if s["symbol"] == symbol), {})
    currency = meta.get("currency") or ""
    span = years_between(bars[0].day, bars[-1].day)
    years = calendar_years(bars)
    # Only years the data actually covers end to end can be called a "best" or
    # "worst" year. A part-year at either edge would be quoted as if it were a full
    # one, which is exactly the kind of number that must never reach a script.
    first_year, last_year = bars[0].day.year, bars[-1].day.year
    complete_years = {
        year: value for year, value in years.items()
        if not (year == first_year and bars[0].day.month > 1)
        and not (year == last_year and bars[-1].day.month < 12)
    }
    drawdown = max_drawdown(bars)
    lump = invest_lump(bars, amount)
    dca = invest_monthly(bars, monthly) if monthly else None

    facts = {
        "symbol": symbol,
        "name": meta.get("name") or symbol,
        "currency": currency,
        "start_day": bars[0].day.isoformat(),
        "end_day": bars[-1].day.isoformat(),
        "years": round(span, 2),
        "start_price": bars[0].close,
        "end_price": bars[-1].close,
        "total_return": total_return(bars),
        "cagr": cagr(bars),
        "max_drawdown": drawdown,
        "volatility": volatility(bars),
        "calendar_years": {str(y): round(r, 4) for y, r in years.items()},
        "best_year": max(complete_years.items(), key=lambda kv: kv[1]) if complete_years else None,
        "worst_year": min(complete_years.items(), key=lambda kv: kv[1]) if complete_years else None,
        "lump_sum": lump,
        "monthly_plan": dca,
        "bars": len(bars),
    }

    # Pre-written, citable statements. A script can use these verbatim; the model's
    # job is to arrange and narrate them, never to recompute them.
    english: list[str] = []
    arabic: list[str] = []
    name = facts["name"]

    if facts["total_return"] is not None:
        english.append(f"{name} returned {_pct(facts['total_return'])} between "
                       f"{facts['start_day']} and {facts['end_day']}.")
        arabic.append(f"{name} حقق {_pct(facts['total_return'])} من {facts['start_day']} "
                      f"إلى {facts['end_day']}.")
    if facts["cagr"] is not None:
        english.append(f"That is {_pct(facts['cagr'])} a year, compounded, over "
                       f"{span:.1f} years.")
        arabic.append(f"يعني {_pct(facts['cagr'])} سنويا بالتراكم على مدى "
                      f"{span:.1f} سنة.")
    if lump:
        english.append(f"{_money(amount, currency)} invested at the start would be "
                       f"{_money(lump['value'], currency)} at the end "
                       f"({lump['multiple']:.1f}x).")
        arabic.append(f"لو استثمرت {_money(amount, currency)} في البداية كانت بقت "
                      f"{_money(lump['value'], currency)} ({lump['multiple']:.1f} ضعف).")
    if dca:
        english.append(f"{_money(dca['monthly'], currency)} a month for {dca['months']} months "
                       f"means {_money(dca['invested'], currency)} invested, worth "
                       f"{_money(dca['value'], currency)} ({_pct(dca['return'])}).")
        arabic.append(f"{_money(dca['monthly'], currency)} شهريا لمدة {dca['months']} شهر "
                      f"يعني {_money(dca['invested'], currency)} استثمرتها، قيمتها "
                      f"{_money(dca['value'], currency)} ({_pct(dca['return'])}).")
    if drawdown:
        english.append(f"The worst fall was {_pct(drawdown['drawdown'])}, from "
                       f"{drawdown['peak_day']} to {drawdown['trough_day']}.")
        arabic.append(f"أسوأ هبوط كان {_pct(drawdown['drawdown'])}، من "
                      f"{drawdown['peak_day']} إلى {drawdown['trough_day']}.")
    if facts["best_year"]:
        year, value = facts["best_year"]
        english.append(f"Best year was {year} at {_pct(value)}.")
        arabic.append(f"أفضل سنة كانت {year} بنسبة {_pct(value)}.")
    if facts["worst_year"]:
        year, value = facts["worst_year"]
        english.append(f"Worst year was {year} at {_pct(value)}.")
        arabic.append(f"أسوأ سنة كانت {year} بنسبة {_pct(value)}.")

    facts["statements"] = {"en": english, "ar": arabic}
    return facts


def compare(store: "MarketStore", symbols: list[str], *, start: str | None = None,
            end: str | None = None) -> dict:
    """Same-window comparison, so the numbers are actually comparable."""
    windows = []
    for symbol in symbols:
        bars = store.bars(symbol.upper(), start, end)
        if len(bars) >= 2:
            windows.append((symbol.upper(), bars))
    if len(windows) < 2:
        raise MarketError("need at least two symbols with data in that range")

    # Align to the widest window every symbol actually covers.
    aligned_start = max(bars[0].day for _, bars in windows)
    aligned_end = min(bars[-1].day for _, bars in windows)
    if aligned_start >= aligned_end:
        raise MarketError("those symbols do not overlap in time")

    rows = []
    for symbol, _ in windows:
        bars = store.bars(symbol, aligned_start.isoformat(), aligned_end.isoformat())
        drawdown = max_drawdown(bars)
        rows.append({
            "symbol": symbol,
            "total_return": total_return(bars),
            "cagr": cagr(bars),
            "max_drawdown": drawdown["drawdown"] if drawdown else None,
            "volatility": volatility(bars),
        })
    rows.sort(key=lambda r: (r["cagr"] is None, -(r["cagr"] or 0)))
    return {"start_day": aligned_start.isoformat(), "end_day": aligned_end.isoformat(),
            "years": round(years_between(aligned_start, aligned_end), 2), "rows": rows}
