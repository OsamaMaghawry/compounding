#!/usr/bin/env python3
"""
EGX Seasonality Live Scanner - Fully Automatic Cycle Rollover
- Run it once a day (e.g. a PythonAnywhere scheduled task). No manual steps needed.
- First run (no saved cycle): starts a cycle on the latest EGX trading day.
- Every run while the cycle is active: refreshes live prices and emails the tracker.
- When HOLDING_DAYS trading days have passed: closes the old cycle (archived to
  cycle_history.json), scans a fresh portfolio on the latest trading day and
  emails it as a NEW CYCLE.
- Optional: CYCLE_ENTRY_DATE starts one cycle on a specific date (used once only).
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import io
import base64
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATIONS - EDIT THESE VALUES ONCE
# ============================================

# EODHD API Configuration
EODHD_API_KEY = os.environ.get("EODHD_API_KEY", "PUT_YOUR_EODHD_API_KEY_HERE")
EODHD_BASE_URL = "https://eodhd.com/api"

# Strategy Parameters
NUM_STOCKS = 10
HOLDING_DAYS = 21
STOP_LOSS_PCT = 5.0

# ENTRY DATE - OPTIONAL, normally leave as None.
# None  -> fully automatic: a new cycle starts by itself every HOLDING_DAYS trading days.
# "YYYY-MM-DD" -> start ONE cycle on that date (e.g. to back-date a cycle). It is applied
#                 once and remembered in cycle_state.json, so you can leave it here and
#                 the following cycles will still roll over automatically.
CYCLE_ENTRY_DATE = None

# EGX trading week (used only as a fallback when the benchmark calendar is unavailable)
EGX_WEEKMASK = "Sun Mon Tue Wed Thu"

# Hypothetical Portfolio Size (EGP)
HYPOTHETICAL_PORTFOLIO_SIZE = 100_000

# Filter Parameters
MIN_CORRELATION = 0.4
MIN_HISTORY_DAYS = 252
MIN_LIQUIDITY = 2_000_000
MAX_RETURN_THRESHOLD = 100

# Month Filters
MONTHS_TO_SKIP = {
    'January': False, 'February': False, 'March': False,
    'April': False, 'May': False, 'June': False,
    'July': False, 'August': False, 'September': False,
    'October': False, 'November': False, 'December': False,
}

RANKING_METHOD = "opportunity_score"

YEAR_WEIGHTS = [
    (3, 0.30), (5, 0.20), (7, 0.15), (10, 0.15), (15, 0.10), (20, 0.10),
]

BATCH_SIZE = 5
BATCH_DELAY = 0.3
MAX_RETRIES = 3

HTML_OUTPUT = "live_scan_report.html"
PDF_OUTPUT = "live_scan_report.pdf"
CHART_PATH = "expected_returns_chart.png"
CYCLE_STATE_FILE = "cycle_state.json"
CYCLE_HISTORY_FILE = "cycle_history.json"
TICKER_CACHE_FILE = "ticker_names_cache.json"

# ============================================
# PDF GENERATION CONFIGURATION
# ============================================
# Set to False to disable PDF generation
GENERATE_PDF = False  # <-- CHANGE THIS to False to disable PDF generation

# ============================================
# EMAIL CONFIGURATION
# ============================================
# Set to True to attach HTML file to email (in addition to email body)
ATTACH_HTML_TO_EMAIL = True  # <-- CHANGE THIS to False to disable HTML attachment

# Email Configuration
EMAIL_CONFIG = {
    "smtp_server": "smtp-relay.brevo.com",
    "smtp_port": 587,
    "sender_email": "osama@optvest.capital",
    "sender_name": "Osama Safwat",
    "smtp_login": os.environ.get("SMTP_LOGIN", "PUT_YOUR_SMTP_LOGIN_HERE"),
    "smtp_password": os.environ.get("SMTP_PASSWORD", "PUT_YOUR_SMTP_PASSWORD_HERE"),
    "recipient_emails": [
        # "someone@example.com",
    ]
}

SEND_EMAIL_AUTOMATICALLY = True
BENCHMARK_SYMBOL = "EGX30.INDX"

# ============================================
# END OF CONFIGURATIONS
# ============================================

MONTH_NAME_TO_NUM = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
}
SKIP_MONTHS = {MONTH_NAME_TO_NUM[m] for m, skip in MONTHS_TO_SKIP.items() if skip}


def fetch_data(symbol: str, from_date: str, to_date: str, is_index: bool = False) -> pd.DataFrame:
    if is_index and not symbol.endswith('.INDX'):
        symbol = f"{symbol}.INDX"
    elif not is_index and '.' not in symbol and not symbol.endswith('.EGX'):
        symbol = f"{symbol}.EGX"

    url = f"{EODHD_BASE_URL}/eod/{symbol}"
    params = {'api_token': EODHD_API_KEY, 'from': from_date, 'to': to_date, 'fmt': 'json'}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep((attempt + 1) * 2)
                continue
            if r.status_code != 200:
                return pd.DataFrame()
            data = r.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df['price'] = df['adjusted_close'].fillna(df['close'])
            return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


def fetch_ticker_info(ticker: str) -> Optional[Dict]:
    """Fetch ticker information from EODHD API including company name."""
    try:
        url = f"{EODHD_BASE_URL}/eod/{ticker}.EGX"
        params = {
            'api_token': EODHD_API_KEY,
            'fmt': 'json'
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if data and isinstance(data, list) and len(data) > 0:
                # The API returns the ticker info in the response
                return {'name': ticker}  # Fallback to ticker if name not available
        return None
    except Exception:
        return None


def get_company_name_from_api(ticker: str) -> str:
    """Get company name from EODHD API ticker details."""
    # Try to fetch from exchanges endpoint
    try:
        url = f"{EODHD_BASE_URL}/exchanges/EGX"
        params = {'api_token': EODHD_API_KEY, 'fmt': 'json'}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                for item in data:
                    if item.get('code') == ticker or item.get('code') == f"{ticker}.EGX":
                        name = item.get('name', '')
                        if name:
                            return name
    except Exception:
        pass

    # Fallback: try to get from ticker search
    try:
        url = f"{EODHD_BASE_URL}/search/{ticker}"
        params = {'api_token': EODHD_API_KEY, 'fmt': 'json', 'exchange': 'EGX'}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                for item in data:
                    if item.get('code') == ticker or item.get('code') == f"{ticker}.EGX":
                        name = item.get('name', '')
                        if name:
                            return name
    except Exception:
        pass

    return ticker  # Return ticker as fallback


def load_tickers() -> List[Dict]:
    """Load tickers from CSV and fetch company names from API."""
    url = "https://drive.google.com/uc?export=download&id=1auS4afvNnZNZnRGp09QPitnoD7M0mt_R"

    # Try to load from cache first
    cache_data = None
    if os.path.exists(TICKER_CACHE_FILE):
        try:
            with open(TICKER_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            cache_age = datetime.now() - datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
            if cache_age.days < 7:  # Use cache if less than 7 days old
                print(f"📋 Using cached ticker names from {cache_data.get('timestamp')}")
                return cache_data.get('tickers', [])
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))

        code_col = df.columns[0]
        tickers = []

        # Get list of unique tickers from CSV
        unique_codes = []
        for _, row in df.iterrows():
            code = str(row[code_col]).replace('.CA', '').replace('.EGX', '')
            if code and code.lower() not in ('nan', 'none', ''):
                unique_codes.append(code)

        # Remove duplicates while preserving order
        seen = set()
        unique_codes = [x for x in unique_codes if not (x in seen or seen.add(x))]

        print(f"📋 Found {len(unique_codes)} unique tickers in CSV")
        print(f"🔄 Fetching company names from API (this may take a moment)...")

        for i, code in enumerate(unique_codes):
            # Get company name from API
            name = get_company_name_from_api(code)

            # If name is still just the code, try alternative method
            if name == code:
                # Try to get from exchange listing with a small delay
                time.sleep(0.1)
                name = get_company_name_from_api(code)

            tickers.append({'code': code, 'name': name})

            if (i + 1) % 50 == 0:
                print(f"   Fetched {i+1}/{len(unique_codes)} names...")

        # Save to cache
        try:
            with open(TICKER_CACHE_FILE, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'tickers': tickers
                }, f, indent=2)
            print(f"💾 Saved ticker names to cache: {TICKER_CACHE_FILE}")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")

        print(f"✅ Loaded {len(tickers)} EGX tickers with names from API")
        if tickers:
            print(f"📋 Sample: {tickers[:3]}")
        return tickers
    except Exception as e:
        print(f"❌ Failed to load tickers: {e}")
        return []


def get_weight_for_year(year: int, current_year: int) -> float:
    years_since = current_year - year
    for max_years, weight in YEAR_WEIGHTS:
        if years_since <= max_years:
            return weight
    return 0.0


def calculate_seasonality(df: pd.DataFrame, analysis_date: str) -> Tuple[List[float], List[float]]:
    if df.empty or len(df) < MIN_HISTORY_DAYS:
        return [], []

    analysis_dt = datetime.strptime(analysis_date, '%Y-%m-%d')
    df = df[df['date'] <= analysis_dt].copy()
    if df.empty or len(df) < MIN_HISTORY_DAYS:
        return [], []

    current_year = analysis_dt.year
    df['year'] = df['date'].dt.year

    daily_returns: List[List[Dict]] = [[] for _ in range(252)]

    for year, year_data in df.groupby('year'):
        weight = get_weight_for_year(year, current_year)
        if weight == 0:
            continue
        prices = year_data.sort_values('date')['price'].values
        for i in range(1, min(len(prices), 253)):
            daily_ret = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
            if abs(daily_ret) < MAX_RETURN_THRESHOLD:
                daily_returns[i - 1].append({'return': daily_ret, 'weight': weight})

    seasonality: List[float] = []
    cumulative: List[float] = []
    cum_sum = 0.0

    for day_rets in daily_returns:
        if day_rets:
            total_w = sum(r['weight'] for r in day_rets)
            avg_ret = (sum(r['return'] * r['weight'] for r in day_rets) / total_w) if total_w > 0 else 0.0
        else:
            avg_ret = 0.0
        cum_sum += avg_ret
        seasonality.append(avg_ret)
        cumulative.append(cum_sum)

    return seasonality, cumulative


def build_trading_calendar(benchmark_df: pd.DataFrame) -> Dict[str, int]:
    calendar: Dict[str, int] = {}
    for year, group in benchmark_df.groupby(benchmark_df['date'].dt.year):
        group_sorted = group.sort_values('date')
        for rank, (_, row) in enumerate(group_sorted.iterrows()):
            calendar[row['date'].strftime('%Y-%m-%d')] = rank
    return calendar


def get_trading_day_from_calendar(date_str: str, calendar: Dict[str, int]) -> int:
    target = datetime.strptime(date_str, '%Y-%m-%d')
    for offset in range(6):
        candidate = (target - timedelta(days=offset)).strftime('%Y-%m-%d')
        if candidate in calendar:
            return calendar[candidate]
    return 0


def get_trading_dates(from_date: str, to_date: str) -> List[str]:
    """Actual EGX trading days between two dates (inclusive), taken from the benchmark history."""
    df = fetch_data(BENCHMARK_SYMBOL, from_date, to_date, is_index=True)
    if df.empty:
        return []
    return [d.strftime('%Y-%m-%d') for d in df['date']]


def get_latest_trading_date() -> str:
    """Most recent EGX trading day that has closing data (today if data is already published)."""
    today = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
    dates = get_trading_dates(start, today)
    return dates[-1] if dates else today


def get_cycle_progress(cycle_start_date: str, holding_days: int) -> Tuple[int, Optional[str]]:
    """
    Count trading days elapsed since the cycle entry date.
    Returns (trading_days_passed, exit_date). exit_date is the holding_days-th trading day
    after entry, or None while the cycle is still running.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    dates = get_trading_dates(cycle_start_date, today)
    if dates:
        after_entry = [d for d in dates if d > cycle_start_date]
        exit_date = after_entry[holding_days - 1] if len(after_entry) >= holding_days else None
        return len(after_entry), exit_date

    # Fallback when benchmark data is unavailable: count Sun-Thu weekdays
    start = np.datetime64(cycle_start_date, 'D') + 1
    end = np.datetime64(today, 'D') + 1
    passed = int(np.busday_count(start, end, weekmask=EGX_WEEKMASK)) if end > start else 0
    exit_date = estimate_cycle_end_date(cycle_start_date, holding_days) if passed >= holding_days else None
    return passed, exit_date


def estimate_cycle_end_date(cycle_start_date: str, holding_days: int) -> str:
    """Planned exit date: holding_days EGX trading days (Sun-Thu) after entry, ignoring holidays."""
    end = np.busday_offset(np.datetime64(cycle_start_date, 'D'), holding_days,
                           roll='forward', weekmask=EGX_WEEKMASK)
    return str(end)


def get_price_on_exact_date(ticker: str, target_date: str) -> Optional[Tuple[str, float]]:
    """Get price on exact target date - no fallback to earlier date."""
    start = target_date
    end = (datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    df = fetch_data(ticker, start, end, is_index=False)
    if df.empty:
        return None
    df_exact = df[df['date'] == pd.to_datetime(target_date)]
    if df_exact.empty:
        return None
    row = df_exact.iloc[0]
    return row['date'].strftime('%Y-%m-%d'), row['price']


def get_price_on_or_before_date(ticker: str, target_date: str) -> Optional[Tuple[str, float]]:
    """Get price on or before target date (for entry when exact date not available)."""
    start = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=10)).strftime('%Y-%m-%d')
    end = (datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    df = fetch_data(ticker, start, end, is_index=False)
    if df.empty:
        return None
    df_before = df[df['date'] <= pd.to_datetime(target_date)]
    if df_before.empty:
        return None
    row = df_before.iloc[-1]
    return row['date'].strftime('%Y-%m-%d'), row['price']


def get_price_on_date(ticker: str, target_date: str) -> Optional[Tuple[str, float]]:
    """Get price on specific date (entry price)."""
    price_info = get_price_on_exact_date(ticker, target_date)
    if not price_info:
        price_info = get_price_on_or_before_date(ticker, target_date)
    return price_info


def get_current_price(ticker: str) -> Optional[Tuple[float, str]]:
    """Get the most recent available price for a ticker."""
    today = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    df = fetch_data(ticker, start, today, is_index=False)
    if df.empty:
        return None
    last_row = df.iloc[-1]
    return last_row['price'], last_row['date'].strftime('%Y-%m-%d')


def analyze_stock_for_entry(ticker: str, name: str, entry_date: str, holding_days: int,
                            trading_calendar: Dict[str, int]) -> Optional[Dict]:
    """Analyze a stock on the entry date to determine if it qualifies."""
    try:
        price_info = get_price_on_date(ticker, entry_date)
        if not price_info:
            return None
        actual_date, actual_price = price_info

        current_month = datetime.strptime(actual_date, '%Y-%m-%d').month
        if current_month in SKIP_MONTHS:
            return None

        df = fetch_data(ticker, "2005-01-01", actual_date, is_index=False)
        if df.empty or len(df) < MIN_HISTORY_DAYS:
            return None

        recent_data = df.tail(20)
        avg_liquidity = (recent_data['volume'] * recent_data['price']).mean()
        if avg_liquidity < MIN_LIQUIDITY:
            return None

        seasonality, historical_cumulative = calculate_seasonality(df, actual_date)
        if not seasonality:
            return None

        current_trading_day = get_trading_day_from_calendar(actual_date, trading_calendar)

        expected_return = 0.0
        for d in range(holding_days):
            day_idx = (current_trading_day + d) % len(seasonality)
            expected_return += seasonality[day_idx]

        if expected_return <= 0:
            return None

        rolling_year = df.tail(252)
        recent_cumulative: List[float] = []
        cum = 0.0
        for i in range(1, len(rolling_year)):
            ret = ((rolling_year.iloc[i]['price'] - rolling_year.iloc[i - 1]['price']) / rolling_year.iloc[i - 1]['price']) * 100
            if abs(ret) < MAX_RETURN_THRESHOLD:
                cum += ret
                recent_cumulative.append(cum)

        min_len = min(len(historical_cumulative), len(recent_cumulative))
        if min_len > 30:
            corr = np.corrcoef(historical_cumulative[:min_len], recent_cumulative[:min_len])[0, 1]
            corr = abs(corr) if not np.isnan(corr) else 0.0
        else:
            corr = 0.0

        if corr < MIN_CORRELATION:
            return None

        corr_weight = 3 if corr > 0.7 else (2 if corr > 0.5 else 1)
        opportunity_score = corr_weight * corr * abs(expected_return)

        stop_price = actual_price * (1 - STOP_LOSS_PCT / 100)

        return {
            'ticker': ticker.replace('.EGX', ''),
            'name': name[:35] if len(name) > 35 else name,
            'expected_return': round(expected_return, 2),
            'correlation': round(corr, 4),
            'opportunity_score': round(opportunity_score, 2),
            'avg_liquidity': round(avg_liquidity, 0),
            'entry_date': actual_date,
            'entry_price': round(actual_price, 2),
            'stop_loss_price': round(stop_price, 2),
            'stop_loss_pct': STOP_LOSS_PCT,
        }
    except Exception:
        return None


def load_saved_cycle() -> Optional[Dict]:
    """Load previously saved cycle state from file."""
    if os.path.exists(CYCLE_STATE_FILE):
        try:
            with open(CYCLE_STATE_FILE, 'r') as f:
                state = json.load(f)
            print(f"📁 Loaded saved cycle from {state.get('cycle_start_date')}")
            return state
        except Exception as e:
            print(f"⚠️ Could not load saved cycle: {e}")
    return None


def save_cycle_state(state: Dict):
    """Save current cycle state to file."""
    try:
        with open(CYCLE_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"💾 Cycle state saved to {CYCLE_STATE_FILE}")
    except Exception as e:
        print(f"⚠️ Could not save cycle state: {e}")


def archive_closed_cycle(closed: Dict):
    """Append a finished cycle (with its final prices) to the history file."""
    history = []
    if os.path.exists(CYCLE_HISTORY_FILE):
        try:
            with open(CYCLE_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not read cycle history: {e}")
    history.append(closed)
    try:
        with open(CYCLE_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"🗄️ Closed cycle archived to {CYCLE_HISTORY_FILE}")
    except Exception as e:
        print(f"⚠️ Could not save cycle history: {e}")


def update_saved_stocks_with_current_prices(saved_stocks: List[Dict], as_of_date: Optional[str] = None) -> List[Dict]:
    """
    Update saved stocks with CURRENT live prices (or the close on as_of_date) and recalculate P/L.
    IMPORTANT: Once a stock is stopped, its P/L is frozen and remains unchanged.
    """
    label = f"closing prices on {as_of_date}" if as_of_date else "latest prices"
    print(f"📊 Updating {len(saved_stocks)} stocks with {label}...")

    for idx, stock in enumerate(saved_stocks):
        ticker = stock['ticker']

        # Check if stock was already stopped in previous runs
        # If status is already 'Stopped', we keep it stopped and freeze the P/L
        if stock.get('status') == 'Stopped':
            print(f"      🔒 {ticker}: Already stopped at {stock.get('current_pnl_pct', 0):+.2f}% - frozen")
            continue  # Skip updating this stock - keep the stopped price

        print(f"   Fetching price for {ticker}...")

        if as_of_date:
            price_info = get_price_on_date(ticker, as_of_date)
            price_data = (price_info[1], price_info[0]) if price_info else None
        else:
            price_data = get_current_price(ticker)
        if price_data:
            current_price, price_date = price_data
            stock['current_price'] = round(current_price, 2)
            stock['last_price_date'] = price_date
            pnl_pct = ((current_price - stock['entry_price']) / stock['entry_price']) * 100
            stock['current_pnl_pct'] = round(pnl_pct, 2)

            # Check if stop loss was triggered
            if pnl_pct <= -stock['stop_loss_pct']:
                stock['status'] = 'Stopped'
                # Freeze the P/L at the stop loss level
                stock['current_pnl_pct'] = -stock['stop_loss_pct']
                stock['current_price'] = stock['stop_loss_price']  # Set price to stop loss price
                print(f"      🔴 {ticker}: STOP LOSS TRIGGERED at {stock['stop_loss_pct']}% loss - FROZEN")
            else:
                stock['status'] = 'Open'
                print(f"      🟢 {ticker}: Entry={stock['entry_price']}, Current={current_price}, P/L={pnl_pct:+.2f}%")
        else:
            # If no current price available, keep the last known values
            stock.setdefault('current_price', stock['entry_price'])
            stock.setdefault('last_price_date', stock['entry_date'])
            stock.setdefault('current_pnl_pct', 0.0)
            stock['status'] = 'Open'
            print(f"      ⚠️ {ticker}: No price available, keeping last known price")

    return saved_stocks


def close_cycle(saved_cycle: Dict, exit_date: Optional[str]) -> Dict:
    """Mark the expired cycle's positions at the exit-day close, archive it and return its summary."""
    start = saved_cycle['cycle_start_date']
    print(f"\n🏁 Closing cycle {start} (exit date: {exit_date or 'latest close'})")
    stocks = update_saved_stocks_with_current_prices(saved_cycle['stocks'], as_of_date=exit_date)
    for s in stocks:
        if s.get('status') == 'Open':
            s['status'] = 'Closed'

    pnl_pct = float(np.mean([s.get('current_pnl_pct', 0) for s in stocks])) if stocks else 0.0
    bench_entry = get_benchmark_performance_on_date(BENCHMARK_SYMBOL, start)
    bench_exit = (get_benchmark_performance_on_date(BENCHMARK_SYMBOL, exit_date) if exit_date
                  else get_current_benchmark_price(BENCHMARK_SYMBOL))
    if bench_entry['price'] and bench_exit['price']:
        bench_pnl = (bench_exit['price'] - bench_entry['price']) / bench_entry['price'] * 100
    else:
        bench_pnl = 0.0

    closed = {
        'cycle_start_date': start,
        'cycle_exit_date': exit_date or max(s.get('last_price_date', start) for s in stocks),
        'portfolio_pnl_pct': round(pnl_pct, 2),
        'benchmark_pnl_pct': round(bench_pnl, 2),
        'stopped_count': sum(1 for s in stocks if s.get('status') == 'Stopped'),
        'num_stocks': len(stocks),
        'stocks': stocks,
        'closed_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    archive_closed_cycle(closed)
    print(f"   Final P/L: {closed['portfolio_pnl_pct']:+.2f}% vs EGX30 {closed['benchmark_pnl_pct']:+.2f}%")
    return closed


def get_benchmark_performance_on_date(benchmark_ticker: str, target_date: str) -> Dict:
    """Get benchmark price on exact target date."""
    price_info = get_price_on_date(benchmark_ticker, target_date)

    if not price_info:
        start = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
        end = (datetime.strptime(target_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        df = fetch_data(benchmark_ticker, start, end, is_index=True)
        if df.empty:
            return {'price': None, 'date': None}
        df_target = df[df['date'] <= pd.to_datetime(target_date)]
        if df_target.empty:
            return {'price': None, 'date': None}
        row = df_target.iloc[-1]
        return {'price': round(row['price'], 2), 'date': row['date'].strftime('%Y-%m-%d')}

    return {'price': round(price_info[1], 2), 'date': price_info[0]}


def get_current_benchmark_price(benchmark_ticker: str) -> Dict:
    """Get current benchmark price and date."""
    today = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    df = fetch_data(benchmark_ticker, start, today, is_index=True)
    if df.empty:
        return {'price': None, 'date': None}
    row = df.iloc[-1]
    return {'price': round(row['price'], 2), 'date': row['date'].strftime('%Y-%m-%d')}


def scan_new_cycle(entry_date: str) -> Tuple[List[Dict], str]:
    """Scan and create a new cycle portfolio using specified entry date."""
    print(f"\n🔄 Starting NEW cycle with entry date: {entry_date}")

    start_bench = (datetime.strptime(entry_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
    end_bench = (datetime.strptime(entry_date, '%Y-%m-%d') + timedelta(days=5)).strftime('%Y-%m-%d')
    benchmark_df = fetch_data(BENCHMARK_SYMBOL, start_bench, end_bench, is_index=True)
    if benchmark_df.empty:
        print("⚠️ Could not fetch benchmark")
        trading_calendar = {}
    else:
        trading_calendar = build_trading_calendar(benchmark_df)

    tickers = load_tickers()
    if not tickers:
        return [], ""

    scan_results = []
    for i, stock in enumerate(tickers):
        if i > 0 and i % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)
        result = analyze_stock_for_entry(stock['code'], stock['name'], entry_date, HOLDING_DAYS, trading_calendar)
        if result:
            scan_results.append(result)
        if (i + 1) % 50 == 0:
            print(f"   Scanned {i+1}/{len(tickers)} stocks...")

    if RANKING_METHOD == 'return_only':
        key_fn = lambda x: x['expected_return']
    elif RANKING_METHOD == 'correlation_only':
        key_fn = lambda x: x['correlation']
    else:
        key_fn = lambda x: x['opportunity_score']

    selected = sorted(scan_results, key=key_fn, reverse=True)[:NUM_STOCKS]

    # Initialize with current prices (as of entry date, they're the same)
    for stock in selected:
        stock['current_price'] = stock['entry_price']
        stock['last_price_date'] = stock['entry_date']
        stock['current_pnl_pct'] = 0.0
        stock['status'] = 'Open'

    return selected, entry_date


def generate_report(stocks: List[Dict], cycle_start_date: str, is_new_cycle: bool = False,
                    previous_cycle: Optional[Dict] = None) -> Dict:
    """Generate report summary from stocks."""
    position_size_egp = HYPOTHETICAL_PORTFOLIO_SIZE / len(stocks) if stocks else 0
    portfolio_pnl_egp = 0
    portfolio_pnl_pct_weighted = 0

    for s in stocks:
        pnl_pct = s.get('current_pnl_pct', 0)
        portfolio_pnl_egp += position_size_egp * (pnl_pct / 100)
        portfolio_pnl_pct_weighted += pnl_pct / len(stocks) if stocks else 0

    avg_return = np.mean([s['expected_return'] for s in stocks]) if stocks else 0

    # Benchmark tracking from cycle start date
    benchmark_entry = get_benchmark_performance_on_date(BENCHMARK_SYMBOL, cycle_start_date)
    benchmark_current = get_current_benchmark_price(BENCHMARK_SYMBOL)

    if benchmark_entry['price'] and benchmark_current['price']:
        benchmark_pnl_pct = ((benchmark_current['price'] - benchmark_entry['price']) / benchmark_entry['price']) * 100
    else:
        benchmark_pnl_pct = 0.0

    latest_price_date = max([s.get('last_price_date', cycle_start_date) for s in stocks]) if stocks else cycle_start_date
    cycle_end_date = estimate_cycle_end_date(cycle_start_date, HOLDING_DAYS)

    return {
        'stocks': stocks,
        'num_stocks': len(stocks),
        'avg_expected_return': round(avg_return, 2),
        'holding_days': HOLDING_DAYS,
        'stop_loss_pct': STOP_LOSS_PCT,
        'hypothetical_portfolio_size': HYPOTHETICAL_PORTFOLIO_SIZE,
        'position_size_per_stock': round(position_size_egp, 2),
        'portfolio_pnl_egp': round(portfolio_pnl_egp, 2),
        'portfolio_pnl_pct': round(portfolio_pnl_pct_weighted, 2),
        'cycle_start_date': cycle_start_date,
        'cycle_end_date': cycle_end_date,
        'open_count': sum(1 for s in stocks if s.get('status') == 'Open'),
        'stopped_count': sum(1 for s in stocks if s.get('status') == 'Stopped'),
        'latest_price_date': latest_price_date,
        'benchmark_entry_price': benchmark_entry['price'],
        'benchmark_entry_date': benchmark_entry['date'],
        'benchmark_current_price': benchmark_current['price'],
        'benchmark_current_date': benchmark_current['date'],
        'benchmark_pnl_pct': round(benchmark_pnl_pct, 2),
        'is_new_cycle': is_new_cycle,
        'previous_cycle': previous_cycle,
    }


def generate_chart_and_save(stocks: List[Dict], holding_days: int, save_path: str) -> str:
    """Generate chart and save to file."""
    if not stocks:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    returns = [s['expected_return'] for s in stocks]
    tickers = [s['ticker'] for s in stocks]
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]

    bars = ax.barh(tickers, returns, color=colors, edgecolor='white', linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Expected Return (%)', fontsize=11)
    ax.set_title(f'Expected Seasonality Return (next {holding_days} trading days)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, ret in zip(bars, returns):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{ret:+.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    return save_path


def generate_pdf_from_html(html_path: str, pdf_path: str):
    """Generate PDF from HTML using weasyprint or wkhtmltopdf."""
    # Check if PDF generation is enabled
    if not GENERATE_PDF:
        print("📄 PDF generation is disabled (GENERATE_PDF = False)")
        return False

    try:
        # Try weasyprint first
        from weasyprint import HTML
        HTML(html_path).write_pdf(pdf_path)
        print(f"📄 PDF generated using WeasyPrint: {pdf_path}")
        return True
    except ImportError:
        try:
            # Try wkhtmltopdf as fallback
            import subprocess
            subprocess.run(['wkhtmltopdf', html_path, pdf_path], check=True, capture_output=True)
            print(f"📄 PDF generated using wkhtmltopdf: {pdf_path}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Neither WeasyPrint nor wkhtmltopdf is available. Please install one of them:")
            print("   - WeasyPrint: pip install weasyprint")
            print("   - wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
            return False
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return False


def generate_email_html(summary: Dict, chart_cid: str) -> str:
    """Generate HTML with proper CID reference for chart."""
    stocks = summary['stocks']
    formatted_date = datetime.now().strftime('%d %B %Y')
    portfolio_size = summary['hypothetical_portfolio_size']
    position_size = summary['position_size_per_stock']
    portfolio_pnl_pct = summary['portfolio_pnl_pct']
    portfolio_pnl_egp = summary['portfolio_pnl_egp']
    cycle_end_date = summary['cycle_end_date']
    open_count = summary['open_count']
    stopped_count = summary['stopped_count']
    num_stocks = summary['num_stocks']
    holding_days = summary['holding_days']
    stop_loss_pct = summary['stop_loss_pct']
    avg_return = summary['avg_expected_return']
    latest_price_date = summary['latest_price_date']
    cycle_start_date = summary['cycle_start_date']

    pnl_color = '#008000' if portfolio_pnl_pct >= 0 else '#cc0000'
    pnl_symbol = '+' if portfolio_pnl_pct >= 0 else ''

    benchmark_pnl = summary['benchmark_pnl_pct']
    outperformance = portfolio_pnl_pct - benchmark_pnl
    outperformance_color = '#008000' if outperformance >= 0 else '#cc0000'

    cycle_status = "🔄 ACTIVE CYCLE" if not summary.get('is_new_cycle') else "🆕 NEW CYCLE STARTED"

    previous = summary.get('previous_cycle')
    previous_cycle_line = ""
    if previous:
        previous_cycle_line = (
            f'<p style="color: #ffffff; font-size: 12px; margin: 8px 0 0 0;">'
            f'🏁 Previous cycle closed ({previous["cycle_start_date"]} → {previous["cycle_exit_date"]}): '
            f'{previous["portfolio_pnl_pct"]:+.2f}% vs EGX30 {previous["benchmark_pnl_pct"]:+.2f}% '
            f'| {previous["stopped_count"]}/{previous["num_stocks"]} stopped. Exit those positions and enter the list below.</p>'
        )

    # Build table rows
    table_rows = ""
    for s in stocks:
        pnl_symbol_s = '+' if s.get('current_pnl_pct', 0) >= 0 else ''
        status_text = s['status']
        status_color = '#27ae60' if status_text == 'Open' else '#c0392b'
        company_display = s['name'] if s['name'] != s['ticker'] else '—'

        table_rows += f'''
        <tr style="border-bottom: 1px solid #e0e0e0;">
            <td style="padding: 10px 8px; font-family: Arial, sans-serif; font-size: 13px;"><strong>{s['ticker']}</strong></td>
            <td style="padding: 10px 8px; font-family: Arial, sans-serif; font-size: 12px; color: #555;">{company_display}</td>
            <td style="padding: 10px 8px; font-family: Arial, sans-serif; font-size: 13px;">{s['entry_date']}</td>
            <td style="padding: 10px 8px; font-family: Arial, sans-serif; font-size: 13px; text-align: right;">{s['entry_price']:,.2f}</td>
            <td style="padding: 10px 8px; font-family: Arial, sans-serif; font-size: 13px; text-align: right;">{s.get('current_price', s['entry_price']):,.2f}</td>
            <td style="padding: 10px 8px; font-family: Arial, sans-serif; font-size: 12px; text-align: right; color: {'#008000' if s.get('current_pnl_pct', 0) >= 0 else '#cc0000'};">{pnl_symbol_s}{s.get('current_pnl_pct', 0):+.2f}%</td>
            <td style="padding: 10px 8px; font-family: Arial, sans-serif; font-size: 12px; text-align: center;"><span style="background: #f0f0f0; padding: 4px 8px;">{s['stop_loss_price']:,.2f}</span></td>
            <td style="padding: 10px 8px; font-family: Arial, sans-serif; font-size: 12px; text-align: center;"><span style="background: {status_color}20; color: {status_color}; padding: 4px 10px;">{status_text}</span></td>
         </tr>
        '''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Optvest EGX Seasonality Report</title>
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; background-color: #f5f7fa;">
    <center style="width: 100%;">
        <div style="max-width: 1100px; margin: 0 auto; background-color: #ffffff;">

            <!-- Header -->
            <div style="background-color: #1a5d7a; padding: 24px 28px;">
                <h1 style="color: #ffffff; font-size: 24px; margin: 0; font-weight: 600;">📈 EGX Seasonality Live Report</h1>
                <p style="color: #cce5f0; font-size: 13px; margin: 8px 0 0 0;">Report Date: {formatted_date} | {cycle_status}</p>
                <p style="color: #cce5f0; font-size: 11px; margin: 4px 0 0 0;">Cycle Entry Date: {cycle_start_date} | Last Price Update: {latest_price_date}</p>
                <p style="color: #cce5f0; font-size: 11px; margin: 2px 0 0 0;">Days in cycle: {(datetime.now() - datetime.strptime(cycle_start_date, '%Y-%m-%d')).days} days</p>
                {previous_cycle_line}
            </div>

            <!-- Stats Cards -->
            <table cellpadding="0" cellspacing="0" width="100%" style="background-color: #f5f7fa; padding: 20px;">
                <tr>
                    <td width="33%" style="padding: 8px;">
                        <table width="100%" cellpadding="12" style="background-color: #ffffff;">
                            <tr><td style="font-size: 12px; color: #666;">SELECTED STOCKS</td></tr>
                            <tr><td style="font-size: 28px; font-weight: bold; color: #1a5d7a;">{len(stocks)}</td></tr>
                        </table>
                    </td>
                    <td width="33%" style="padding: 8px;">
                        <table width="100%" cellpadding="12" style="background-color: #ffffff;">
                            <tr><td style="font-size: 12px; color: #666;">AVG EXPECTED RETURN</td></tr>
                            <tr><td style="font-size: 28px; font-weight: bold; color: #008000;">{avg_return:+.1f}%</td></tr>
                        </table>
                    </td>
                    <td width="34%" style="padding: 8px;">
                        <table width="100%" cellpadding="12" style="background-color: #ffffff;">
                            <tr><td style="font-size: 12px; color: #666;">STATUS</td></tr>
                            <tr><td style="font-size: 18px; font-weight: bold; color: #1a5d7a;">{open_count} Open / {stopped_count} Stopped</td></tr>
                        </table>
                    </td>
                 </tr>
                <tr>
                    <td width="33%" style="padding: 8px;">
                        <table width="100%" cellpadding="12" style="background-color: #ffffff;">
                            <tr><td style="font-size: 12px; color: #666;">HOLDING PERIOD</td></tr>
                            <tr><td style="font-size: 22px; font-weight: bold; color: #1a5d7a;">{holding_days} days</td></tr>
                        </table>
                    </td>
                    <td width="33%" style="padding: 8px;">
                        <table width="100%" cellpadding="12" style="background-color: #ffffff;">
                            <tr><td style="font-size: 12px; color: #666;">STOP LOSS</td></tr>
                            <tr><td style="font-size: 22px; font-weight: bold; color: #1a5d7a;">{stop_loss_pct}%</td></tr>
                        </table>
                    </td>
                    <td width="34%" style="padding: 8px;">
                        <table width="100%" cellpadding="12" style="background-color: #1a5d7a;">
                            <tr><td style="font-size: 12px; color: #cce5f0;">PORTFOLIO SIZE</td></tr>
                            <tr><td style="font-size: 20px; font-weight: bold; color: #ffffff;">{portfolio_size:,.0f} EGP</td></tr>
                        </table>
                    </td>
                 </tr>
             </table>

            <!-- Expected Returns Chart -->
            <div style="margin: 0 20px 20px 20px; text-align: center;">
                <img src="cid:{chart_cid}" alt="Expected Returns Chart" style="max-width: 100%; border: 1px solid #e0e0e0; border-radius: 8px;">
            </div>

            <!-- Portfolio Tracker Card -->
            <div style="background-color: #1a5d7a; margin: 0 20px 20px 20px; padding: 20px;">
                <h3 style="color: #ffffff; font-size: 16px; margin: 0 0 16px 0;">💰 Hypothetical Portfolio Tracker (Cycle: {cycle_start_date} → {cycle_end_date})</h3>
                <table width="100%" cellpadding="8">
                    <tr>
                        <td align="center" style="color: #ffffff;">
                            <div style="font-size: 22px; font-weight: bold;">{position_size:,.0f} EGP</div>
                            <div style="font-size: 11px;">Per Stock</div>
                        </td>
                        <td align="center" style="color: #ffffff;">
                            <div style="font-size: 22px; font-weight: bold; color: {pnl_color};">{pnl_symbol}{portfolio_pnl_pct:+.2f}%</div>
                            <div style="font-size: 11px;">Portfolio P/L</div>
                        </td>
                        <td align="center" style="color: #ffffff;">
                            <div style="font-size: 22px; font-weight: bold; color: {pnl_color};">{pnl_symbol}{portfolio_pnl_egp:+,.0f} EGP</div>
                            <div style="font-size: 11px;">Total P/L</div>
                        </td>
                     </tr>
                </table>
            </div>

            <!-- Current Cycle Tracker Table -->
            <div style="margin: 20px;">
                <h2 style="color: #1a4b64; font-size: 18px; margin: 0 0 8px 0;">🏆 Current Cycle Tracker</h2>
                <p style="font-size: 11px; color: #666; margin: 0 0 12px 0;">Entry Date: {cycle_start_date} | Last Price Update: {latest_price_date}</p>
                <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 13px; border: 1px solid #e0e0e0;">
                    <thead>
                        <tr style="background-color: #f2f2f2;">
                            <th style="padding: 12px 8px; text-align: left;">Ticker</th>
                            <th style="padding: 12px 8px; text-align: left;">Company</th>
                            <th style="padding: 12px 8px; text-align: left;">Entry Date</th>
                            <th style="padding: 12px 8px; text-align: right;">Entry Price</th>
                            <th style="padding: 12px 8px; text-align: right;">Current Price</th>
                            <th style="padding: 12px 8px; text-align: right;">P/L %</th>
                            <th style="padding: 12px 8px; text-align: center;">Stop Loss</th>
                            <th style="padding: 12px 8px; text-align: center;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}

                        <tr style="background-color: #e8f4f8; font-weight: bold; border-top: 2px solid #ccc;">
                            <td colspan="5" style="padding: 12px 8px; text-align: right;">Portfolio Weighted Average →</td>
                            <td style="padding: 12px 8px; text-align: right; color: {pnl_color};">{pnl_symbol}{portfolio_pnl_pct:+.2f}%</td>
                            <td colspan="2" style="padding: 12px 8px; text-align: center;">vs {len(stocks)} stocks</td>
                         </tr>

                        <tr style="background-color: #f5f5f5;">
                            <td colspan="5" style="padding: 12px 8px; text-align: right;">EGX30 Benchmark (same period) →<br><span style="font-size: 10px; color: #888;">Entry: {summary['benchmark_entry_date']} | Last: {summary['benchmark_current_date']}</span></td>
                            <td style="padding: 12px 8px; text-align: right; color: {'#008000' if benchmark_pnl >= 0 else '#cc0000'};">{'+' if benchmark_pnl >= 0 else ''}{benchmark_pnl:+.2f}%</td>
                            <td colspan="2" style="padding: 12px 8px; text-align: center;">@{summary['benchmark_current_price'] or 0:,.2f} EGP</td>
                         </tr>

                        <tr style="background-color: #e8f5e9; font-weight: bold;">
                            <td colspan="5" style="padding: 12px 8px; text-align: right;">📈 Outperformance (Portfolio - EGX30) →</td>
                            <td style="padding: 12px 8px; text-align: right; color: {outperformance_color};">{'+' if outperformance >= 0 else ''}{outperformance:+.2f}%</td>
                            <td colspan="2" style="padding: 12px 8px; text-align: center;">&nbsp;</td>
                         </tr>
                    </tbody>
                </table>
            </div>

            <!-- Strategy Guide -->
            <div style="background-color: #fef9e6; margin: 20px; border-left: 4px solid #e67e22; padding: 16px 24px; text-align: left;">
                <h3 style="color: #e67e22; font-size: 16px; margin: 0 0 12px 0;">📊 Trade Setup & Strategy Guide</h3>
                <ul style="margin: 0; padding-left: 20px; color: #333; font-size: 13px; line-height: 1.6;">
                    <li><strong>Capital allocation:</strong> Divide total portfolio equity by <strong>{len(stocks)}</strong> stocks (currently {position_size:,.0f} EGP per position).</li>
                    <li><strong>Entry:</strong> Execute limit orders near current price within 24 hours of report.</li>
                    <li><strong>Risk management:</strong> Strictly adhere to <strong>{stop_loss_pct}%</strong> stop loss below entry price.</li>
                    <li><strong>Exit:</strong> After <strong>{holding_days} trading days</strong> from entry, close all remaining positions.</li>
                    <li><strong>Cycle discipline:</strong> Approximately 50% of trades historically hit stop loss — winners compensate when risk control is applied.</li>
                    <li><strong>Rebalance:</strong> A fresh scan runs automatically every {holding_days} trading days — the report will say NEW CYCLE STARTED when it is time to re-enter.</li>
                </ul>
            </div>

            <!-- Footer -->
            <div style="background-color: #f5f7fa; padding: 16px 20px; text-align: center; border-top: 1px solid #e0e0e0;">
                <p style="font-size: 11px; color: #888; margin: 0;">
                    ⚠️ This report is generated by an algorithmic seasonality model. Past performance does not guarantee future results.<br>
                    Hypothetical {portfolio_size:,.0f} EGP portfolio simulation for tracking purposes only. Actual results may vary.<br>
                    Optvest Capital | EGX Seasonality Strategy | Generated on {formatted_date}
                </p>
            </div>
        </div>
    </center>
</body>
</html>'''
    return html


def send_email_report(html_content: str, chart_path: str, html_path: str = None, pdf_path: str = None,
                      is_new_cycle: bool = False):
    """Send email with HTML content, chart image, HTML file attachment, and optional PDF attachment."""
    if not SEND_EMAIL_AUTOMATICALLY:
        print("📧 Email sending disabled")
        return

    recipients = EMAIL_CONFIG.get('recipient_emails', [])
    if not recipients:
        print("⚠️ No recipient emails configured.")
        return

    prefix = "🆕 NEW CYCLE | " if is_new_cycle else ""
    subject = f"📊 {prefix}Optvest Live Seasonality Report | {NUM_STOCKS} stocks | Stop {STOP_LOSS_PCT}% | {datetime.now().strftime('%Y-%m-%d')}"

    # Create multipart message
    msg = MIMEMultipart('mixed')
    msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
    msg['Bcc'] = ', '.join(recipients)
    msg['Subject'] = subject

    # Create related part for HTML with embedded image
    msg_related = MIMEMultipart('related')
    msg.attach(msg_related)

    # Create alternative part for plain text and HTML
    msg_alternative = MIMEMultipart('alternative')
    msg_related.attach(msg_alternative)

    # Plain text version
    plain_text = f"""EGX Seasonality Report
Report Date: {datetime.now().strftime('%Y-%m-%d')}
Top {NUM_STOCKS} stocks with expected returns.

This email contains an HTML version with full formatting and a PDF attachment.
Please view the HTML version or PDF attachment for complete details."""
    msg_alternative.attach(MIMEText(plain_text, 'plain'))

    # HTML version (email body)
    msg_alternative.attach(MIMEText(html_content, 'html'))

    # Attach chart image with CID for HTML embedding
    if os.path.exists(chart_path):
        with open(chart_path, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data, name=os.path.basename(chart_path))
        image.add_header('Content-ID', '<expected_returns_chart>')
        image.add_header('Content-Disposition', 'inline', filename=os.path.basename(chart_path))
        msg_related.attach(image)
        print(f"📊 Chart attached: {chart_path}")
    else:
        print(f"⚠️ Chart file not found: {chart_path}")

    # Attach HTML file if enabled - embed chart as base64 for standalone viewing
    if ATTACH_HTML_TO_EMAIL and html_path and os.path.exists(html_path):
        try:
            # Read the HTML file
            with open(html_path, 'r', encoding='utf-8') as f:
                html_data = f.read()

            # Embed the chart as base64 in the HTML attachment
            if os.path.exists(chart_path):
                with open(chart_path, 'rb') as f:
                    img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')

                # Replace the cid reference with base64 in the HTML attachment
                html_data = html_data.replace(
                    'src="cid:expected_returns_chart"',
                    f'src="data:image/png;base64,{img_base64}"'
                )

            html_attachment = MIMEText(html_data, 'html', 'utf-8')
            html_attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(html_path))
            msg.attach(html_attachment)
            print(f"📎 HTML file attached: {html_path}")
        except Exception as e:
            print(f"⚠️ Could not attach HTML file: {e}")

    # Attach PDF if it exists
    if pdf_path and os.path.exists(pdf_path):
        try:
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            pdf_attachment = MIMEApplication(pdf_data, _subtype='pdf')
            pdf_attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
            msg.attach(pdf_attachment)
            print(f"📎 PDF attached: {pdf_path}")
        except Exception as e:
            print(f"⚠️ Could not attach PDF: {e}")
    else:
        if GENERATE_PDF:
            print(f"⚠️ PDF file not found: {pdf_path}")

    try:
        print(f"📧 Connecting to SMTP server...")
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'], timeout=30) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['smtp_login'], EMAIL_CONFIG['smtp_password'])
            print(f"📧 Sending to {len(recipients)} recipients...")
            server.send_message(msg)
        print(f"✅ Email sent successfully to {len(recipients)} recipients.")
    except Exception as e:
        print(f"❌ Email failed: {e}")


def main():
    print("\n" + "=" * 70)
    print("🏦 EGX SEASONALITY LIVE SCANNER - Automatic Cycle Rollover")
    print("=" * 70)

    manual_entry_date = CYCLE_ENTRY_DATE.strip() if CYCLE_ENTRY_DATE and CYCLE_ENTRY_DATE.strip() else None

    print(f"Configuration:")
    print(f"   Number of stocks: {NUM_STOCKS}")
    print(f"   Holding days: {HOLDING_DAYS}")
    print(f"   Stop loss: {STOP_LOSS_PCT}%")
    print(f"   Hypothetical Portfolio: {HYPOTHETICAL_PORTFOLIO_SIZE:,} EGP")
    print(f"   Manual entry date: {manual_entry_date or 'none (automatic)'}")
    print(f"   Email sending: {'ON' if SEND_EMAIL_AUTOMATICALLY else 'OFF'}")
    print(f"   PDF generation: {'ON' if GENERATE_PDF else 'OFF'}")
    print(f"   Attach HTML to email: {'ON' if ATTACH_HTML_TO_EMAIL else 'OFF'}")
    print("=" * 70)

    saved_cycle = load_saved_cycle()
    previous_cycle = None
    new_entry_date = None
    manual_date_used = saved_cycle.get('manual_entry_date_used') if saved_cycle else None

    # A manual date is applied only once: if it is the saved cycle's start date, or it was
    # already used before, it is ignored and the normal automatic rollover takes over.
    if manual_entry_date and (saved_cycle is None or
                              manual_entry_date not in (saved_cycle.get('cycle_start_date'), manual_date_used)):
        print(f"\n📅 Manual entry date {manual_entry_date} set - starting a cycle on that date (one time only).")
        new_entry_date = manual_entry_date
        manual_date_used = manual_entry_date
    elif saved_cycle is None:
        new_entry_date = get_latest_trading_date()
        print(f"\n📅 No saved cycle - starting the first cycle on the latest trading day: {new_entry_date}")
    else:
        saved_start_date = saved_cycle['cycle_start_date']
        if manual_entry_date and manual_date_used is None and manual_entry_date == saved_start_date:
            manual_date_used = manual_entry_date  # remember that the old config date was already applied
        days_passed, exit_date = get_cycle_progress(saved_start_date, HOLDING_DAYS)
        print(f"\n📆 Cycle {saved_start_date}: {days_passed}/{HOLDING_DAYS} trading days passed")
        if days_passed >= HOLDING_DAYS:
            previous_cycle = close_cycle(saved_cycle, exit_date)
            new_entry_date = get_latest_trading_date()
            print(f"\n🔁 Cycle complete - rolling over automatically to a new cycle on {new_entry_date}")

    if new_entry_date:
        stocks, cycle_start_date = scan_new_cycle(new_entry_date)
        is_new_cycle = True
    else:
        print(f"\n📊 Continuing EXISTING cycle from {saved_cycle['cycle_start_date']}")
        print(f"   Fetching latest prices for all positions...")
        stocks = update_saved_stocks_with_current_prices(saved_cycle['stocks'])
        cycle_start_date = saved_cycle['cycle_start_date']
        is_new_cycle = False

    if not stocks:
        # Saved state is left untouched, so the next scheduled run simply tries again.
        print("❌ No qualifying stocks found.")
        return

    # Generate report summary
    summary = generate_report(stocks, cycle_start_date, is_new_cycle, previous_cycle)
    summary['stocks'] = stocks

    # Save cycle state for next run
    save_cycle_state({
        'cycle_start_date': cycle_start_date,
        'stocks': stocks,
        'manual_entry_date_used': manual_date_used,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    print(f"\n✅ Report generated:")
    print(f"   Cycle: {cycle_start_date} → {summary['cycle_end_date']}")
    print(f"   Portfolio P/L: {summary['portfolio_pnl_pct']:+.2f}% ({summary['portfolio_pnl_egp']:+,.0f} EGP)")
    print(f"   Status: {summary['open_count']} Open / {summary['stopped_count']} Stopped")
    print(f"   Last price update: {summary['latest_price_date']}")

    # Generate chart
    generate_chart_and_save(stocks, HOLDING_DAYS, CHART_PATH)
    print(f"📊 Chart saved: {CHART_PATH}")

    # Generate HTML report
    html_report = generate_email_html(summary, "expected_returns_chart")
    with open(HTML_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(html_report)
    print(f"\n📄 HTML report saved: {HTML_OUTPUT}")

    # Generate PDF from HTML (if enabled)
    pdf_generated = False
    if GENERATE_PDF:
        pdf_generated = generate_pdf_from_html(HTML_OUTPUT, PDF_OUTPUT)
    else:
        print("📄 PDF generation skipped (GENERATE_PDF = False)")

    # Send email with HTML body, HTML file attachment, and PDF attachment
    send_email_report(
        html_report,
        CHART_PATH,
        HTML_OUTPUT,
        PDF_OUTPUT if (GENERATE_PDF and pdf_generated) else None,
        is_new_cycle=is_new_cycle,
    )
    print("\n✨ Done.")


if __name__ == "__main__":
    main()
