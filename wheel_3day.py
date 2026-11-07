# ============================================================================
# MULTI-SYMBOL 3-DAY WHEEL STRATEGY
# WITH CATEGORIES (GROWTH, DEFENSIVE, UNCORRELATED)
#
# Fixes in this version (see the notes at the bottom of the file):
#   1. Option chain search really filters by expiry (the old `expiration_dates=` was silently ignored)
#   2. Profit target / stop loss send a real buy-to-close order (they only edited the state file before)
#   3. Positions survive restarts and new days (the state file was wiped every new day)
#   4. One wheel per symbol; category / total limits count assigned stock too
#   5. Every cycle is checked against the real Alpaca account and open orders
#   6. Assignment / expiry is read from the account, not guessed from the next morning's price
#   7. Orders that fill while being cancelled are still recorded
#   8. Covered calls have their own minimum credit and log why none was sold
# ============================================================================

# ============================================================================
# ========================== USER CONFIGURATION =============================
# ============================================================================
import os

STRATEGY_NAME = "WHEEL_3DAY"
STRATEGY_ID = "WHEEL_3DAY"

# Keys: set the environment variables ALPACA_API_KEY / ALPACA_SECRET_KEY, or paste them here
# (never commit real keys to GitHub).
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "PUT_YOUR_ALPACA_API_KEY_HERE")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "PUT_YOUR_ALPACA_SECRET_KEY_HERE")
ALPACA_PAPER = True

RISK_FREE_RATE = 0.035
MAX_DAILY_LOSS = 500.0
MAX_BUYING_POWER_USE = 0.90
MIN_OPTIONS_BP_RESERVE = 500.0
MONITOR_INTERVAL_SECONDS = 30
EXPIRY_CHECK_MINUTES = 30            # how often expired / assigned options are checked against the account
MAX_HISTORY_RECORDS = 100

# ---------- NEW SETTINGS ----------
MAX_WHEELS_PER_SYMBOL = 1            # one wheel per symbol: a CSP, OR the assigned stock + its covered call
EXPIRY_SEARCH_EXTRA_DAYS = 4         # look for the listed expiry closest to days_to_expiry, up to this many days later
STRIKE_SEARCH_RANGE = 0.25           # only load strikes within +/-25% of the price (keeps the chain small)
ORDER_FILL_TIMEOUT_SECONDS = 30      # an unfilled limit order is cancelled after this long
CLOSE_PRICE_BUFFER = 0.00            # buy-to-close limit = ask + this ($); 0 = pay the ask
COUNT_UNTRACKED_BROKER_POSITIONS = True   # True = a short put / short call / 100+ shares in the account for one of
                                          # these symbols that this strategy did not open still blocks a new CSP
ADOPT_UNTRACKED_STOCK = False        # True = 100+ shares of a strategy symbol found in the account (not opened by this
                                     # strategy) are treated as wheel stock and covered calls are sold on them.
                                     # Only use it if this account runs nothing else on these symbols.
LOG_NO_CC_EVERY_MINUTES = 15         # how often "no covered call found" and its reason is logged

# ========================== CATEGORY CONFIGURATION ==========================
# YOU define the symbols - only include ones with weekly options!
# cc_min_credit = minimum premium for a covered call (separate from the CSP min_credit)

SYMBOL_CATEGORIES = {
    'GROWTH': {
        'symbols': ['NVDA', 'SPY', 'QQQ', 'AAPL', 'AMZN', 'TSLA', 'MSFT', 'AMD', 'GOOGL'],
        'max_positions': 2,           # max growth symbols at a time (CSP or assigned stock)
        'requires_pairing': True,     # Must have defensive OR uncorrelated
        'pair_with': ['DEFENSIVE', 'UNCORRELATED'],
        'csp_delta_target': 0.12,
        'csp_delta_range': (0.02, 0.16),
        'cc_delta_target': 0.25,
        'cc_delta_range': (0.02, 0.30),
        'position_size': 1,
        'min_credit': 0.20,
        'max_credit': 2.00,
        'cc_min_credit': 0.05,
        'profit_target': 0.50,
        'stop_loss': 1.5,
        'days_to_expiry': 5
    },

    'DEFENSIVE': {
        'symbols': ['WMT', 'PG', 'JNJ', 'KO', 'PEP'],
        'max_positions': 2,
        'requires_pairing': False,
        'pair_with': [],
        'csp_delta_target': 0.12,
        'csp_delta_range': (0.05, 0.20),
        'cc_delta_target': 0.25,
        'cc_delta_range': (0.05, 0.30),
        'position_size': 1,
        'min_credit': 0.20,
        'max_credit': 1.50,
        'cc_min_credit': 0.05,
        'profit_target': 0.50,
        'stop_loss': 1.5,
        'days_to_expiry': 5
    },

    'UNCORRELATED': {
        'symbols': ['GLD', 'IBIT', 'XOP', 'SLV'],
        'max_positions': 2,
        'requires_pairing': False,
        'pair_with': [],
        'csp_delta_target': 0.12,
        'csp_delta_range': (0.02, 0.16),
        'cc_delta_target': 0.25,
        'cc_delta_range': (0.02, 0.30),
        'position_size': 1,
        'min_credit': 0.20,
        'max_credit': 2.00,
        'cc_min_credit': 0.05,
        'profit_target': 0.50,
        'stop_loss': 2,
        'days_to_expiry': 5
    }
}

# ========================== GLOBAL LIMITS ==========================
MAX_TOTAL_CSP_POSITIONS = 5           # Max CSP positions across ALL symbols
MAX_TOTAL_STOCK_POSITIONS = 5         # Max stock positions across ALL symbols (CSPs count too: each can be assigned)

STATE_FILE = f"{STRATEGY_ID}_state.json"
HEARTBEAT_FILE = f"{STRATEGY_ID}_heartbeat.txt"
LOG_FILE = f"{STRATEGY_ID}_strategy.log"

# ============================================================================
# ========================== END OF CONFIGURATION ============================
# ============================================================================

import sys, json, time, warnings, gc, signal, traceback, re
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime, date, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Any

from alpaca.data import StockHistoricalDataClient, OptionHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest, OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus

# ========================== LOGGING ==========================
import logging
from logging.handlers import RotatingFileHandler

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(STRATEGY_ID)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.propagate = False

# (the old version printed every line twice: once by the console handler, once by print)
def log_info(msg): logger.info(msg)
def log_error(msg): logger.error(f"❌ {msg}")
def log_warning(msg): logger.warning(f"⚠️ {msg}")
def log_success(msg): logger.info(f"✅ {msg}")

# ========================== SIGNAL HANDLER ==========================
def signal_handler(sig, frame):
    log_info(f"\n🛑 Received signal {sig}, saving state...")
    if 'master_manager' in globals():
        master_manager.save_state()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ========================== MARKET HOURS ==========================
NY_TZ = ZoneInfo("America/New_York")

def get_market_holidays(year: int) -> List[date]:
    holidays = [date(year, 1, 1)]
    mlk = date(year, 1, 1)
    while mlk.weekday() != 0: mlk += timedelta(days=1)
    holidays.append(mlk + timedelta(days=14))
    pres = date(year, 2, 1)
    while pres.weekday() != 0: pres += timedelta(days=1)
    holidays.append(pres + timedelta(days=14))
    good_friday = {2024: date(2024,3,29), 2025: date(2025,4,18), 2026: date(2026,4,3),
                   2027: date(2027,3,26), 2028: date(2028,4,14), 2029: date(2029,3,30), 2030: date(2030,4,19)}
    if year in good_friday: holidays.append(good_friday[year])
    mem = date(year, 5, 31)
    while mem.weekday() != 0: mem -= timedelta(days=1)
    holidays.append(mem)
    holidays.append(date(year, 6, 19))
    holidays.append(date(year, 7, 4))
    labor = date(year, 9, 1)
    while labor.weekday() != 0: labor += timedelta(days=1)
    holidays.append(labor)
    thanks = date(year, 11, 1)
    while thanks.weekday() != 3: thanks += timedelta(days=1)
    holidays.append(thanks + timedelta(days=21))
    holidays.append(date(year, 12, 25))
    adjusted = []
    for h in holidays:
        if h.weekday() == 5: adjusted.append(h - timedelta(days=1))
        elif h.weekday() == 6: adjusted.append(h + timedelta(days=1))
        else: adjusted.append(h)
    return adjusted

MARKET_HOLIDAYS = {}
for y in range(2020, 2031): MARKET_HOLIDAYS[y] = get_market_holidays(y)

def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5: return False
    if d.year in MARKET_HOLIDAYS and d in MARKET_HOLIDAYS[d.year]: return False
    return True

def get_next_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_trading_day(d): d += timedelta(days=1)
    return d

def is_market_open(now: datetime) -> Tuple[bool, str]:
    d = now.date()
    if d.weekday() >= 5: return False, "Weekend"
    if d.year in MARKET_HOLIDAYS and d in MARKET_HOLIDAYS[d.year]: return False, "Holiday"
    if now.time() < dt_time(9, 30): return False, "Pre-market"
    if now.time() >= dt_time(16, 0): return False, "After-hours"
    return True, "Open"

def update_heartbeat():
    try:
        with open(HEARTBEAT_FILE, 'w') as f:
            f.write(f"{datetime.now(NY_TZ).isoformat()}\n")
    except Exception: pass

# ========================== ALPACA API ==========================
log_info(f"Initializing Alpaca clients for {STRATEGY_ID}...")

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
stock_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
option_data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
log_success("Alpaca clients initialized")

# ========================== SAFE CONVERSION FUNCTIONS ==========================
def safe_float(value):
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def safe_int(value):
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0

def safe_bool(value):
    if value is None:
        return False
    try:
        return bool(value)
    except (ValueError, TypeError):
        return False

def enum_text(value) -> str:
    """'filled' for OrderStatus.FILLED, 'short' for PositionSide.SHORT, etc."""
    return str(getattr(value, 'value', value) or '').lower()

# ========================== OPTION PRICING ==========================
def calculate_implied_volatility(price, S, K, T, r, opt_type):
    if T <= 0: return 0.01
    def f(sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if opt_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) - price
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1) - price
    try: return brentq(f, 0.001, 5.0)
    except Exception: return None

def calculate_delta(price, K, S, T, r, opt_type):
    if T <= 0:
        if opt_type == 'put': return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    iv = calculate_implied_volatility(price, S, K, T, r, opt_type)
    if not iv: return None
    d1 = (np.log(S/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
    if opt_type == 'call': return norm.cdf(d1)
    return -norm.cdf(-d1)

def years_to_expiry(expiry: date) -> float:
    """Time from now to the 16:00 ET close on the expiry date, in years (the old code always used 5 days)."""
    close = datetime.combine(expiry, dt_time(16, 0), tzinfo=NY_TZ)
    return max((close - datetime.now(NY_TZ)).total_seconds(), 0) / (365.25 * 24 * 3600)

# ========================== DATA FUNCTIONS ==========================
def get_current_price(symbol: str):
    """Latest trade price, falling back to the quote midpoint."""
    try:
        try:
            trades = stock_data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
            if symbol in trades:
                price = safe_float(trades[symbol].price)
                if price > 0:
                    return price
        except Exception as e:
            log_warning(f"Trade API error for {symbol}: {e}")

        q = stock_data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        bid = safe_float(getattr(q, 'bid_price', 0))
        ask = safe_float(getattr(q, 'ask_price', 0))
        if ask <= 0 and bid > 0: return bid
        if bid <= 0 and ask > 0: return ask
        if bid > 0 and ask > 0: return (bid + ask) / 2
        return None
    except Exception as e:
        log_warning(f"Error getting price for {symbol}: {e}")
        return None

def get_option_quote(symbol: str):
    q = get_option_quotes([symbol]).get(symbol)
    return q

def get_option_quotes(symbols: List[str]) -> Dict[str, Dict]:
    """Quotes for many option symbols at once (the old code made one request per strike)."""
    out = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        try:
            res = option_data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=batch))
        except Exception as e:
            log_warning(f"Option quote error: {e}")
            continue
        for sym, q in res.items():
            bid, ask = safe_float(q.bid_price), safe_float(q.ask_price)
            out[sym] = {'bid': bid, 'ask': ask, 'mid': (bid + ask) / 2}
    return out

def get_option_chain(underlying: str, opt_type: str, price: float, days: int):
    """
    Contracts of the listed expiry closest to `days` from today (never earlier than tomorrow,
    at most days + EXPIRY_SEARCH_EXTRA_DAYS). Returns (expiry_date, [{'symbol','strike'}]) or (None, []).
    """
    today = datetime.now(NY_TZ).date()
    contracts, token = [], None
    try:
        while True:
            req = GetOptionContractsRequest(
                underlying_symbols=[underlying], type=opt_type,
                expiration_date_gte=today + timedelta(days=1),
                expiration_date_lte=today + timedelta(days=days + EXPIRY_SEARCH_EXTRA_DAYS),
                strike_price_gte=str(round(price * (1 - STRIKE_SEARCH_RANGE), 2)),
                strike_price_lte=str(round(price * (1 + STRIKE_SEARCH_RANGE), 2)),
                limit=1000, page_token=token)
            res = trading_client.get_option_contracts(req)
            contracts += res.option_contracts or []
            token = getattr(res, 'next_page_token', None)
            if not token:
                break
    except Exception as e:
        log_warning(f"[{underlying}] option chain error: {e}")
        return None, []
    if not contracts:
        return None, []
    target = today + timedelta(days=days)
    expiries = sorted({c.expiration_date for c in contracts}, key=lambda e: (abs((e - target).days), e))
    expiry = expiries[0]
    return expiry, [{'symbol': c.symbol, 'strike': safe_float(c.strike_price)} for c in contracts if c.expiration_date == expiry]

def get_account_info():
    """Account info with safe None handling - never crashes"""
    try:
        a = trading_client.get_account()
        return {
            'cash': safe_float(a.cash),
            'equity': safe_float(a.equity),
            'portfolio_value': safe_float(a.portfolio_value),
            'buying_power': safe_float(a.buying_power),
            'options_buying_power': safe_float(a.options_buying_power) if hasattr(a, 'options_buying_power') else safe_float(a.buying_power) * 0.5,
            'maintenance_margin': safe_float(a.maintenance_margin),
        }
    except Exception as e:
        log_warning(f"Failed to get account info: {e}")
        return {}

OCC_RE = re.compile(r'^([A-Z.]+)(\d{6})([CP])(\d{8})$')

def parse_occ(symbol: str):
    """'SPY261113P00650000' -> {'underlying':'SPY','expiry':date,'type':'P','strike':650.0}"""
    m = OCC_RE.match(symbol or '')
    if not m:
        return None
    root, ymd, cp, strike = m.groups()
    return {'underlying': root, 'expiry': date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6])),
            'type': cp, 'strike': int(strike) / 1000}

def parse_option_symbol(symbol: str, underlying: str):
    p = parse_occ(symbol)
    if not p or p['underlying'] != underlying:
        return None
    return {'strike': p['strike'], 'expiry': p['expiry'].isoformat(), 'type': p['type']}

def option_expired(option_symbol: str, now: datetime) -> bool:
    p = parse_occ(option_symbol)
    if not p:
        return False
    return now >= datetime.combine(p['expiry'], dt_time(16, 0), tzinfo=NY_TZ)

# ========================== BROKER SNAPSHOT ==========================
def broker_snapshot() -> Optional[Dict]:
    """
    What the Alpaca account really holds for every symbol:
      shares[sym], short_puts[sym] = {option_symbol: contracts}, short_calls[sym] = {...},
      open_orders[sym] = number of open orders on the stock or its options.
    Returns None if Alpaca could not be reached (then nothing new is opened).
    """
    try:
        positions = trading_client.get_all_positions()
        orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    except Exception as e:
        log_warning(f"Could not read positions/orders from Alpaca: {e}")
        return None
    snap = {'shares': {}, 'short_puts': {}, 'short_calls': {}, 'long_options': {}, 'open_orders': {}}
    for p in positions:
        qty = abs(safe_float(p.qty))
        short = enum_text(getattr(p, 'side', '')) == 'short' or safe_float(p.qty) < 0
        occ = parse_occ(p.symbol)
        if occ:
            if short:
                key = 'short_puts' if occ['type'] == 'P' else 'short_calls'
                snap[key].setdefault(occ['underlying'], {})[p.symbol] = int(qty)
            else:
                snap['long_options'].setdefault(occ['underlying'], {})[p.symbol] = int(qty)
        elif not short:
            snap['shares'][p.symbol] = snap['shares'].get(p.symbol, 0) + int(qty)
    for o in orders:
        occ = parse_occ(o.symbol)
        und = occ['underlying'] if occ else o.symbol
        snap['open_orders'][und] = snap['open_orders'].get(und, 0) + 1
    return snap

# ========================== POSITION CLASS ==========================
class WheelPosition:
    def __init__(self, pos_type: str):
        self.position_type = pos_type
        self.option_symbol = None
        self.strike = None
        self.entry_price = None
        self.entry_time = None
        self.status = 'OPEN'
        self.quantity = 0
        self.collateral = 0
        self.shares_held = 0
        self.order_id = None
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.exit_reason = None
        self.delta_at_entry = None
        self.match_id = None
        self.client_order_id = None
        self.underlying_symbol = None

def iso(t):
    return t.isoformat() if hasattr(t, 'isoformat') else str(t)

# ========================== ORDER PLACEMENT ==========================
def new_client_order_id(symbol: str, kind: str) -> str:
    return f"{STRATEGY_ID}_{symbol}_{kind}_{datetime.now(NY_TZ).strftime('%Y%m%d_%H%M%S_%f')}"

def submit_limit_and_wait(option_symbol: str, qty: int, side: OrderSide, limit_price: float,
                          symbol: str, kind: str) -> Tuple[int, float, Optional[str], Optional[str]]:
    """
    Sends a limit order and waits up to ORDER_FILL_TIMEOUT_SECONDS. Unfilled -> cancelled.
    Returns (filled_qty, avg_fill_price, order_id, client_order_id). A fill that happens
    while the order is being cancelled is still counted (the old code lost those trades).
    """
    client_order_id = new_client_order_id(symbol, kind)
    try:
        order = trading_client.submit_order(LimitOrderRequest(
            symbol=option_symbol, qty=qty, side=side, type=OrderType.LIMIT,
            limit_price=round(limit_price, 2), time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id))
    except Exception as e:
        log_error(f"[{symbol}] {kind} order rejected: {e}")
        return 0, 0.0, None, None
    log_info(f"📤 [{symbol}] {kind} {enum_text(side)} {qty} {option_symbol} @ ${limit_price:.2f} (Client ID: {client_order_id})")

    o = order
    deadline = time.time() + ORDER_FILL_TIMEOUT_SECONDS
    while time.time() < deadline:
        time.sleep(1)
        try:
            o = trading_client.get_order_by_id(order.id)
        except Exception:
            continue
        st = enum_text(o.status)
        if st == 'filled' or st in ('canceled', 'rejected', 'expired'):
            break
    if enum_text(o.status) != 'filled':
        try:
            trading_client.cancel_order_by_id(order.id)
        except Exception:
            pass   # already filled or gone - the final status below decides
        for _ in range(10):
            time.sleep(1)
            try:
                o = trading_client.get_order_by_id(order.id)
            except Exception:
                continue
            if enum_text(o.status) in ('filled', 'canceled', 'rejected', 'expired'):
                break
    filled = safe_int(safe_float(o.filled_qty))
    avg = safe_float(o.filled_avg_price)
    if filled > 0:
        log_success(f"[{symbol}] {kind} FILLED {filled}/{qty} @ ${avg:.2f}")
    else:
        log_warning(f"[{symbol}] {kind} not filled ({enum_text(o.status)})")
    return filled, avg, str(order.id), client_order_id

# ========================== WHEEL MANAGER (PER SYMBOL) ==========================
class WheelManager:
    def __init__(self, symbol: str, category: str, config: Dict):
        self.symbol = symbol
        self.category = category
        self.config = config
        self.csp_positions = []
        self.stock_positions = []
        self.cc_positions = []
        self.closed_positions = []
        self.daily_pnl = 0.0
        self.total_collateral = 0.0
        self.total_shares = 0
        self.trades_today = 0
        self.last_loss_time = None
        self.last_no_cc_log = None

    def active_wheels(self) -> int:
        """A symbol is 'in use' while it has a CSP or assigned stock (the stock's CC belongs to that same wheel)."""
        return len(self.csp_positions) + len(self.stock_positions)

    def can_open_csp(self, master_state, snap, price=None):
        """Cheap checks first; account and price come from the snapshot / caller (saves API calls)."""
        config = self.config
        if self.active_wheels() >= MAX_WHEELS_PER_SYMBOL:
            return False, f"{self.symbol} already has a wheel running"
        if snap is None:
            return False, "Alpaca positions unknown"
        if snap['open_orders'].get(self.symbol):
            return False, f"open order(s) on {self.symbol} in the account"
        if COUNT_UNTRACKED_BROKER_POSITIONS and (snap['short_puts'].get(self.symbol) or snap['short_calls'].get(self.symbol)
                                                 or snap['shares'].get(self.symbol, 0) >= 100):
            return False, f"account already holds {self.symbol} options/shares"
        if master_state.get_category_count(self.category) >= config['max_positions']:
            return False, f"Category {self.category} at max ({config['max_positions']})"
        if master_state.get_total_csp_count() >= MAX_TOTAL_CSP_POSITIONS:
            return False, "Max total CSPs"
        if master_state.get_total_csp_count() + master_state.get_total_stock_count() >= MAX_TOTAL_STOCK_POSITIONS:
            return False, "Max total stock positions (each CSP can be assigned)"
        if config.get('requires_pairing', False):
            if not any(master_state.get_category_count(c) > 0 for c in config.get('pair_with', [])):
                return False, "Growth requires defensive OR uncorrelated position"
        if master_state.daily_pnl < -MAX_DAILY_LOSS:
            return False, "Daily loss limit"
        acc = master_state.account
        if not acc:
            return False, "No account"
        bp = acc.get('options_buying_power', 0)
        if bp < MIN_OPTIONS_BP_RESERVE:
            return False, f"Low BP: ${bp:.0f}"
        price = price or get_current_price(self.symbol)
        if not price:
            return False, "No price"
        if price * 0.95 * 100 * config.get('position_size', 1) > bp * MAX_BUYING_POWER_USE:
            return False, "Insufficient BP"
        return True, "OK"

    def can_open_cc(self, snap):
        if self.total_shares < 100:
            return False, "No stock"
        covered = sum(p.shares_held for p in self.cc_positions)
        if covered + 100 > self.total_shares:
            return False, "All covered"
        if snap is None:
            return False, "Alpaca positions unknown"
        if snap['open_orders'].get(self.symbol):
            return False, "open order on this symbol"
        held = snap['shares'].get(self.symbol, 0)
        if held < covered + 100:
            return False, f"account holds only {held} shares"
        return True, "OK"

    # ---------- bookkeeping ----------
    def add_csp(self, pos, master_state):
        pos.entry_time = datetime.now(NY_TZ)
        pos.underlying_symbol = self.symbol
        pos.match_id = f"{STRATEGY_ID}_{self.symbol}_CSP_{pos.strike}_{pos.entry_time.strftime('%Y%m%d_%H%M%S')}"
        self.csp_positions.append(pos)
        self.total_collateral += pos.collateral
        self.trades_today += 1
        master_state.trades_today += 1
        log_info(f"📈 [{self.symbol}] CSP OPENED: {pos.quantity}x ${pos.strike:.2f} {pos.option_symbol} @ ${pos.entry_price:.2f}")
        master_state.save_state()

    def add_cc(self, pos, master_state):
        pos.entry_time = datetime.now(NY_TZ)
        pos.underlying_symbol = self.symbol
        pos.match_id = f"{STRATEGY_ID}_{self.symbol}_CC_{pos.strike}_{pos.entry_time.strftime('%Y%m%d_%H%M%S')}"
        self.cc_positions.append(pos)
        self.trades_today += 1
        master_state.trades_today += 1
        log_info(f"📈 [{self.symbol}] CC OPENED: {pos.quantity}x ${pos.strike:.2f} {pos.option_symbol} @ ${pos.entry_price:.2f}")
        master_state.save_state()

    def add_stock(self, shares, cost, strike, premium, master_state, note='ASSIGNED'):
        pos = WheelPosition('STOCK')
        pos.shares_held = shares
        pos.entry_price = strike - premium
        pos.strike = strike
        pos.entry_time = datetime.now(NY_TZ)
        pos.collateral = cost
        pos.underlying_symbol = self.symbol
        pos.match_id = f"{STRATEGY_ID}_{self.symbol}_STOCK_{strike}_{pos.entry_time.strftime('%Y%m%d_%H%M%S')}"
        self.stock_positions.append(pos)
        self.total_shares += shares
        self.total_collateral += cost
        log_info(f"📊 [{self.symbol}] STOCK ADDED ({note}): {shares} shares, cost basis ${pos.entry_price:.2f}")
        master_state.save_state()

    def _record_closed(self, pos, kind, reason, exit_price, pnl, master_state):
        pos.status = 'CLOSED'
        pos.exit_time = datetime.now(NY_TZ)
        pos.exit_reason = reason
        pos.exit_price = exit_price
        pos.pnl = pnl
        self.daily_pnl += pnl
        master_state.daily_pnl += pnl
        if pnl < 0: self.last_loss_time = datetime.now(NY_TZ)
        rec = {
            'strategy_id': STRATEGY_ID, 'symbol': self.symbol, 'category': self.category,
            'type': kind, 'strike': pos.strike, 'entry_price': pos.entry_price, 'exit_price': exit_price,
            'pnl': pnl, 'exit_reason': reason, 'entry_time': iso(pos.entry_time), 'exit_time': iso(pos.exit_time),
            'option_symbol': pos.option_symbol, 'quantity': pos.quantity, 'shares': pos.shares_held,
            'match_id': pos.match_id, 'client_order_id': pos.client_order_id
        }
        self.closed_positions.append(rec)
        master_state.closed_positions.append(rec)

    def close_csp(self, pos, reason, exit_price, pnl, master_state):
        self.csp_positions.remove(pos)
        self.total_collateral -= pos.collateral
        self._record_closed(pos, 'CSP', reason, exit_price, pnl, master_state)
        log_info(f"📉 [{self.symbol}] CSP CLOSED: ${pos.strike:.2f} P&L: ${pnl:+.2f} ({reason})")
        master_state.save_state()

    def close_cc(self, pos, reason, exit_price, pnl, master_state):
        self.cc_positions.remove(pos)
        self._record_closed(pos, 'CC', reason, exit_price, pnl, master_state)
        log_info(f"📉 [{self.symbol}] CC CLOSED: ${pos.strike:.2f} P&L: ${pnl:+.2f} ({reason})")
        master_state.save_state()

    def close_stock(self, pos, exit_price, master_state, reason='CALLED_AWAY'):
        pnl = (exit_price - pos.entry_price) * pos.shares_held
        self.stock_positions.remove(pos)
        self.total_shares -= pos.shares_held
        self.total_collateral -= pos.collateral
        self._record_closed(pos, 'STOCK', reason, exit_price, pnl, master_state)
        log_success(f"🔄 [{self.symbol}] CYCLE COMPLETE - {pos.shares_held} shares out at ${exit_price:.2f}, P&L: ${pnl:+.2f} ({reason})")
        master_state.save_state()

    # ---------- expiries / assignments, read from the account ----------
    def reconcile(self, master_state, snap):
        """
        Compares what this strategy thinks it holds with the Alpaca account.
        A tracked option that is no longer in the account:
          - after its expiry: assigned (CSP -> shares appeared) / called away (CC -> shares left) / expired worthless
          - before its expiry: closed outside the strategy -> removed, P&L unknown (logged)
        """
        now = datetime.now(NY_TZ)
        held_puts = snap['short_puts'].get(self.symbol, {})
        held_calls = snap['short_calls'].get(self.symbol, {})
        shares = snap['shares'].get(self.symbol, 0)

        for csp in self.csp_positions[:]:
            if csp.option_symbol in held_puts:
                continue
            if option_expired(csp.option_symbol, now):
                if shares >= self.total_shares + 100 * csp.quantity:
                    log_info(f"⚠️ [{self.symbol}] CSP ASSIGNED: ${csp.strike} - {100 * csp.quantity} shares in the account")
                    self.csp_positions.remove(csp)
                    self.total_collateral -= csp.collateral
                    self.add_stock(100 * csp.quantity, csp.strike * 100 * csp.quantity, csp.strike, csp.entry_price, master_state)
                    self._record_closed(csp, 'CSP', 'ASSIGNED', csp.strike, 0.0, master_state)
                else:
                    pnl = csp.entry_price * 100 * csp.quantity
                    log_success(f"💰 [{self.symbol}] CSP EXPIRED WORTHLESS: ${csp.strike} P&L: +${pnl:.2f}")
                    self.close_csp(csp, 'EXPIRED', 0, pnl, master_state)
            else:
                log_warning(f"[{self.symbol}] CSP {csp.option_symbol} is no longer in the account (closed outside the strategy?) - removed")
                self.close_csp(csp, 'CLOSED_OUTSIDE_STRATEGY', None, 0.0, master_state)

        for cc in self.cc_positions[:]:
            if cc.option_symbol in held_calls:
                continue
            if option_expired(cc.option_symbol, now):
                if shares <= self.total_shares - cc.shares_held:
                    log_info(f"⚠️ [{self.symbol}] CC EXERCISED: shares called away at ${cc.strike}")
                    self.close_cc(cc, 'EXERCISED', cc.strike, cc.entry_price * 100 * cc.quantity, master_state)
                    stock = next((s for s in self.stock_positions if s.shares_held >= cc.shares_held), None)
                    if stock:
                        self.close_stock(stock, cc.strike, master_state)
                else:
                    pnl = cc.entry_price * 100 * cc.quantity
                    log_success(f"💰 [{self.symbol}] CC EXPIRED WORTHLESS: ${cc.strike} P&L: +${pnl:.2f}")
                    self.close_cc(cc, 'EXPIRED', 0, pnl, master_state)
            else:
                log_warning(f"[{self.symbol}] CC {cc.option_symbol} is no longer in the account (closed outside the strategy?) - removed")
                self.close_cc(cc, 'CLOSED_OUTSIDE_STRATEGY', None, 0.0, master_state)

        # Shares sold outside the strategy
        if self.total_shares > shares and not self.cc_positions:
            log_warning(f"[{self.symbol}] strategy expects {self.total_shares} shares, account holds {shares} - adjusting")
            for s in self.stock_positions[:]:
                if self.total_shares <= shares:
                    break
                price = get_current_price(self.symbol) or s.entry_price
                self.close_stock(s, price, master_state, reason='SOLD_OUTSIDE_STRATEGY')

        # Optional: take over shares that are in the account but not tracked
        if ADOPT_UNTRACKED_STOCK and not self.csp_positions and shares - self.total_shares >= 100:
            lots = (shares - self.total_shares) // 100 * 100
            basis = master_state.broker_avg_cost.get(self.symbol) or get_current_price(self.symbol) or 0
            log_info(f"[{self.symbol}] adopting {lots} untracked shares from the account (cost ${basis:.2f})")
            self.add_stock(lots, basis * lots, basis, 0.0, master_state, note='ADOPTED FROM ACCOUNT')

    # ---------- profit target / stop loss: real closing orders ----------
    def buy_to_close(self, pos, reason, ask, master_state, kind):
        limit = ask + CLOSE_PRICE_BUFFER
        filled, avg, _, _ = submit_limit_and_wait(pos.option_symbol, pos.quantity, OrderSide.BUY, limit, self.symbol, f"CLOSE_{kind}")
        if filled <= 0:
            return
        if filled < pos.quantity:   # partial: keep the rest open
            pnl = (pos.entry_price - avg) * 100 * filled
            pos.quantity -= filled
            if kind == 'CSP':
                part = pos.collateral * filled / (pos.quantity + filled)
                pos.collateral -= part
                self.total_collateral -= part
            else:
                pos.shares_held = pos.quantity * 100
            self.daily_pnl += pnl
            master_state.daily_pnl += pnl
            log_info(f"[{self.symbol}] {kind} partly closed {filled}, P&L ${pnl:+.2f}; {pos.quantity} still open")
            master_state.save_state()
            return
        pnl = (pos.entry_price - avg) * 100 * pos.quantity
        if kind == 'CSP':
            self.close_csp(pos, reason, avg, pnl, master_state)
        else:
            self.close_cc(pos, reason, avg, pnl, master_state)

    def monitor_positions(self, master_state):
        config = self.config
        quotes = get_option_quotes([p.option_symbol for p in self.csp_positions + self.cc_positions])
        for csp in self.csp_positions[:]:
            q = quotes.get(csp.option_symbol)
            if not q or q['ask'] <= 0: continue
            pnl_pct = (csp.entry_price - q['ask']) / csp.entry_price if csp.entry_price > 0 else 0
            if pnl_pct >= config['profit_target']:
                log_success(f"🎯 [{self.symbol}] CSP PROFIT TARGET: {pnl_pct*100:.1f}% - buying back")
                self.buy_to_close(csp, 'PROFIT_TARGET', q['ask'], master_state, 'CSP')
            elif pnl_pct <= -config['stop_loss']:
                log_warning(f"🛑 [{self.symbol}] CSP STOP LOSS: {pnl_pct*100:.1f}% - buying back")
                self.buy_to_close(csp, 'STOP_LOSS', q['ask'], master_state, 'CSP')
        for cc in self.cc_positions[:]:
            q = quotes.get(cc.option_symbol)
            if not q or q['ask'] <= 0: continue
            pnl_pct = (cc.entry_price - q['ask']) / cc.entry_price if cc.entry_price > 0 else 0
            if pnl_pct >= config['profit_target']:
                log_success(f"🎯 [{self.symbol}] CC PROFIT TARGET: {pnl_pct*100:.1f}% - buying back")
                self.buy_to_close(cc, 'PROFIT_TARGET', q['ask'], master_state, 'CC')

    # ---------- finding contracts ----------
    def get_csp_candidates(self, price):
        config = self.config
        expiry, puts = get_option_chain(self.symbol, 'put', price, config.get('days_to_expiry', 5))
        if not puts:
            return None
        T = years_to_expiry(expiry)
        puts = [p for p in puts if p['strike'] < price]
        quotes = get_option_quotes([p['symbol'] for p in puts])
        candidates = []
        for p in puts:
            q = quotes.get(p['symbol'])
            if not q or q['bid'] <= 0: continue
            delta = calculate_delta(q['bid'], p['strike'], price, T, RISK_FREE_RATE, 'put')
            if delta is None: continue
            d = abs(delta)
            if config['csp_delta_range'][0] <= d <= config['csp_delta_range'][1] and config['min_credit'] <= q['bid'] <= config['max_credit']:
                candidates.append({'symbol': p['symbol'], 'strike': p['strike'], 'bid': q['bid'], 'ask': q['ask'],
                                   'price': price, 'delta': d, 'credit': q['bid'], 'collateral': p['strike'] * 100, 'expiry': expiry.isoformat()})
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(x['delta'] - config['csp_delta_target']))

    def get_cc_candidates(self, price, cost_basis):
        """Returns (best_candidate or None, reason text when none)."""
        config = self.config
        expiry, calls = get_option_chain(self.symbol, 'call', price, config.get('days_to_expiry', 5))
        if not calls:
            return None, "no call contracts listed in the expiry window"
        T = years_to_expiry(expiry)
        floor = max(price, cost_basis)
        calls = [c for c in calls if c['strike'] > price and c['strike'] >= cost_basis]
        if not calls:
            return None, f"no strikes above max(price ${price:.2f}, cost basis ${cost_basis:.2f}) for {expiry}"
        quotes = get_option_quotes([c['symbol'] for c in calls])
        min_credit = config.get('cc_min_credit', config['min_credit'])
        seen = {'no_bid': 0, 'delta': 0, 'credit': 0}
        candidates = []
        for c in calls:
            q = quotes.get(c['symbol'])
            if not q or q['bid'] <= 0:
                seen['no_bid'] += 1; continue
            delta = calculate_delta(q['bid'], c['strike'], price, T, RISK_FREE_RATE, 'call')
            if delta is None or not (config['cc_delta_range'][0] <= delta <= config['cc_delta_range'][1]):
                seen['delta'] += 1; continue
            if not (min_credit <= q['bid'] <= config['max_credit']):
                seen['credit'] += 1; continue
            candidates.append({'symbol': c['symbol'], 'strike': c['strike'], 'bid': q['bid'], 'ask': q['ask'],
                               'delta': delta, 'credit': q['bid'], 'expiry': expiry.isoformat()})
        if not candidates:
            return None, (f"{len(calls)} calls >= ${floor:.2f} for {expiry}: {seen['no_bid']} no bid, "
                          f"{seen['delta']} delta outside {config['cc_delta_range']}, "
                          f"{seen['credit']} premium outside ${min_credit:.2f}-${config['max_credit']:.2f}")
        return min(candidates, key=lambda x: abs(x['delta'] - config['cc_delta_target'])), "OK"


# ========================== MASTER MANAGER ==========================
class MultiSymbolWheelManager:
    def __init__(self):
        self.symbol_managers: Dict[str, WheelManager] = {}
        self.daily_pnl = 0.0
        self.total_collateral = 0.0
        self.trades_today = 0
        self.closed_positions = []
        self.last_loss_time = None
        self.current_day = datetime.now(NY_TZ).strftime('%Y-%m-%d')
        self.snap = None
        self.account = {}
        self.broker_avg_cost = {}

        for category, config in SYMBOL_CATEGORIES.items():
            for symbol in config['symbols']:
                if symbol in self.symbol_managers:
                    log_warning(f"{symbol} is listed in more than one category - using {self.symbol_managers[symbol].category}")
                    continue
                self.symbol_managers[symbol] = WheelManager(symbol, category, config)

        self.load_state()

    # ---------- counts ----------
    def get_category_count(self, category: str) -> int:
        """Symbols in use in a category: open CSPs AND assigned stock (the old code ignored the stock)."""
        return sum(m.active_wheels() for m in self.symbol_managers.values() if m.category == category)

    def get_total_csp_count(self) -> int:
        return sum(len(m.csp_positions) for m in self.symbol_managers.values())

    def get_total_stock_count(self) -> int:
        return sum(len(m.stock_positions) for m in self.symbol_managers.values())

    def get_total_cc_count(self) -> int:
        return sum(len(m.cc_positions) for m in self.symbol_managers.values())

    def new_day_check(self):
        today = datetime.now(NY_TZ).strftime('%Y-%m-%d')
        if today != self.current_day:
            log_info(f"📅 New day {today} - daily P&L and trade counters reset (positions are kept)")
            self.current_day = today
            self.daily_pnl = 0.0
            self.trades_today = 0
            for m in self.symbol_managers.values():
                m.daily_pnl = 0.0
                m.trades_today = 0
            self.save_state()

    def refresh_snapshot(self):
        self.snap = broker_snapshot()
        self.account = get_account_info() if self.snap is not None else {}
        if self.snap is not None and ADOPT_UNTRACKED_STOCK:
            try:
                self.broker_avg_cost = {p.symbol: safe_float(p.avg_entry_price) for p in trading_client.get_all_positions()}
            except Exception:
                pass
        return self.snap

    def reconcile(self):
        if self.refresh_snapshot() is None:
            return
        for m in self.symbol_managers.values():
            m.reconcile(self, self.snap)

    # ---------- opening ----------
    def select_best_symbol(self, category: str) -> Optional[Dict]:
        config = SYMBOL_CATEGORIES[category]
        best, best_score = None, -float('inf')
        for symbol in config['symbols']:
            manager = self.symbol_managers.get(symbol)
            if manager is None or manager.category != category:
                continue
            price = get_current_price(symbol)
            if not price:
                continue
            ok, why = manager.can_open_csp(self, self.snap, price)
            if not ok:
                continue
            csp = manager.get_csp_candidates(price)
            if not csp:
                continue
            score = csp['credit'] * 100
            spread = csp['ask'] - csp['bid']
            if spread < 0.10: score += 0.50
            elif spread < 0.20: score += 0.25
            if score > best_score:
                best_score = score
                best = {'symbol': symbol, 'csp': csp, 'price': price, 'score': score}
        return best

    def open_csp_for_symbol(self, symbol: str, csp: Dict, config: Dict) -> bool:
        manager = self.symbol_managers[symbol]
        self.refresh_snapshot()   # re-check the account right before sending the order
        can, reason = manager.can_open_csp(self, self.snap, csp.get('price'))
        if not can:
            log_warning(f"[{symbol}] Cannot open CSP: {reason}")
            return False
        qty = config.get('position_size', 1)
        filled, avg, oid, client_id = submit_limit_and_wait(csp['symbol'], qty, OrderSide.SELL, csp['credit'], symbol, 'CSP')
        if filled <= 0:
            return False
        pos = WheelPosition('CSP')
        pos.option_symbol = csp['symbol']
        pos.strike = csp['strike']
        pos.entry_price = avg
        pos.quantity = filled
        pos.collateral = csp['strike'] * 100 * filled
        pos.delta_at_entry = csp['delta']
        pos.order_id = oid
        pos.client_order_id = client_id
        manager.add_csp(pos, self)
        return True

    def open_cc_for_symbol(self, symbol: str) -> bool:
        manager = self.symbol_managers[symbol]
        can, reason = manager.can_open_cc(self.snap)
        if not can:
            return False
        price = get_current_price(symbol)
        if not price:
            return False
        uncovered = manager.total_shares - sum(p.shares_held for p in manager.cc_positions)
        qty = uncovered // 100
        cost_basis = (sum(s.entry_price * s.shares_held for s in manager.stock_positions) / manager.total_shares
                      if manager.total_shares else 0)
        cc, why = manager.get_cc_candidates(price, cost_basis)
        if not cc:
            now = datetime.now(NY_TZ)
            if manager.last_no_cc_log is None or (now - manager.last_no_cc_log).total_seconds() >= LOG_NO_CC_EVERY_MINUTES * 60:
                log_warning(f"[{symbol}] holding {manager.total_shares} shares but no covered call found: {why}")
                manager.last_no_cc_log = now
            return False
        filled, avg, oid, client_id = submit_limit_and_wait(cc['symbol'], qty, OrderSide.SELL, cc['credit'], symbol, 'CC')
        if filled <= 0:
            return False
        pos = WheelPosition('CC')
        pos.option_symbol = cc['symbol']
        pos.strike = cc['strike']
        pos.entry_price = avg
        pos.quantity = filled
        pos.shares_held = filled * 100
        pos.delta_at_entry = cc['delta']
        pos.order_id = oid
        pos.client_order_id = client_id
        manager.add_cc(pos, self)
        return True

    def manage_positions(self):
        for manager in self.symbol_managers.values():
            if manager.csp_positions or manager.cc_positions:
                manager.monitor_positions(self)

    def open_new_positions(self):
        if self.refresh_snapshot() is None:
            return   # never open anything without knowing what the account holds

        # Covered calls first, on every symbol with uncovered wheel stock
        for symbol, manager in self.symbol_managers.items():
            if manager.total_shares >= 100:
                self.open_cc_for_symbol(symbol)

        has_growth = self.get_category_count('GROWTH') > 0
        priority = ['DEFENSIVE', 'UNCORRELATED', 'GROWTH'] if has_growth else ['GROWTH', 'DEFENSIVE', 'UNCORRELATED']
        for category in priority:
            config = SYMBOL_CATEGORIES[category]
            if self.get_category_count(category) >= config['max_positions']:
                continue
            if self.get_total_csp_count() >= MAX_TOTAL_CSP_POSITIONS:
                break
            best = self.select_best_symbol(category)
            if best:
                self.open_csp_for_symbol(best['symbol'], best['csp'], config)

    # ---------- state ----------
    def save_state(self):
        try:
            state = {
                'strategy_id': STRATEGY_ID,
                'timestamp': datetime.now(NY_TZ).isoformat(),
                'current_day': self.current_day,
                'daily_pnl': self.daily_pnl,
                'trades_today': self.trades_today,
                'symbols': {}
            }
            for symbol, manager in self.symbol_managers.items():
                state['symbols'][symbol] = {
                    'category': manager.category,
                    'daily_pnl': manager.daily_pnl,
                    'total_collateral': manager.total_collateral,
                    'total_shares': manager.total_shares,
                    'trades_today': manager.trades_today,
                    'csp_positions': [p.__dict__ for p in manager.csp_positions],
                    'stock_positions': [p.__dict__ for p in manager.stock_positions],
                    'cc_positions': [p.__dict__ for p in manager.cc_positions],
                    'closed_positions': manager.closed_positions[-MAX_HISTORY_RECORDS:]
                }
            state['closed_positions'] = self.closed_positions[-MAX_HISTORY_RECORDS:]
            tmp = STATE_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, STATE_FILE)   # never leaves a half-written state file
            return True
        except Exception as e:
            log_error(f"Error saving state: {e}")
            return False

    def load_state(self):
        """Positions are always loaded. Only the daily P&L / trade counters start fresh on a new day."""
        try:
            if not os.path.exists(STATE_FILE):
                return False
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            if state.get('strategy_id') != STRATEGY_ID:
                return False
            same_day = state.get('timestamp', '')[:10] == self.current_day
            if same_day:
                self.daily_pnl = state.get('daily_pnl', 0.0)
                self.trades_today = state.get('trades_today', 0)
            else:
                log_info("📅 New day - daily P&L and trade counters reset (positions kept)")

            for symbol, sdata in state.get('symbols', {}).items():
                manager = self.symbol_managers.get(symbol)
                if manager is None:
                    if sdata.get('csp_positions') or sdata.get('stock_positions') or sdata.get('cc_positions'):
                        log_warning(f"{symbol} has saved positions but is no longer in SYMBOL_CATEGORIES - not managed")
                    continue
                for key, kind in (('csp_positions', 'CSP'), ('stock_positions', 'STOCK'), ('cc_positions', 'CC')):
                    for pdata in sdata.get(key, []):
                        pos = WheelPosition(kind)
                        for k, v in pdata.items():
                            if hasattr(pos, k):
                                setattr(pos, k, v)
                        getattr(manager, key).append(pos)
                manager.total_collateral = sdata.get('total_collateral', 0.0)
                manager.total_shares = sum(p.shares_held for p in manager.stock_positions)
                manager.closed_positions = sdata.get('closed_positions', [])
                if same_day:
                    manager.daily_pnl = sdata.get('daily_pnl', 0.0)
                    manager.trades_today = sdata.get('trades_today', 0)
            self.closed_positions = state.get('closed_positions', [])
            log_info(f"✅ Loaded state: {self.get_total_csp_count()} CSP, {self.get_total_stock_count()} stock, {self.get_total_cc_count()} CC")
            return True
        except Exception as e:
            log_warning(f"Could not load state: {e}")
            return False

    def print_status(self):
        now = datetime.now(NY_TZ)
        is_open, reason = is_market_open(now)
        acc = get_account_info()
        log_info(f"\n{'='*80}")
        log_info(f"📊 STATUS - {now.strftime('%Y-%m-%d %H:%M:%S ET')}  Market: {'🟢 OPEN' if is_open else '🔴 CLOSED'} ({reason})")
        log_info(f"   Equity ${acc.get('equity', 0):,.2f} | Cash ${acc.get('cash', 0):,.2f} | Options BP ${acc.get('options_buying_power', 0):,.2f}")
        for category, config in SYMBOL_CATEGORIES.items():
            log_info(f"   {category}: {self.get_category_count(category)}/{config['max_positions']} symbols in use")
        log_info(f"   CSP {self.get_total_csp_count()} | STOCK {self.get_total_stock_count()} | CC {self.get_total_cc_count()} | "
                 f"Daily P&L ${self.daily_pnl:+,.2f} | Trades today {self.trades_today}")
        for symbol, manager in self.symbol_managers.items():
            if not (manager.csp_positions or manager.stock_positions or manager.cc_positions):
                continue
            quotes = get_option_quotes([p.option_symbol for p in manager.csp_positions + manager.cc_positions])
            log_info(f"   [{symbol}] ({manager.category}):")
            for kind, plist in (('CSP', manager.csp_positions), ('CC', manager.cc_positions)):
                for p in plist:
                    q = quotes.get(p.option_symbol)
                    if q:
                        pnl = (p.entry_price - q['ask']) * 100 * p.quantity
                        log_info(f"      {kind} {p.quantity}x ${p.strike:.2f} {p.option_symbol}: entry ${p.entry_price:.2f} | P&L ${pnl:+.2f}")
            if manager.stock_positions:
                price = get_current_price(symbol)
                for s in manager.stock_positions:
                    if price:
                        log_info(f"      STOCK {s.shares_held} sh @ ${s.entry_price:.2f} | unrealized ${(price - s.entry_price) * s.shares_held:+.2f}")
        if self.closed_positions:
            log_info("   Recent closed:")
            for pos in self.closed_positions[-3:]:
                log_info(f"      [{pos.get('symbol', '')}] {pos.get('type')}: ${pos.get('pnl', 0):+.2f} ({pos.get('exit_reason')})")
        log_info(f"{'='*80}")


# ========================== MAIN LOOP ==========================
def run_strategy():
    log_info("\n" + "="*80)
    log_info(f"🚀 {STRATEGY_NAME} STRATEGY  (ID {STRATEGY_ID})")
    for category, config in SYMBOL_CATEGORIES.items():
        log_info(f"   {category}: {', '.join(config['symbols'])} (max {config['max_positions']})")
    log_info("="*80)

    global master_manager
    master_manager = MultiSymbolWheelManager()
    master_manager.reconcile()   # catch up on anything that expired / was assigned while stopped

    last_reconcile = datetime.now(NY_TZ)
    last_status = datetime.now(NY_TZ) - timedelta(minutes=5)
    last_heartbeat = datetime.now(NY_TZ) - timedelta(minutes=5)
    loops = 0
    log_info("🚀 ENTERING MAIN LOOP")

    while True:
        try:
            now = datetime.now(NY_TZ)
            master_manager.new_day_check()
            is_open, reason = is_market_open(now)

            if not is_open:
                next_open = datetime.combine(now.date() if now.time() < dt_time(9, 30) else now.date() + timedelta(days=1),
                                             dt_time(9, 30), tzinfo=NY_TZ)
                if not is_trading_day(next_open.date()):
                    next_open = datetime.combine(get_next_trading_day(next_open.date()), dt_time(9, 30), tzinfo=NY_TZ)
                sleep_sec = (next_open - now).total_seconds()
                log_info(f"\n🔴 Market is CLOSED - {reason}. Next open {next_open.strftime('%Y-%m-%d %H:%M ET')} "
                         f"(sleeping {sleep_sec/3600:.1f} h)")
                master_manager.save_state()
                time.sleep(max(sleep_sec, 0) + 60)
                master_manager.new_day_check()
                master_manager.reconcile()   # expiries / assignments from the previous session
                last_reconcile = datetime.now(NY_TZ)
                continue

            if (now - last_heartbeat).total_seconds() >= 60:
                update_heartbeat()
                last_heartbeat = now

            if (now - last_reconcile).total_seconds() >= EXPIRY_CHECK_MINUTES * 60:
                master_manager.reconcile()
                last_reconcile = now

            master_manager.manage_positions()
            master_manager.open_new_positions()

            if (now - last_status).total_seconds() >= 60:
                master_manager.print_status()
                master_manager.save_state()
                last_status = now

            loops += 1
            if loops % 10 == 0:
                gc.collect()
            time.sleep(MONITOR_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log_info("\n🛑 Stopping...")
            master_manager.save_state()
            break
        except Exception as e:
            log_error(f"Error in main loop: {e}")
            traceback.print_exc()
            master_manager.save_state()
            time.sleep(MONITOR_INTERVAL_SECONDS * 2)


# ========================== START ==========================
if __name__ == "__main__":
    log_info("\n" + "="*80)
    log_info(f"⚠️  {'LIVE' if not ALPACA_PAPER else 'PAPER'} TRADING - {STRATEGY_NAME}")
    log_info("="*80)
    if ALPACA_API_KEY.startswith("PUT_YOUR") or ALPACA_SECRET_KEY.startswith("PUT_YOUR"):
        log_error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY (environment variables or the top of this file).")
        sys.exit(1)
    try:
        for category, config in SYMBOL_CATEGORIES.items():
            for symbol in config['symbols']:
                price = get_current_price(symbol)
                if price: log_success(f"   {symbol}: ${price:.2f}")
                else: log_warning(f"   {symbol}: failed to get price")
        acc = trading_client.get_account()
        log_info(f"\n✅ Account: {acc.account_number} | Equity ${safe_float(acc.equity):,.2f} | Cash ${safe_float(acc.cash):,.2f}")
    except Exception as e:
        log_error(f"Connection failed: {e}")
        sys.exit(1)

    run_strategy()

# ============================================================================
# WHAT WAS WRONG IN THE PREVIOUS VERSION
# ----------------------------------------------------------------------------
# Doubling / opening against the limits:
#  - Profit target / stop loss only removed the trade from the state file; no buy-to-close order was sent.
#    The option stayed open at Alpaca, the strategy thought the slot was free and sold another one.
#  - On every new day load_state() discarded all saved positions, so after any restart the strategy
#    started "empty" and opened new CSPs on top of the ones still open in the account.
#  - Per-symbol limit used the category's max_positions (2), so the same symbol could get 2 CSPs.
#  - Category counts ignored assigned stock, so an assignment freed the slot for another CSP.
#  - MAX_TOTAL_STOCK_POSITIONS was never checked.
#  - If an order filled while it was being cancelled after 30 s, the fill was not recorded.
#  - Nothing ever looked at the real Alpaca positions or open orders.
# No covered calls:
#  - GetOptionContractsRequest(expiration_dates=[...]) is not an Alpaca field and was silently dropped,
#    so the chain request had no expiry filter and returned only the first 100 contracts (Alpaca's default):
#    for big chains (SPY, QQQ, NVDA...) these are low strikes of the nearest expiry, i.e. no calls above the
#    price at all. Deltas were also computed as if every contract had 5 days left.
#  - After an assignment the stock was forgotten on the next day (same state wipe as above), and
#    assignment was guessed from the next morning's price instead of read from the account.
#  - A CC profit target only edited the state file: the call stayed short at Alpaca and no new CC could be sold.
#  - Covered calls had to pay at least min_credit ($0.20) at delta <= 0.30 above cost basis, which defensive
#    names often can't - and nothing was logged, so it was invisible. Now cc_min_credit + a logged reason.
# ============================================================================
