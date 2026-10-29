import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import requests
from transformers import pipeline
from fake_useragent import UserAgent
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import alpaca_trade_api as tradeapi
from pytz import timezone
import time
from pathlib import Path
import pandas_market_calendars as mcal
import holidays
import threading
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ALL SETTINGS AT THE TOP
# ============================================================================

# ============================================================================
# ALTON BOOK - Alton runs as its own portfolio inside the account
# ============================================================================
# Alton only counts and trades its OWN shares of the Alton tickers and its OWN cash
# (both recorded in strategy_state/). Positions, options and margin of other
# strategies in the same account are ignored and never touched.
ALTON_CASH_RESERVE_PCT = 0.5        # % of Alton's own value kept in cash when fully invested (0.5 -> 99.5% invested)
ALTON_REBALANCE_TOLERANCE = 0.20    # a ticker is only traded when it is more than 20% away from its target value
MIN_TRADE_VALUE = 500               # never place an Alton order smaller than this ($)
MIN_CONFIDENCE_TO_SWITCH = 0.0      # e.g. 0.65 = a BUY/SELL flip with less than 65% confidence is ignored
SH_HEDGE_ENABLED = True             # SELL cycles build the monthly SH hedge (SH_MONTHLY_ALLOCATION) inside the Alton book
ORDER_WAIT_MINUTES = 30             # how long to wait for orders to fill before cancelling what is left

# Alton's starting value is taken automatically from the Alton ETFs it holds the first time
# this version runs. To move money in or out of Alton, set a dollar amount here (e.g. 120000):
# Alton's cash is adjusted ONCE so its total value equals that amount, then it keeps compounding.
ALTON_CAPITAL_RESET = None

# ============================================================================
# MERCURY BANK CONFIGURATION
# ============================================================================
MERCURY_BANK_ENABLED = False

# ============================================================================
# HEDGE MODE CONFIGURATION
# ============================================================================
HEDGE_MODE_ENABLED = False
HEDGE_THRESHOLD_MULTIPLIER = 0.5

# ============================================================================
# HEDGE GROUPS CONFIGURATION
# ============================================================================
HEDGE_GROUPS = {
    "SH": {
        "enabled": True,
        "tickers": ["SPY", "IVV", "VOO"],
        "description": "Broad Market Hedge"
    },
    "PSQ": {
        "enabled": True,
        "tickers": ["QQQ", "XLC", "MAGS"],
        "description": "Growth Hedge"
    },
}

HEDGE_TICKERS_LIST = [ticker for ticker, config in HEDGE_GROUPS.items() if config["enabled"]]

# ============================================================================
# STRATEGY IDS FOR ORDER TRACKING
# ============================================================================
ALTON_STRATEGY_ID = "ALTON"
RS_STRATEGY_ID = "RS"
MM_STRATEGY_ID = "MM"

# ============================================================================
# TEST MODE CONFIGURATION
# ============================================================================
TEST_MODE = False

# ============================================================================
# MONEY MARKET CONFIGURATION
# ============================================================================
MONEY_MARKET_SWEEP_ENABLED = False
MONEY_MARKET_SWEEP_TICKERS = ["SGOV"]

MONEY_MARKET_LEVERAGE_ENABLED = False
MONEY_MARKET_LEVERAGE_TICKERS = ["SGOV"]
MONEY_MARKET_LEVERAGE_PERCENT = 0.00
MONEY_MARKET_LEVERAGE_MIN_POSITION = 10000
MONEY_MARKET_LEVERAGE_REBALANCE_THRESHOLD = 0.10

# ============================================================================
# HYBRID PARAMETERS
# ============================================================================
rs_allocation_percentage = 0.00     # share of Alton's value given to RS when RS is switched on
rs_num_stocks = 50
rs_rebalance_day = 3
rs_rsi_period = 50
rs_max_tickers_to_process = 3000
rs_min_order_cost = 0.0
rs_lookback_days = 100

PRICE_SHOCK_THRESHOLD = 0.10
VOLUME_SPIKE_THRESHOLD = 3.0
FILTER_RECENT_DAYS = 60

CASH_EQUIVALENTS = ["SGOV"]
rs_invalid_tickers = {'BRK.B', 'SQ', 'MRO', 'HEI.A', 'CTLT', 'CHK', 'BF.B', 'TPX', 'SWN', 'UHAL.B'}

# ============================================================================
# ALTON TICKERS - The core Alton portfolio
# ============================================================================
ALTON_TICKERS = {
    "SPY": 0.50,
    "QQQ": 0.25,
    "IVV": 0.15,
    "VOO": 0.10,
}

# ============================================================================
# OLD buy_tickers - Kept for backward compatibility
# ============================================================================
buy_tickers = ALTON_TICKERS.copy()

_active_weight_sum = sum(w for w in buy_tickers.values() if w > 0)
if _active_weight_sum > 0:
    buy_tickers = {t: (w / _active_weight_sum if w > 0 else 0.0) for t, w in buy_tickers.items()}

rebalancing_tickers = ["SGOV"] + HEDGE_TICKERS_LIST

SH_MONTHLY_ALLOCATION = {
    1: 0.00, 2: 0.00, 3: 0.10, 4: 0.10,
    5: 0.25, 6: 0.25, 7: 0.20, 8: 0.25,
    9: 0.30, 10: 0.15, 11: 0.00, 12: 0.00,
}

MONTH_NAMES = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

execution_days = 3
min_cash_percentage = ALTON_CASH_RESERVE_PCT / 100   # fraction, used by the RS part
sgov_sweep_percentage = 1.00
withdrawal_percentage = 0.0
withdrawal_completed = False

# ============================================================================
# REBALANCING CONFIGURATION
# ============================================================================
RS_REBALANCE_TOLERANCE = 0.20

# ============================================================================
# EXECUTION LOOP CONFIGURATION
# ============================================================================
EXECUTION_LOOP_INTERVAL = 10
MAX_EXECUTION_ITERATIONS = 1000

# ============================================================================
# API CLIENTS
# ============================================================================
API_CLIENTS = [
    {
        "name": "Alton RS",
        "api_key": os.environ.get("ALPACA_API_KEY", "PUT_YOUR_ALPACA_API_KEY_HERE"),
        "secret_key": os.environ.get("ALPACA_SECRET_KEY", "PUT_YOUR_ALPACA_SECRET_KEY_HERE"),
        "base_url": "https://paper-api.alpaca.markets/"
    },
]

# ============================================================================
# EMAIL CONFIGURATION
# ============================================================================
EMAIL_SENDER = "osamamaghawry@gmail.com"
EMAIL_RECEIVER = "osama@optvest.capital"
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "PUT_YOUR_GMAIL_APP_PASSWORD_HERE")

# ============================================================================
# MERCURY BANK API CONFIGURATION
# ============================================================================
MERCURY_API_TOKEN = os.environ.get("MERCURY_API_TOKEN", "PUT_YOUR_MERCURY_TOKEN_HERE")
MERCURY_API_BASE = "https://api.mercury.com/api/v1"

# ============================================================================
# TRANSPARENCY & AUDIT CONFIGURATION
# ============================================================================
AUDIT_ENABLED = True
AUDIT_DIR = Path("audit_trails")
AUDIT_DIR.mkdir(exist_ok=True)
FEATURE_DRIFT_THRESHOLD = 30
MAX_HISTORY_RECORDS = 500

# ============================================================================
# PORTFOLIO TEMPLATES & CACHES
# ============================================================================
portfolio_template = {"cash": 0, "other_stocks": {}}

TICKERS = ["SPY", "VIXY", "TLT", "GLD", "IBIT", "SH", "VGSH", "IEF", "HYG", "CPER"]
REQUIRED_COLUMNS = {f"{t}_Close" for t in TICKERS}

active_orders = {}
order_lock = threading.Lock()
monitor_thread_running = False
monitor_thread_instance = None

try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", truncation=True, device="cpu")
except Exception as e:
    print(f"Error initializing summarizer: {e}")
    summarizer = None

cached_market_data = {"data": {}, "timestamp": None}

valid_russell_tickers = []
if rs_allocation_percentage > 0:
    try:
        russell_df = pd.read_csv("https://drive.google.com/uc?export=download&id=1P2kjzCa3mJbU0JsvVY24jE9yEqhiE9P6")
        ticker_column = 'Symbol'
        if ticker_column not in russell_df.columns:
            raise KeyError(f"Column '{ticker_column}' not found")
        russell_df[ticker_column] = russell_df[ticker_column].astype(str).str.strip()
        valid_russell_tickers = russell_df[ticker_column].dropna().tolist()
        valid_russell_tickers = valid_russell_tickers[:rs_max_tickers_to_process]
        print(f"Loaded {len(valid_russell_tickers)} tickers from Russell 3000 CSV")
    except Exception as e:
        print(f"WARNING: Error loading Russell 3000 CSV: {e}")
        rs_allocation_percentage = 0.0
else:
    print("RS allocation is 0% - skipping Russell 3000 CSV load")


# ============================================================================
# DECISION TRANSPARENCY CLASS
# ============================================================================

class DecisionTransparency:
    """
    Provides full transparency for Random Forest decisions.
    Tracks feature importance, decision paths, and contribution analysis.
    """

    def __init__(self, model, feature_names, api_key=None):
        self.model = model
        self.feature_names = feature_names
        self.api_key = api_key
        self.decision_history = []
        self.feature_importance_history = []
        self.audit_dir = AUDIT_DIR
        self.audit_dir.mkdir(exist_ok=True)

    def analyze_decision(self, X_sample, sample_index=0):
        if hasattr(X_sample, 'iloc'):
            X_array = X_sample.iloc[[sample_index]].values
        else:
            X_array = X_sample[sample_index:sample_index+1]

        prediction = int(self.model.predict(X_array)[0])
        probabilities = self.model.predict_proba(X_array)[0]
        confidence = float(max(probabilities))

        sample_importance = self._get_local_importance(X_array)
        global_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        tree_votes = self._get_tree_votes(X_array)
        top_rules = self._get_top_rules(X_array)

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'prediction': 'BUY' if prediction == 1 else 'SELL',
            'confidence': confidence,
            'probability_buy': float(probabilities[1]),
            'probability_sell': float(probabilities[0]),
            'vote_count': {
                'buy_votes': sum(1 for v in tree_votes if v == 1),
                'sell_votes': sum(1 for v in tree_votes if v == 0),
                'total_trees': len(tree_votes)
            },
            'local_importance': sample_importance,
            'global_importance': global_importance,
            'top_rules': top_rules,
            'feature_values': dict(zip(self.feature_names, X_array[0].tolist()))
        }

        self.decision_history.append(analysis)
        return analysis

    def _get_local_importance(self, X_array):
        base_pred = self.model.predict_proba(X_array)[0]
        importance = {}

        for i, feature in enumerate(self.feature_names):
            perturbed = X_array.copy()
            perturbed[0][i] = perturbed[0][i] * 1.1
            new_pred = self.model.predict_proba(perturbed)[0]
            change = abs(new_pred[1] - base_pred[1])
            importance[feature] = change

        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _get_tree_votes(self, X_array):
        votes = []
        for tree in self.model.estimators_:
            vote = int(tree.predict(X_array)[0])
            votes.append(vote)
        return votes

    def _get_top_rules(self, X_array, top_n=5):
        rules = []
        for i, tree in enumerate(self.model.estimators_[:30]):
            path = self._get_tree_path(tree, X_array)
            if path:
                rules.append(path)

        rules.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return rules[:top_n]

    def _get_tree_path(self, tree, X_array):
        try:
            leaf_id = tree.apply(X_array)[0]
            node_indicator = tree.decision_path(X_array)
            node_indices = node_indicator.indices

            path_rules = []
            for node_id in node_indices:
                if node_id != leaf_id:
                    feature = self.feature_names[tree.tree_.feature[node_id]]
                    threshold = tree.tree_.threshold[node_id]
                    value = X_array[0][tree.tree_.feature[node_id]]
                    direction = '≤' if value <= threshold else '>'

                    path_rules.append({
                        'feature': feature,
                        'threshold': threshold,
                        'value': value,
                        'direction': direction
                    })

            leaf_value = tree.tree_.value[leaf_id][0]
            prediction = 1 if leaf_value[1] > leaf_value[0] else 0
            confidence = max(leaf_value) / sum(leaf_value)

            return {
                'path': path_rules,
                'prediction': 'BUY' if prediction == 1 else 'SELL',
                'confidence': float(confidence)
            }
        except:
            return None

    def generate_transparency_report(self, analysis, output_format='html'):
        if output_format == 'html':
            return self._generate_html_report(analysis)
        elif output_format == 'json':
            return json.dumps(analysis, indent=2)
        else:
            return analysis

    def _generate_html_report(self, analysis):
        top_features = list(analysis['local_importance'].items())[:10]

        comparison = []
        for feature, local_imp in list(analysis['local_importance'].items())[:10]:
            global_imp = analysis['global_importance'].get(feature, 0)
            comparison.append({
                'feature': feature,
                'local': local_imp,
                'global': global_imp,
                'value': analysis['feature_values'].get(feature, 0)
            })

        total_trees = analysis['vote_count']['total_trees']
        buy_pct = (analysis['vote_count']['buy_votes'] / total_trees * 100) if total_trees > 0 else 0
        sell_pct = (analysis['vote_count']['sell_votes'] / total_trees * 100) if total_trees > 0 else 0

        html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 850px; margin: 10px auto; padding: 15px; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1a237e; padding-bottom: 10px; margin-bottom: 15px;">
                <h3 style="color: #1a237e; margin: 0;">🎯 Decision Intelligence</h3>
                <span style="font-size: 11px; color: #666;">{analysis['timestamp'][:16]}</span>
            </div>

            <div style="display: flex; gap: 15px; margin: 10px 0;">
                <div style="flex: 1; background: {'#e8f5e9' if analysis['prediction'] == 'BUY' else '#ffebee'}; padding: 12px; border-radius: 6px; text-align: center;">
                    <div style="font-size: 10px; color: #666; text-transform: uppercase;">Decision</div>
                    <div style="font-size: 28px; font-weight: 700; color: {'#2e7d32' if analysis['prediction'] == 'BUY' else '#c62828'};">{analysis['prediction']}</div>
                </div>
                <div style="flex: 1; background: #e3f2fd; padding: 12px; border-radius: 6px; text-align: center;">
                    <div style="font-size: 10px; color: #666; text-transform: uppercase;">Confidence</div>
                    <div style="font-size: 28px; font-weight: 700; color: #1a237e;">{analysis['confidence']:.1%}</div>
                </div>
                <div style="flex: 1; background: #f3e5f5; padding: 12px; border-radius: 6px; text-align: center;">
                    <div style="font-size: 10px; color: #666; text-transform: uppercase;">Forest Vote</div>
                    <div style="font-size: 18px; font-weight: 700; color: #4a148c;">
                        BUY: {analysis['vote_count']['buy_votes']} | SELL: {analysis['vote_count']['sell_votes']}
                    </div>
                    <div style="font-size: 10px; color: #999;">({buy_pct:.0f}% / {sell_pct:.0f}%)</div>
                </div>
            </div>

            <div style="background: #fff; padding: 12px; border-radius: 6px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h4 style="color: #1a237e; margin: 0 0 8px 0; font-size: 13px;">📊 Top Feature Contributions (Today's Decision)</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background: #e8eaf6;">
                            <th style="padding: 6px; text-align: left;">Feature</th>
                            <th style="padding: 6px; text-align: right;">Value</th>
                            <th style="padding: 6px; text-align: right;">Importance</th>
                            <th style="padding: 6px; text-align: right;">Global Avg</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for item in comparison[:8]:
            bar_width = item['local'] * 100
            html += f"""
                        <tr style="border-bottom: 1px solid #f0f0f0;">
                            <td style="padding: 6px; font-weight: 600; color: #1a237e; font-size: 11px;">{item['feature']}</td>
                            <td style="padding: 6px; text-align: right; font-size: 11px;">{item['value']:.3f}</td>
                            <td style="padding: 6px; text-align: right;">
                                <div style="background: #e0e0e0; height: 6px; border-radius: 3px; width: 80px; display: inline-block;">
                                    <div style="background: #1a237e; height: 6px; border-radius: 3px; width: {bar_width}%;"></div>
                                </div>
                                <span style="margin-left: 6px; font-size: 11px;">{item['local']:.1%}</span>
                            </td>
                            <td style="padding: 6px; text-align: right; font-size: 11px;">{item['global']:.1%}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
            </div>

            <div style="background: #fff; padding: 12px; border-radius: 6px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h4 style="color: #1a237e; margin: 0 0 8px 0; font-size: 13px;">🔍 Sample Decision Rules (Top 3 Trees)</h4>
        """

        for i, rule in enumerate(analysis['top_rules'][:3], 1):
            if rule and 'path' in rule:
                path_str = " → ".join([f"{r['feature']} {r['direction']} {r['threshold']:.2f}" for r in rule['path'][:3]])
                html += f"""
                    <div style="background: #f5f5f5; padding: 8px; margin: 5px 0; border-radius: 4px; border-left: 3px solid {'#4caf50' if rule['prediction'] == 'BUY' else '#f44336'}; font-size: 11px;">
                        <strong>Tree #{i}:</strong> {rule['prediction']} (conf: {rule['confidence']:.1%})<br>
                        <span style="color: #666;">{path_str}</span>
                    </div>
                """

        html += """
            </div>

            <div style="background: #fff3e0; padding: 10px; border-radius: 6px; border-left: 4px solid #ff9800; margin: 10px 0;">
                <p style="margin: 0; color: #555; font-size: 12px;">
                    <strong>💡 Summary:</strong> The model recommended <strong>{prediction}</strong> with {confidence:.1%} confidence.
                    Primary drivers: {top_features}. Forest consensus: {buy_pct:.0f}% BUY vs {sell_pct:.0f}% SELL.
                </p>
            </div>
        </div>
        """.format(
            prediction=analysis['prediction'],
            confidence=analysis['confidence'] * 100,
            top_features=', '.join([f"{f[0]} ({f[1]*100:.1f}%)" for f in top_features[:3]]),
            buy_pct=buy_pct,
            sell_pct=sell_pct
        )

        return html

    def track_feature_importance_history(self, importance_dict):
        timestamp = datetime.now().isoformat()
        self.feature_importance_history.append({
            'timestamp': timestamp,
            'importance': importance_dict.copy()
        })

        if len(self.feature_importance_history) > MAX_HISTORY_RECORDS:
            self.feature_importance_history = self.feature_importance_history[-MAX_HISTORY_RECORDS:]

        if len(self.feature_importance_history) % 10 == 0:
            self._save_history()

    def _save_history(self):
        if not AUDIT_ENABLED:
            return
        try:
            history_file = self.audit_dir / f"feature_history_{self.api_key[-8:] if self.api_key else 'default'}.json"
            with open(history_file, 'w') as f:
                json.dump(self.feature_importance_history[-MAX_HISTORY_RECORDS:], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")

    def load_history(self):
        if not AUDIT_ENABLED:
            return
        try:
            history_file = self.audit_dir / f"feature_history_{self.api_key[-8:] if self.api_key else 'default'}.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.feature_importance_history = json.load(f)
                print(f"Loaded {len(self.feature_importance_history)} historical records")
        except Exception as e:
            print(f"Warning: Could not load history: {e}")

    def get_importance_trend(self, feature_name, window=30):
        importances = []
        timestamps = []

        for record in self.feature_importance_history[-window:]:
            imp = record['importance'].get(feature_name, 0)
            importances.append(imp)
            timestamps.append(record['timestamp'])

        return timestamps, importances

    def generate_feature_dashboard(self, save_path=None):
        if len(self.feature_importance_history) < 5:
            print("Insufficient history for dashboard")
            return None

        history = self.feature_importance_history[-100:]

        data = []
        for record in history:
            row = {'timestamp': record['timestamp']}
            row.update(record['importance'])
            data.append(row)

        df = pd.DataFrame(data)
        if df.empty:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        top_features = df.mean().sort_values(ascending=False).head(8).index.tolist()

        for feature in top_features:
            if feature in df.columns:
                axes[0].plot(range(len(df)), df[feature].values, label=feature, alpha=0.7, linewidth=2)

        axes[0].set_title('Top 8 Feature Importance Over Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Decision Number')
        axes[0].set_ylabel('Importance')
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        axes[0].grid(True, alpha=0.3)

        if not df.empty:
            current_importance = df.iloc[-1]
            avg_importance = df.mean()

            features = top_features
            current_vals = [current_importance.get(f, 0) for f in features]
            avg_vals = [avg_importance.get(f, 0) for f in features]

            x = np.arange(len(features))
            width = 0.35

            axes[1].bar(x - width/2, current_vals, width, label='Current', color='#1a237e', alpha=0.8)
            axes[1].bar(x + width/2, avg_vals, width, label='30-Day Avg', color='#ff9800', alpha=0.8)
            axes[1].set_title('Current vs Historical Feature Importance', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Features')
            axes[1].set_ylabel('Importance')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(features, rotation=45, ha='right', fontsize=10)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path and AUDIT_ENABLED:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")

        plt.close()

        return fig

    def check_feature_drift(self, threshold_pct=None):
        if threshold_pct is None:
            threshold_pct = FEATURE_DRIFT_THRESHOLD

        if len(self.feature_importance_history) < 10:
            return "Insufficient data for drift analysis (need at least 10 decisions)"

        current = self.feature_importance_history[-1]['importance']

        baseline_records = self.feature_importance_history[-30:]
        baseline = {}
        for record in baseline_records:
            for feature, imp in record['importance'].items():
                if feature not in baseline:
                    baseline[feature] = []
                baseline[feature].append(imp)
        baseline = {k: np.mean(v) for k, v in baseline.items() if v}

        drift_alert = []
        for feature, current_imp in current.items():
            baseline_imp = baseline.get(feature, 0)
            if baseline_imp > 0.001:
                drift_pct = abs((current_imp - baseline_imp) / baseline_imp * 100)
                if drift_pct > threshold_pct:
                    drift_alert.append({
                        'feature': feature,
                        'current': current_imp,
                        'baseline': baseline_imp,
                        'drift_pct': drift_pct
                    })

        drift_alert.sort(key=lambda x: x['drift_pct'], reverse=True)

        if drift_alert:
            alert_message = f"⚠️ FEATURE DRIFT ALERT - {len(drift_alert)} features exceeded {threshold_pct}% threshold:\n"
            for item in drift_alert[:5]:
                alert_message += f"  - {item['feature']}: {item['drift_pct']:.1f}% (current: {item['current']:.3f}, baseline: {item['baseline']:.3f})\n"
            return alert_message

        return "✅ No significant feature drift detected"


# ============================================================================
# SIGNAL HISTORY MANAGER
# ============================================================================

class SignalHistoryManager:
    """
    Manages the signal history file - tracks every signal generated by the model.
    This is the source of truth for what signals occurred on which dates.
    """

    def __init__(self, api_key, signal_dir="signal_history"):
        self.api_key = api_key
        self.signal_dir = Path(signal_dir)
        self.signal_dir.mkdir(exist_ok=True)
        self.history_file = self.signal_dir / f"signal_history_{api_key[-8:]}.csv"
        self.backup_file = self.signal_dir / f"signal_history_backup_{api_key[-8:]}.csv"
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.history_file.exists():
            df = pd.DataFrame(columns=[
                'date', 'signal', 'confidence', 'feature_importance',
                'signal_numeric', 'feature_names', 'top_features'
            ])
            df.to_csv(self.history_file, index=False)

    def add_signal(self, date, signal, confidence, feature_importance=None, feature_names=None):
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)

        signal_numeric = 1 if signal == 'BUY' else 0

        top_features = ""
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_5 = sorted_features[:5]
            top_features = ", ".join([f"{f}: {imp:.2%}" for f, imp in top_5])

        importance_json = json.dumps(feature_importance) if feature_importance else "{}"
        feature_names_json = json.dumps(feature_names) if feature_names else "[]"

        try:
            df = pd.read_csv(self.history_file)

            if date_str in df['date'].values:
                df.loc[df['date'] == date_str, 'signal'] = signal
                df.loc[df['date'] == date_str, 'confidence'] = confidence
                df.loc[df['date'] == date_str, 'feature_importance'] = importance_json
                df.loc[df['date'] == date_str, 'signal_numeric'] = signal_numeric
                df.loc[df['date'] == date_str, 'feature_names'] = feature_names_json
                df.loc[df['date'] == date_str, 'top_features'] = top_features
                df.to_csv(self.history_file, index=False)
            else:
                new_row = pd.DataFrame([{
                    'date': date_str,
                    'signal': signal,
                    'confidence': confidence,
                    'feature_importance': importance_json,
                    'signal_numeric': signal_numeric,
                    'feature_names': feature_names_json,
                    'top_features': top_features
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.history_file, index=False)

            self._create_backup()

        except Exception as e:
            print(f"Warning: Could not add signal to history: {e}")
            self._ensure_file_exists()

    def get_signal_on_date(self, date):
        try:
            df = pd.read_csv(self.history_file)
            if isinstance(date, datetime):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)

            row = df[df['date'] == date_str]
            if not row.empty:
                return {
                    'date': row.iloc[0]['date'],
                    'signal': row.iloc[0]['signal'],
                    'confidence': float(row.iloc[0]['confidence']),
                    'signal_numeric': int(row.iloc[0]['signal_numeric']),
                    'top_features': row.iloc[0]['top_features']
                }
            return None
        except Exception:
            return None

    def get_latest_signal(self):
        try:
            df = pd.read_csv(self.history_file)
            if df.empty:
                return None
            latest = df.iloc[-1]
            return {
                'date': latest['date'],
                'signal': latest['signal'],
                'confidence': float(latest['confidence']),
                'signal_numeric': int(latest['signal_numeric']),
                'top_features': latest['top_features']
            }
        except Exception:
            return None

    def get_yesterday_signal(self):
        try:
            df = pd.read_csv(self.history_file)
            if len(df) < 2:
                return None
            yesterday = df.iloc[-2]
            return {
                'date': yesterday['date'],
                'signal': yesterday['signal'],
                'confidence': float(yesterday['confidence']),
                'signal_numeric': int(yesterday['signal_numeric']),
                'top_features': yesterday['top_features']
            }
        except Exception:
            return None

    def get_signal_sequence(self, days_back=10):
        try:
            df = pd.read_csv(self.history_file)
            if df.empty:
                return []
            recent = df.tail(days_back)
            return recent.to_dict('records')
        except Exception:
            return []

    def has_signal_today(self, date=None):
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        else:
            date = str(date)

        try:
            df = pd.read_csv(self.history_file)
            return date in df['date'].values
        except Exception:
            return False

    def _create_backup(self):
        try:
            import shutil
            shutil.copy2(self.history_file, self.backup_file)
        except Exception:
            pass


# ============================================================================
# STRATEGY STATE MANAGER
# ============================================================================

class StrategyStateManager:
    """
    Saves the Alton strategy state (strategy_state/strategy_state_XXXXXXXX.json).

    It holds the Alton BOOK - Alton's own share counts (ledger_qty) and own cash
    (ledger_cash) - and the 3-day cycle:
    - direction: BUY or SELL (the last signal Alton acted on)
    - current_day: 1..execution_days of the running cycle (execution_days = cycle complete)
    - start_pct / progress_pct: how invested Alton was when the cycle started / should be now
      (0% = all Alton cash, 100% = fully invested)
    - last_step_date: the trading day the cycle last moved forward
    """

    STATE_NO_POSITION = "NO_POSITION"
    STATE_BUYING_DAY_1 = "BUYING_DAY_1"
    STATE_BUYING_DAY_2 = "BUYING_DAY_2"
    STATE_BUYING_DAY_3 = "BUYING_DAY_3"
    STATE_FULLY_INVESTED_BUY = "FULLY_INVESTED_BUY"
    STATE_SELLING_DAY_1 = "SELLING_DAY_1"
    STATE_SELLING_DAY_2 = "SELLING_DAY_2"
    STATE_SELLING_DAY_3 = "SELLING_DAY_3"
    STATE_FULLY_INVESTED_SELL = "FULLY_INVESTED_SELL"

    STATE_DESCRIPTIONS = {
        STATE_NO_POSITION: "No position",
        STATE_BUYING_DAY_1: "Buying - Day 1",
        STATE_BUYING_DAY_2: "Buying - Day 2",
        STATE_BUYING_DAY_3: "Buying - Day 3",
        STATE_FULLY_INVESTED_BUY: "Fully invested - holding",
        STATE_SELLING_DAY_1: "Selling - Day 1",
        STATE_SELLING_DAY_2: "Selling - Day 2",
        STATE_SELLING_DAY_3: "Selling - Day 3",
        STATE_FULLY_INVESTED_SELL: "Fully in Alton cash - holding",
    }

    def __init__(self, api_key, state_dir="strategy_state"):
        self.api_key = api_key
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / f"strategy_state_{api_key[-8:]}.json"
        self.backup_file = self.state_dir / f"strategy_state_backup_{api_key[-8:]}.json"
        self._ensure_state_file_exists()

    def _ensure_state_file_exists(self):
        if not self.state_file.exists():
            default_state = self._get_default_state()
            self._save_state(default_state)

    def _get_default_state(self):
        # No Alton book yet: it is created from the account's Alton holdings on the first run.
        return {
            'state': self.STATE_NO_POSITION,
            'last_updated': None,
            'version': 3
        }

    def _save_state(self, state):
        state['last_updated'] = datetime.now().isoformat()
        state['state_description'] = self.STATE_DESCRIPTIONS.get(state.get('state'), 'Unknown')
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            self._create_backup()
        except Exception as e:
            print(f"Warning: Could not save state: {e}")

    def _create_backup(self):
        try:
            import shutil
            shutil.copy2(self.state_file, self.backup_file)
        except Exception:
            pass

    def load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            return state
        except Exception:
            return self._get_default_state()

    def update_state(self, updates):
        current = self.load_state()
        current.update(updates)
        self._save_state(current)
        return current

    def get(self, key, default=None):
        state = self.load_state()
        return state.get(key, default)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_portfolio_name(api_key):
    for client in API_CLIENTS:
        if client["api_key"] == api_key:
            return client["name"]
    return "Unknown Portfolio"

def fetch_current_positions(api):
    try:
        positions = api.list_positions()
        return {"cash": float(api.get_account().cash), "other_stocks": {p.symbol: float(p.qty) for p in positions}}
    except:
        return portfolio_template

def calculate_portfolio_value(portfolio, api, include_mercury=True):
    total = float(portfolio["cash"])
    for ticker, qty in portfolio["other_stocks"].items():
        price = fetch_real_time_price(api, ticker)
        total += float(qty) * price if price else 0
    if include_mercury and MERCURY_BANK_ENABLED:
        mercury_balance = get_mercury_balance()
        total += mercury_balance
    return total if total > 0 else 1

def build_holdings(portfolio, portfolio_value, api):
    holdings = {}
    for ticker, qty in portfolio["other_stocks"].items():
        price = fetch_real_time_price(api, ticker)
        if price:
            value = qty * price
            holdings[ticker] = {'qty': qty, 'value': value, 'weight': (value / portfolio_value) * 100}
    return holdings

def get_mercury_balance():
    if not MERCURY_BANK_ENABLED:
        return 0
    try:
        headers = {"Authorization": f"Bearer {MERCURY_API_TOKEN}", "Content-Type": "application/json"}
        response = requests.get(f"{MERCURY_API_BASE}/accounts", headers=headers, timeout=10)
        if response.status_code == 200:
            accounts_data = response.json()
            total_balance = 0
            if "accounts" in accounts_data:
                for account in accounts_data["accounts"]:
                    if account.get("status") == "active":
                        available = float(account.get("availableBalance", 0))
                        total_balance += available
            print(f"[OK] Mercury Bank balance: ${total_balance:,.2f}")
            return total_balance
        else:
            return 0
    except Exception as e:
        return 0

def verify_api_connection(api, api_key):
    try:
        account = api.get_account()
        print(f"[OK] API connection successful - Cash: ${float(account.cash):.2f}")
        return True
    except Exception as e:
        print(f"[X] API connection failed: {e}")
        return False

def is_market_open_today():
    if TEST_MODE:
        print(f"[TEST MODE] Market check bypassed - treating as open")
        return True

    try:
        nyse = mcal.get_calendar('NYSE')
        today = datetime.now().strftime('%Y-%m-%d')
        schedule = nyse.schedule(start_date=today, end_date=today)
        is_trading_day = len(schedule) > 0
        if not is_trading_day:
            day_of_week = datetime.now().weekday()
            if day_of_week >= 5:
                print(f"Market closed: Weekend")
            else:
                print(f"Market closed: Holiday")
        else:
            print(f"Trading day confirmed")
        return is_trading_day
    except Exception as e:
        return datetime.now().weekday() < 5

def is_market_open_now():
    if TEST_MODE:
        print(f"[TEST MODE] Market hours check bypassed - treating as open")
        return True

    pacific = timezone('America/Los_Angeles')
    now = datetime.now(pacific)
    market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
    if not is_market_open_today():
        return False
    return market_open <= now <= market_close

def is_market_open():
    if TEST_MODE:
        return True
    return is_market_open_now()

def send_email_with_attachment(subject, body, attachment_path=None):
    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = EMAIL_SENDER, EMAIL_RECEIVER, subject
        msg.attach(MIMEText(body, 'html'))
        if attachment_path and os.path.isfile(attachment_path):
            with open(attachment_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
            msg.attach(part)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email sent.")
    except Exception as e:
        print(f"Email failed: {e}")

def cancel_open_orders(api, symbol=None):
    try:
        for order in api.list_orders(status="open"):
            if symbol is None or order.symbol == symbol:
                api.cancel_order(order.id)
    except:
        pass

def fetch_real_time_price(api, ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            return api.get_latest_trade(ticker).price
        except:
            if attempt < max_retries - 1 and not TEST_MODE:
                time.sleep(1)
    return None

def fetch_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                df = df.rename(columns={c: f"{ticker}_{c}" for c in ["Open", "Close", "High", "Low"]})
                data[ticker] = df
                print(f"✓ Downloaded {ticker}")
            else:
                print(f"⚠️ Empty data for {ticker}")
        except Exception as e:
            print(f"⚠️ Failed to download {ticker}: {e}")
    return data

def combine_data(data):
    combined = pd.DataFrame()
    for ticker, df in data.items():
        if not df.empty:
            combined = df if combined.empty else combined.join(df, how="outer")
    if not combined.empty:
        combined.columns = [f"{col[0]}" for col in combined.columns]
    return combined

def calculate_limit_price(price, side, margin=0.0002):
    return round(price * (1 - margin) if side == "sell" else price * (1 + margin), 2)


# ============================================================================
# HEDGE CALCULATION FUNCTIONS
# ============================================================================

def calculate_actual_etf_values(portfolio, api):
    etf_values = {}
    for ticker in buy_tickers.keys():
        if ticker not in HEDGE_TICKERS_LIST and ticker != "SGOV":
            qty = float(portfolio["other_stocks"].get(ticker, 0))
            if qty > 0:
                price = fetch_real_time_price(api, ticker)
                if price:
                    etf_values[ticker] = qty * price
    return etf_values

def calculate_hedge_targets(alton_portfolio_value, monthly_allocation, etf_values, hedge_groups, target_allocations):
    hedge_total = 1 + monthly_allocation
    total_etf_value = sum(etf_values.values())

    if monthly_allocation > 0:
        threshold_pct = monthly_allocation * HEDGE_THRESHOLD_MULTIPLIER
        threshold_value = alton_portfolio_value * threshold_pct
    else:
        threshold_pct = 0
        threshold_value = 0

    print(f"\n[HEDGE] Monthly Allocation: {monthly_allocation*100:.1f}%")
    print(f"[HEDGE] Threshold ({HEDGE_THRESHOLD_MULTIPLIER*100:.0f}% of monthly): {threshold_pct*100:.1f}% (${threshold_value:,.2f})")
    print(f"[HEDGE] ETFs Held: ${total_etf_value:,.2f}")

    if monthly_allocation == 0:
        use_actual = False
        etf_values_to_use = {}
        total_etf_value = 0
        print(f"[HEDGE] Monthly allocation is 0% - no hedge needed")
    elif total_etf_value < threshold_value:
        use_actual = False
        etf_values_to_use = {ticker: alton_portfolio_value * weight for ticker, weight in target_allocations.items() if weight > 0}
        total_etf_value = sum(etf_values_to_use.values())
        print(f"[HEDGE] ETFs held below {threshold_pct*100:.1f}% threshold")
        print(f"[HEDGE] Using minimum hedge: {monthly_allocation*100:.0f}% on target allocations")
    else:
        use_actual = True
        etf_values_to_use = etf_values
        total_etf_value = sum(etf_values_to_use.values())
        print(f"[HEDGE] ETFs held above {threshold_pct*100:.1f}% threshold")
        print(f"[HEDGE] Using actual holdings with {hedge_total*100:.0f}% hedge total")

    if total_etf_value == 0:
        targets = {}
        for hedge_ticker in hedge_groups.keys():
            targets[hedge_ticker] = {
                "target_pct": 0,
                "target_value": 0,
                "group_tickers": [],
                "group_value": 0,
                "description": "No ETFs",
                "use_actual": use_actual,
                "min_hedge": not use_actual and monthly_allocation > 0
            }
        return targets, total_etf_value, use_actual

    targets = {}
    grouped_tickers = set()

    for hedge_ticker, group_config in hedge_groups.items():
        if not group_config["enabled"]:
            continue

        group_tickers = group_config["tickers"]
        group_value = sum(etf_values_to_use.get(t, 0) for t in group_tickers)
        grouped_tickers.update(group_tickers)

        if use_actual:
            target_pct = hedge_total * (group_value / total_etf_value)
        else:
            target_pct = monthly_allocation * (group_value / total_etf_value)

        target_value = alton_portfolio_value * target_pct

        targets[hedge_ticker] = {
            "target_pct": target_pct,
            "target_value": target_value,
            "group_tickers": group_tickers,
            "group_value": group_value,
            "description": group_config["description"],
            "use_actual": use_actual,
            "min_hedge": not use_actual and monthly_allocation > 0
        }

    ungrouped_tickers = set(etf_values_to_use.keys()) - grouped_tickers
    ungrouped_value = sum(etf_values_to_use.get(t, 0) for t in ungrouped_tickers)

    if ungrouped_value > 0 and "SH" in targets:
        sh_target = targets["SH"]
        if use_actual:
            ungrouped_target_pct = hedge_total * (ungrouped_value / total_etf_value)
        else:
            ungrouped_target_pct = monthly_allocation * (ungrouped_value / total_etf_value)
        ungrouped_target_value = alton_portfolio_value * ungrouped_target_pct

        sh_target["target_pct"] += ungrouped_target_pct
        sh_target["target_value"] += ungrouped_target_value
        sh_target["group_tickers"] = sh_target.get("group_tickers", []) + list(ungrouped_tickers)
        sh_target["group_value"] += ungrouped_value
        sh_target["description"] += " + Others"

    return targets, total_etf_value, use_actual

def print_hedge_summary(hedge_targets, total_etf_value, alton_portfolio_value, hedge_total, execution_day, use_actual, monthly_allocation):
    print(f"\n{'='*60}")
    print(f"  HEDGE CALCULATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Alton Portfolio Value: ${alton_portfolio_value:,.2f}")
    print(f"  Total ETFs Used: ${total_etf_value:,.2f}")
    if use_actual:
        print(f"  Using: Actual Holdings ({hedge_total*100:.0f}% hedge total)")
    else:
        if monthly_allocation > 0:
            print(f"  Using: Minimum Hedge ({monthly_allocation*100:.0f}% on target allocations)")
        else:
            print(f"  Using: No Hedge (monthly allocation is 0%)")
    print(f"  Day: {execution_day}/3")
    print(f"\n  {'Ticker':<8} {'Group':<20} {'Group Value':<15} {'Target %':<12} {'Target $':<15} {'Day {execution_day} $':<15}")
    print(f"  {'-'*80}")

    for ticker, target in hedge_targets.items():
        group_desc = target["description"][:20]
        group_value = target["group_value"]
        target_pct = target["target_pct"] * 100
        target_value = target["target_value"]

        if execution_day == 1:
            day_value = target_value / 3
        elif execution_day == 2:
            day_value = target_value * 2 / 3
        else:
            day_value = target_value

        min_hedge_tag = " (min)" if target.get("min_hedge", False) else ""
        print(f"  {ticker:<8} {group_desc:<20} ${group_value:>12,.2f}  {target_pct:>6.2f}%     ${target_value:>12,.2f}   ${day_value:>12,.2f}{min_hedge_tag}")

    print(f"{'='*60}\n")


# ============================================================================
# EMAIL FUNCTIONS
# ============================================================================

EMAIL_CSS = """
<style>
body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:0;padding:0;background:#f4f4f4}
.container{max-width:750px;margin:0 auto;background:#fff}
.header{background:linear-gradient(135deg,#1a237e 0%,#283593 100%);color:#fff;padding:25px 30px}
.header h1{margin:0;font-size:24px;font-weight:600}
.header .subtitle{color:#b3e5fc;font-size:14px;margin-top:5px}
.header .timestamp{color:#90caf9;font-size:12px;margin-top:10px}
.content{padding:25px 30px}
.signal-box{border-radius:8px;padding:20px;margin:15px 0;text-align:center}
.signal-buy{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);border-left:5px solid #4caf50}
.signal-sell{background:linear-gradient(135deg,#ffebee 0%,#ffcdd2 100%);border-left:5px solid #f44336}
.signal-neutral{background:linear-gradient(135deg,#fff3e0 0%,#ffe0b2 100%);border-left:5px solid #ff9800}
.signal-label{font-size:14px;color:#666;text-transform:uppercase;letter-spacing:1px}
.signal-value{font-size:32px;font-weight:700;margin:10px 0}
.signal-buy .signal-value{color:#2e7d32}
.signal-sell .signal-value{color:#c62828}
.signal-neutral .signal-value{color:#ef6c00}
.confidence-bar{height:8px;background:#e0e0e0;border-radius:4px;margin-top:10px;overflow:hidden}
.confidence-fill{height:100%;border-radius:4px}
.confidence-high{background:linear-gradient(90deg,#4caf50,#81c784)}
.confidence-medium{background:linear-gradient(90deg,#ff9800,#ffb74d)}
.confidence-low{background:linear-gradient(90deg,#f44336,#e57373)}
.section{margin:20px 0}
.section-title{font-size:16px;font-weight:600;color:#1a237e;border-bottom:2px solid #e8eaf6;padding-bottom:8px;margin-bottom:15px}
.metrics-grid{display:table;width:100%;border-collapse:collapse}
.metric-row{display:table-row}
.metric-cell{display:table-cell;padding:12px 15px;border-bottom:1px solid #f0f0f0}
.metric-label{color:#666;font-size:13px;width:40%}
.metric-value{font-weight:600;color:#333;text-align:right}
.metric-positive{color:#2e7d32}
.metric-negative{color:#c62828}
.holdings-table{width:100%;border-collapse:collapse;margin:10px 0}
.holdings-table th{background:#f5f5f5;padding:10px;text-align:left;font-size:12px;color:#666;text-transform:uppercase}
.holdings-table td{padding:10px;border-bottom:1px solid #f0f0f0}
.holdings-table .ticker{font-weight:600;color:#1a237e}
.holdings-table .weight{text-align:right}
.trade-item{background:#fafafa;border-radius:6px;padding:12px 15px;margin:8px 0;display:table;width:100%;box-sizing:border-box}
.trade-buy{border-left:4px solid #4caf50}
.trade-sell{border-left:4px solid #f44336}
.trade-ticker{font-weight:600;color:#333;display:table-cell;width:20%}
.trade-details{color:#666;font-size:13px;display:table-cell}
.trade-amount{font-weight:600;text-align:right;display:table-cell;width:25%}
.footer{background:#f5f5f5;padding:20px 30px;text-align:center;border-top:1px solid#e0e0e0}
.footer-text{color:#999;font-size:12px}
.alert-box{background:#fff3e0;border-left:4px solid #ff9800;padding:15px;margin:15px 0;border-radius:0 6px 6px 0}
.alert-title{font-weight:600;color:#e65100;margin-bottom:5px}
.alert-text{color:#666;font-size:13px}
.error-box{background:#ffebee;border-left:4px solid #f44336;padding:15px;margin:15px 0;border-radius:0 6px 6px 0}
.error-title{font-weight:600;color:#c62828;margin-bottom:5px}
.success-box{background:#e8f5e9;border-left:4px solid #4caf50;padding:15px;margin:15px 0;border-radius:0 6px 6px 0}
.success-title{font-weight:600;color:#2e7d32;margin-bottom:5px}
.divider{height:1px;background:#e0e0e0;margin:20px 0}
.sh-allocation{background:linear-gradient(135deg,#fce4ec 0%,#f8bbd9 100%);border-radius:8px;padding:15px;margin:15px 0}
.sh-title{font-weight:600;color:#880e4f;margin-bottom:10px}
.sh-percentage{font-size:24px;font-weight:700;color:#ad1457}
.vmfxx-box{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);border-radius:8px;padding:15px;margin:15px 0;border-left:4px solid #4caf50}
.vmfxx-title{font-weight:600;color:#2e7d32;margin-bottom:10px}
.vmfxx-amount{font-size:24px;font-weight:700;color:#1b5e20}
.vmfxx-yield{font-size:13px;color:#666;margin-top:5px}
.rs-box{background:linear-gradient(135deg,#e8eaf6 0%,#c5cae9 100%);border-radius:8px;padding:15px;margin:15px 0;border-left:4px solid #3f51b5}
.rs-title{font-weight:600;color:#283593;margin-bottom:10px}
.rs-percentage{font-size:24px;font-weight:700;color:#1a237e}
.transparency-box{background:#f5f5f5;border-radius:8px;padding:15px;margin:15px 0;border:1px solid #e0e0e0}
.transparency-title{font-weight:600;color:#1a237e;margin-bottom:10px}
.cycle-progress{background:#f5f5f5;border-radius:8px;padding:15px;margin:15px 0}
.cycle-title{font-size:14px;color:#666;margin-bottom:10px}
.cycle-days{display:table;width:100%}
.cycle-day{display:table-cell;text-align:center;padding:8px}
.cycle-day-circle{width:35px;height:35px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:600;font-size:14px}
.cycle-day-complete{background:#4caf50;color:#fff}
.cycle-day-current{background:#1a237e;color:#fff;box-shadow:0 0 0 3px #c5cae9}
.cycle-day-pending{background:#e0e0e0;color:#999}
.cycle-day-label{font-size:11px;color:#999;margin-top:5px}
.market-pulse{background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);border-radius:8px;padding:20px;margin:20px 0}
.market-pulse-title{font-weight:600;color:#1565c0;margin-bottom:15px;font-size:16px}
.market-indicator{display:table;width:100%;margin:8px 0}
.indicator-name{display:table-cell;color:#666;font-size:13px;width:35%}
.indicator-value{display:table-cell;font-weight:600}
.indicator-change{display:table-cell;text-align:right;font-size:13px;width:25%}
.vix-gauge{text-align:center;margin:15px 0}
.vix-value{font-size:36px;font-weight:700}
.vix-low{color:#4caf50}
.vix-moderate{color:#ff9800}
.vix-high{color:#f44336}
.vix-extreme{color:#9c27b0}
.status-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600}
.status-open{background:#e8f5e9;color:#2e7d32}
.status-closed{background:#ffebee;color:#c62828}
</style>
"""

def create_email_header(title, subtitle=None, email_type="update", portfolio_name="Portfolio"):
    pacific = timezone('America/Los_Angeles')
    timestamp = datetime.now(pacific).strftime('%B %d, %Y at %I:%M %p PDT')
    icons = {"update": "", "error": "", "success": "[OK]", "pre-run": "", "post-run": ""}
    icon = icons.get(email_type, "")
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ''
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">{EMAIL_CSS}</head><body><div class="container"><div class="header"><h1>{icon} {portfolio_name}: {title}</h1>{subtitle_html}<div class="timestamp">{timestamp}</div></div><div class="content">"""

def create_email_footer():
    return """</div><div class="footer"><div class="footer-text">Alton RS Hybrid Model - Automated Portfolio Management</div></div></div></body></html>"""

def create_market_status_badge(is_open):
    return '<span class="status-badge status-open">Market Open</span>' if is_open else '<span class="status-badge status-closed">Market Closed</span>'

def create_alert_box(title, message, alert_type="warning"):
    if alert_type == "error":
        return f'<div class="error-box"><div class="error-title">{title}</div><div class="alert-text">{message}</div></div>'
    elif alert_type == "success":
        return f'<div class="success-box"><div class="success-title">{title}</div><div class="alert-text">{message}</div></div>'
    return f'<div class="alert-box"><div class="alert-title">{title}</div><div class="alert-text">{message}</div></div>'

def create_signal_box(signal, confidence, is_new_cycle=False, transparency_report=None):
    if signal == 1:
        signal_class, signal_text, signal_icon = "signal-buy", "BUY", ""
    elif signal == 0:
        signal_class, signal_text, signal_icon = "signal-sell", "SELL", ""
    else:
        signal_class, signal_text, signal_icon = "signal-neutral", "NEUTRAL", ""
    confidence_pct = confidence * 100
    conf_class = "confidence-high" if confidence >= 0.75 else "confidence-medium" if confidence >= 0.60 else "confidence-low"
    new_badge = '<span class="status-badge" style="background:#e3f2fd;color:#1565c0;margin-left:10px;">NEW CYCLE</span>' if is_new_cycle else ''

    html = f"""<div class="signal-box {signal_class}"><div class="signal-label">Model Signal (Alton Portion: {(1-rs_allocation_percentage)*100:.0f}%) {new_badge}</div><div class="signal-value">{signal_icon} {signal_text}</div><div style="font-size:14px;color:#666;">Confidence: {confidence_pct:.1f}%</div><div class="confidence-bar"><div class="confidence-fill {conf_class}" style="width:{confidence_pct}%;"></div></div></div>"""

    if transparency_report:
        html += f'<div class="transparency-box">{transparency_report}</div>'

    return html

def create_cycle_progress(current_day, total_days, signal, today_progress=None, cycle_status=None):
    days_html = ""
    for day in range(1, total_days + 1):
        if cycle_status == "COMPLETE":
            day_class = "cycle-day-complete"
        elif day < current_day:
            day_class = "cycle-day-complete"
        elif day == current_day:
            day_class = "cycle-day-current"
        else:
            day_class = "cycle-day-pending"
        days_html += f'<div class="cycle-day"><div class="cycle-day-circle {day_class}">{day}</div><div class="cycle-day-label">Day {day}</div></div>'

    if cycle_status == "COMPLETE":
        signal_text = "CYCLE COMPLETE"
        progress_html = '<div style="margin-top:10px;font-size:12px;color:#2e7d32;font-weight:600;">✓ All execution days completed. Waiting for next signal.</div>'
    else:
        signal_text = "Buy" if signal == 1 else "Sell" if signal == 0 else ""
        progress_html = f'<div style="margin-top:10px;font-size:12px;color:#666;">Today\'s progress: {today_progress*100:.1f}%</div>' if today_progress is not None else ''

    return f'<div class="cycle-progress"><div class="cycle-title">Execution Cycle Progress - {signal_text} (Alton Portion)</div><div class="cycle-days">{days_html}</div>{progress_html}</div>'

def create_cash_info_box(cash_amount, sweep_mm_amount, portfolio_value, min_cash, alpaca_cash=0, mercury_balance=0, leveraged_mm_value=0):
    cash_pct = (cash_amount / portfolio_value * 100) if portfolio_value > 0 else 0
    sweep_pct = (sweep_mm_amount / portfolio_value * 100) if portfolio_value > 0 else 0
    mercury_pct = (mercury_balance / portfolio_value * 100) if portfolio_value > 0 else 0
    leveraged_pct = (leveraged_mm_value / portfolio_value * 100) if portfolio_value > 0 else 0
    total_liquid = cash_amount + sweep_mm_amount + mercury_balance
    daily_sweep_earnings = sweep_mm_amount * 0.0345 / 365 if sweep_mm_amount > 0 else 0
    daily_alpaca_earnings = alpaca_cash * 0.0375 / 365 if alpaca_cash > 0 else 0
    daily_mercury_earnings = mercury_balance * 0.0345 / 365 if mercury_balance > 0 else 0
    daily_leveraged_earnings = leveraged_mm_value * 0.0345 / 365 if leveraged_mm_value > 0 else 0
    total_daily_earnings = daily_sweep_earnings + daily_alpaca_earnings + daily_mercury_earnings + daily_leveraged_earnings

    earnings_breakdown = ''
    if sweep_mm_amount > 0:
        earnings_breakdown += f'${daily_sweep_earnings:.2f}/day Sweep MM (3.45%)'
    if leveraged_mm_value > 0:
        if earnings_breakdown: earnings_breakdown += '<br>'
        earnings_breakdown += f'${daily_leveraged_earnings:.2f}/day Leveraged MM (3.45%)'
    if alpaca_cash > 0:
        if earnings_breakdown: earnings_breakdown += '<br>'
        earnings_breakdown += f'${daily_alpaca_earnings:.2f}/day Alpaca Cash (3.75%)'
    if mercury_balance > 0:
        if earnings_breakdown: earnings_breakdown += '<br>'
        earnings_breakdown += f'${daily_mercury_earnings:.2f}/day Mercury Bank (3.45%)'
    if not earnings_breakdown:
        earnings_breakdown = 'No interest-bearing cash'

    leveraged_html = ''
    if leveraged_mm_value > 0:
        leveraged_html = f'''
        <div style="margin:5px 0;background:#e8eaf6;padding:8px;border-radius:4px;">
            Leveraged MM: <strong>${leveraged_mm_value:,.2f}</strong> ({leveraged_pct:.1f}%) <span style="color:#666;">(on margin)</span>
        </div>
        '''

    return f'''
    <div class="vmfxx-box">
        <div class="vmfxx-title">Cash & Liquid Positions</div>
        <div class="vmfxx-amount">${total_liquid:,.2f}</div>
        <div style="font-size:14px;color:#666;margin-top:10px;padding:10px;background:#fff;border-radius:4px;">
            <div style="margin:5px 0;">Alpaca Cash: <strong>${cash_amount:,.2f}</strong> ({cash_pct:.1f}%)</div>
            <div style="margin:5px 0;">Sweep MM: <strong>${sweep_mm_amount:,.2f}</strong> ({sweep_pct:.1f}%)</div>
            {leveraged_html}
            <div style="margin:5px 0;">Mercury Bank: <strong>${mercury_balance:,.2f}</strong> ({mercury_pct:.1f}%)</div>
            <div style="margin:5px 0;border-top:1px solid #ddd;padding-top:5px;">Alpaca Sweep Cash: <strong>${alpaca_cash:,.2f}</strong> (earning 3.75%)</div>
        </div>
        <div class="vmfxx-yield">
            <strong>Daily Earnings:</strong><br>{earnings_breakdown}
            <br><strong>Total: ${total_daily_earnings:.2f}/day</strong>
            <br><br><em>Strategy: {sgov_sweep_percentage*100:.0f}% excess -> Sweep MM, {(1-sgov_sweep_percentage)*100:.0f}% -> Alpaca Sweep</em>
        </div>
    </div>'''

def create_portfolio_metrics(portfolio_value, cash, min_cash, holdings):
    cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 0
    excess_cash = max(0, cash - min_cash)
    holdings_rows = "".join([f'<tr><td class="ticker">{t}</td><td>{d.get("qty",0):.4f}</td><td>${d.get("value",0):,.2f}</td><td class="weight">{d.get("weight",0):.1f}%</td></tr>' for t, d in holdings.items()])
    return f"""<div class="section"><div class="section-title">Portfolio Overview</div><div class="metrics-grid"><div class="metric-row"><div class="metric-cell metric-label">Total Value</div><div class="metric-cell metric-value">${portfolio_value:,.2f}</div></div><div class="metric-row"><div class="metric-cell metric-label">Cash Balance</div><div class="metric-cell metric-value">${cash:,.2f} ({cash_pct:.1f}%)</div></div><div class="metric-row"><div class="metric-cell metric-label">Min Cash Reserve</div><div class="metric-cell metric-value">${min_cash:,.2f}</div></div><div class="metric-row"><div class="metric-cell metric-label">Excess Cash</div><div class="metric-cell metric-value">${excess_cash:,.2f}</div></div></div><table class="holdings-table"><thead><tr><th>Ticker</th><th>Shares</th><th>Value</th><th style="text-align:right;">Weight</th></tr></thead><tbody>{holdings_rows}</tbody></table></div>"""

def create_trade_list(trades):
    if not trades:
        return '<div class="section"><div class="section-title">Trades Executed</div><div class="alert-box"><div class="alert-text">No trades executed this session.</div></div></div>'
    trades_html = "".join([f'<div class="trade-item {"trade-buy" if t["side"]=="buy" else "trade-sell"}"><div class="trade-ticker">{"[+]" if t["side"]=="buy" else "[-]"} {t["ticker"]}</div><div class="trade-details">{t["qty"]:.4f} shares @ ${t["price"]:.2f}</div><div class="trade-amount">${t["total"]:,.2f}</div></div>' for t in trades])
    return f'<div class="section"><div class="section-title">Trades Executed</div>{trades_html}</div>'

def create_sh_allocation_box(month_name, allocation_pct):
    return f'<div class="sh-allocation"><div class="sh-title">Hedge Allocation</div><div style="font-size:13px;color:#666;">{month_name} Target</div><div class="sh-percentage">{allocation_pct*100:.0f}%</div></div>'

def create_rs_allocation_box(rs_pct, rs_stocks_count, is_rebalance_day_flag, rebalance_day_name):
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
    return f'''<div class="rs-box">
        <div class="rs-title">RS Momentum Strategy</div>
        <div class="rs-percentage">{rs_pct*100:.0f}% of Portfolio</div>
        <div style="font-size:13px;color:#666;margin-top:8px;">
            Holding top {rs_stocks_count} momentum stocks from Russell 3000<br>
            Rebalance Day: {day_names.get(rs_rebalance_day, "Unknown")}<br>
            {"<strong style='color:#2e7d32;'>TODAY IS REBALANCE DAY</strong>" if is_rebalance_day_flag else "Next rebalance: " + rebalance_day_name}<br>
            Always invested (not affected by BUY/SELL signals)
        </div>
    </div>'''

def create_strategy_section(target_allocations, exec_days, min_cash_pct, sh_alloc, month_name):
    allocs = "".join([f'<div class="metric-row"><div class="metric-cell metric-label">{t} (Alton)</div><div class="metric-cell metric-value">{w*(1-rs_allocation_percentage)*100:.1f}%</div></div>' for t, w in target_allocations.items() if w > 0])
    mm_sweep_text = "Enabled" if MONEY_MARKET_SWEEP_ENABLED else "Disabled"
    mm_leverage_text = f"Enabled ({MONEY_MARKET_LEVERAGE_PERCENT*100:.0f}%)" if MONEY_MARKET_LEVERAGE_ENABLED else "Disabled"
    hedge_mode_text = "Enabled (Leverage Hedge)" if HEDGE_MODE_ENABLED else "Disabled (Original Strategy)"

    hedge_groups_text = ""
    if HEDGE_MODE_ENABLED:
        for ticker, config in HEDGE_GROUPS.items():
            if config["enabled"]:
                tickers_str = ", ".join(config["tickers"])
                hedge_groups_text += f"<br>  - {ticker}: {config['description']} ({tickers_str})"

    return f"""<div class="section"><div class="section-title">Strategy Configuration</div><div class="metrics-grid">{allocs}<div class="metric-row"><div class="metric-cell metric-label">RS Momentum</div><div class="metric-cell metric-value">{rs_allocation_percentage*100:.1f}%</div></div><div class="metric-row"><div class="metric-cell metric-label">Minimum Cash</div><div class="metric-cell metric-value">{min_cash_pct*100:.1f}%</div></div><div class="metric-row"><div class="metric-cell metric-label">Hedge Mode</div><div class="metric-cell metric-value">{hedge_mode_text}</div></div><div class="metric-row"><div class="metric-cell metric-label">Hedge Groups</div><div class="metric-cell metric-value" style="text-align:left;font-size:12px;">{hedge_groups_text}</div></div><div class="metric-row"><div class="metric-cell metric-label">MM Sweep</div><div class="metric-cell metric-value">{mm_sweep_text}</div></div><div class="metric-row"><div class="metric-cell metric-label">MM Leverage</div><div class="metric-cell metric-value">{mm_leverage_text}</div></div><div class="metric-row"><div class="metric-cell metric-label">Execution Period</div><div class="metric-cell metric-value">{exec_days} days</div></div><div class="metric-row"><div class="metric-cell metric-label">Monthly Hedge</div><div class="metric-cell metric-value">{sh_alloc*100:.1f}%</div></div></div></div>"""

def create_market_pulse_section(market_data):
    vix = market_data.get('vix', 0)
    vix_change = market_data.get('vix_change', 0)
    spy_change = market_data.get('spy_change', 0)
    qqq_change = market_data.get('qqq_change', 0)
    tlt_change = market_data.get('tlt_change', 0)
    gld_change = market_data.get('gld_change', 0)
    btc_change = market_data.get('btc_change', 0)
    oil_change = market_data.get('oil_change', 0)
    dxy_value = market_data.get('dxy', 0)
    top_stories = market_data.get('top_stories', [])
    summary = market_data.get('summary', '')
    fear_greed = market_data.get('fear_greed', 50)

    vix_class = "vix-low" if vix < 15 else "vix-moderate" if vix < 20 else "vix-high" if vix < 30 else "vix-extreme"
    vix_label = "Low Volatility" if vix < 15 else "Moderate" if vix < 20 else "Elevated" if vix < 30 else "Extreme"

    def fmt_chg(val):
        if val > 0: return f'<span class="metric-positive">+{val:.2f}%</span>'
        elif val < 0: return f'<span class="metric-negative">{val:.2f}%</span>'
        return f'{val:.2f}%'

    stories_html = ""
    for story in top_stories[:6]:
        title = story.get('title', '')
        summary_text = story.get('summary', '')
        source = story.get('source', '')
        time_str = story.get('time', '')
        time_badge = f'<span style="color:#999;font-size:11px;margin-left:8px;">{time_str}</span>' if time_str else ''
        summary_para = f'<p style="margin:8px 0 0 0;font-size:13px;color:#555;line-height:1.5;">{summary_text}</p>' if summary_text else ''
        stories_html += f'''<div style="padding:15px 0;border-bottom:1px solid #e0e0e0;"><div style="font-size:11px;color:#1565c0;text-transform:uppercase;font-weight:600;">{source}{time_badge}</div><div style="font-size:15px;color:#222;font-weight:600;margin-top:5px;line-height:1.4;">{title}</div>{summary_para}</div>'''
    if not stories_html:
        stories_html = '<div style="color:#666;font-size:13px;padding:15px 0;">No market stories available at this time.</div>'

    fg_label = 'Extreme Fear' if fear_greed < 25 else 'Fear' if fear_greed < 45 else 'Neutral' if fear_greed < 55 else 'Greed' if fear_greed < 75 else 'Extreme Greed'
    fg_color = '#c62828' if fear_greed < 25 else '#ef6c00' if fear_greed < 45 else '#666' if fear_greed < 55 else '#2e7d32' if fear_greed < 75 else '#1b5e20'

    ai_summary_html = ''
    if summary:
        ai_summary_html = f'<div style="background:#fff;border-radius:6px;padding:15px;margin:15px 0;border-left:4px solid #1565c0;"><div style="font-size:12px;color:#1565c0;font-weight:600;margin-bottom:8px;">AI MARKET SUMMARY</div><div style="font-size:14px;color:#333;line-height:1.6;">{summary}</div></div>'

    return f"""
    <div class="market-pulse">
        <div class="market-pulse-title">Market Pulse</div>
        <div class="vix-gauge">
            <div style="font-size:12px;color:#666;text-transform:uppercase;">VIX Fear Index</div>
            <div class="vix-value {vix_class}">{vix:.1f}</div>
            <div style="font-size:13px;color:#666;">{vix_label} - {fmt_chg(vix_change)} today</div>
        </div>
        <div style="display:table;width:100%;margin:15px 0;background:#fff;border-radius:6px;padding:10px;">
            <div style="display:table-cell;text-align:center;padding:8px;border-right:1px solid #eee;"><div style="font-size:11px;color:#666;">Fear & Greed</div><div style="font-size:20px;font-weight:700;color:{fg_color};">{fear_greed}</div><div style="font-size:10px;color:#999;">{fg_label}</div></div>
            <div style="display:table-cell;text-align:center;padding:8px;border-right:1px solid #eee;"><div style="font-size:11px;color:#666;">S&P 500</div><div style="font-size:16px;font-weight:600;">{fmt_chg(spy_change)}</div></div>
            <div style="display:table-cell;text-align:center;padding:8px;border-right:1px solid #eee;"><div style="font-size:11px;color:#666;">Nasdaq</div><div style="font-size:16px;font-weight:600;">{fmt_chg(qqq_change)}</div></div>
            <div style="display:table-cell;text-align:center;padding:8px;border-right:1px solid #eee;"><div style="font-size:11px;color:#666;">Bonds</div><div style="font-size:16px;font-weight:600;">{fmt_chg(tlt_change)}</div></div>
            <div style="display:table-cell;text-align:center;padding:8px;"><div style="font-size:11px;color:#666;">Gold</div><div style="font-size:16px;font-weight:600;">{fmt_chg(gld_change)}</div></div>
        </div>
        <div style="display:table;width:100%;margin-bottom:15px;">
            <div class="market-indicator"><div class="indicator-name">Bitcoin</div><div class="indicator-change">{fmt_chg(btc_change)}</div></div>
            <div class="market-indicator"><div class="indicator-name">Crude Oil</div><div class="indicator-change">{fmt_chg(oil_change)}</div></div>
            <div class="market-indicator"><div class="indicator-name">US Dollar (DXY)</div><div class="indicator-change">{dxy_value:.2f}</div></div>
        </div>
        {ai_summary_html}
        <div style="margin-top:20px;"><div style="font-weight:600;color:#1565c0;margin-bottom:5px;font-size:16px;">Today's Top Stories</div><div style="font-size:12px;color:#888;margin-bottom:10px;">What's moving the markets today</div>{stories_html}</div>
    </div>"""

def fetch_market_data():
    global cached_market_data
    if cached_market_data["data"] and cached_market_data["timestamp"]:
        if (datetime.now() - cached_market_data["timestamp"]).total_seconds() < 900:
            return cached_market_data["data"]

    market_data = {'vix': 0, 'vix_change': 0, 'spy_change': 0, 'qqq_change': 0, 'tlt_change': 0,
                   'gld_change': 0, 'btc_change': 0, 'oil_change': 0, 'dxy': 0, 'fear_greed': 50,
                   'headlines': [], 'summary': '', 'top_stories': []}

    tickers_map = {'^VIX': 'vix', 'SPY': 'spy', 'QQQ': 'qqq', 'TLT': 'tlt', 'GLD': 'gld', 'BTC-USD': 'btc', 'USO': 'oil', 'UUP': 'dxy'}
    for ticker, key in tickers_map.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                current, previous = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                change = ((current - previous) / previous) * 100
                if key == 'vix':
                    market_data['vix'], market_data['vix_change'] = current, change
                elif key == 'dxy':
                    market_data['dxy'] = current
                else:
                    market_data[f'{key}_change'] = change
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    try:
        ua = UserAgent()
        user_agent = ua.random
    except:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.5"}
    top_stories = []

    try:
        resp = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            market_data['fear_greed'] = int(data.get('fear_and_greed', {}).get('score', 50))
    except:
        pass

    try:
        spy = yf.Ticker("SPY")
        news = spy.news
        if news:
            for item in news[:6]:
                title = item.get('title', '')
                if title and len(title) > 20:
                    top_stories.append({"source": item.get('publisher', 'Market News'), "title": title, "summary": item.get('summary', '')[:200], "time": ""})
    except:
        pass

    market_data['top_stories'] = top_stories if top_stories else [{"source": "Market Data", "title": "Market data unavailable", "summary": "", "time": ""}]
    market_data['headlines'] = [{"source": s["source"], "headline": s["title"]} for s in top_stories[:12]]
    if top_stories and len(top_stories) > 2:
        market_data['summary'] = f"Market moving on {len(top_stories)} key stories today."

    cached_market_data = {"data": market_data, "timestamp": datetime.now()}
    return market_data

def fetch_market_pulse(date, signal_today=None):
    return fetch_market_data()


# ============================================================================
# MONITOR THREAD FUNCTIONS
# ============================================================================

def monitor_orders_loop(api):
    global monitor_thread_running

    print(f"[Monitor] Starting single monitor thread with price adjustment")
    order_attempts = {}

    while monitor_thread_running:
        try:
            with order_lock:
                if not active_orders:
                    time.sleep(2)
                    continue

                current_prices = {}
                for order_id, order_info in active_orders.items():
                    ticker = order_info['ticker']
                    if ticker not in current_prices:
                        price = fetch_real_time_price(api, ticker)
                        if price:
                            current_prices[ticker] = price

                open_orders = api.list_orders(status='open', limit=200)
                open_order_dict = {order.id: order for order in open_orders}

                for order_id, order_info in list(active_orders.items()):
                    ticker = order_info['ticker']
                    side = order_info['side']
                    original_qty = order_info['qty']
                    strategy_id = order_info['strategy_id']

                    current_price = current_prices.get(ticker)
                    if not current_price:
                        continue

                    attempt = order_attempts.get(order_id, 1)

                    if order_id in open_order_dict:
                        open_order = open_order_dict[order_id]
                        current_limit = float(open_order.limit_price) if open_order.limit_price else order_info['price']

                        if side == 'buy':
                            base_limit = round(current_price * 1.0002, 2)
                        else:
                            base_limit = round(current_price * 0.9998, 2)

                        if attempt <= 3:
                            new_limit = base_limit
                        else:
                            adjustment = 0.0005 * min(attempt - 3, 15)
                            if side == 'buy':
                                new_limit = round(base_limit * (1 + adjustment), 2)
                            else:
                                new_limit = round(base_limit * (1 - adjustment), 2)

                        if abs(new_limit - current_limit) > 0.01:
                            print(f"  🔄 [{strategy_id}] Adjusting {ticker} {side}: ${current_limit:.2f} -> ${new_limit:.2f} (market: ${current_price:.2f}, attempt: {attempt})")

                            try:
                                api.cancel_order(order_id)
                                time.sleep(0.3)

                                # Only re-submit what has not filled yet, so a partial fill is never bought/sold twice
                                try:
                                    already_filled = float(api.get_order(order_id).filled_qty or 0)
                                except Exception:
                                    already_filled = float(open_order.filled_qty or 0)
                                remaining_qty = round(original_qty - already_filled, 6)
                                if remaining_qty <= 0.0001:
                                    del active_orders[order_id]
                                    if order_id in order_attempts:
                                        del order_attempts[order_id]
                                    continue

                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                client_id = f"{strategy_id}_ADJ_{ticker}_{timestamp}_{attempt}"

                                new_order = api.submit_order(
                                    symbol=ticker,
                                    qty=remaining_qty,
                                    side=side,
                                    type='limit',
                                    limit_price=new_limit,
                                    time_in_force='day',
                                    client_order_id=client_id
                                )

                                order_attempts[new_order.id] = attempt + 1
                                del active_orders[order_id]
                                active_orders[new_order.id] = {
                                    'ticker': ticker,
                                    'qty': remaining_qty,
                                    'side': side,
                                    'price': new_limit,
                                    'client_id': client_id,
                                    'strategy_id': strategy_id,
                                    'submitted_at': datetime.now()
                                }

                                if order_id in order_attempts:
                                    del order_attempts[order_id]

                            except Exception as e:
                                print(f"  ❌ [{strategy_id}] Failed to adjust {ticker}: {e}")
                        else:
                            if attempt % 6 == 0:
                                print(f"  ⏳ [{strategy_id}] {ticker} {side}: waiting at ${current_limit:.2f} (market: ${current_price:.2f}, attempt: {attempt})")

                    elif order_id not in open_order_dict:
                        try:
                            filled_order = api.get_order(order_id)
                            if filled_order.status == 'filled':
                                filled_qty = float(filled_order.filled_qty)
                                filled_price = float(filled_order.filled_avg_price) if filled_order.filled_avg_price else order_info.get('price', 0)
                                print(f"  ✅ ORDER FILLED: [{strategy_id}] {ticker} - {filled_qty:.4f} @ ${filled_price:.2f}")
                            elif filled_order.status == 'canceled':
                                print(f"  ⚠️ ORDER CANCELLED: [{strategy_id}] {ticker}")
                        except Exception as e:
                            print(f"  ⚠️ Error checking order {order_id}: {e}")

                        if order_id in order_attempts:
                            del order_attempts[order_id]
                        if order_id in active_orders:
                            del active_orders[order_id]

            time.sleep(5)

        except Exception as e:
            print(f"[Monitor] Error in monitor loop: {e}")
            time.sleep(5)

def start_monitor_thread(api):
    global monitor_thread_running, monitor_thread_instance
    monitor_thread_running = True

    monitor_thread_instance = threading.Thread(
        target=monitor_orders_loop,
        args=(api,),
        daemon=True
    )
    monitor_thread_instance.start()
    print(f"[Monitor] Monitor thread started")
    return monitor_thread_instance

def stop_monitor_thread():
    global monitor_thread_running
    monitor_thread_running = False
    print(f"[Monitor] Monitor thread stopping")


# ============================================================================
# ORDER SUBMISSION FUNCTIONS
# ============================================================================

def submit_order_no_polling(api, ticker, qty, side, limit_price, strategy_id):
    if qty <= 0.0001:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    client_id = f"{strategy_id}_{side}_{ticker}_{timestamp}_{unique_id}"

    try:
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type='limit',
            limit_price=limit_price,
            time_in_force='day',
            client_order_id=client_id
        )

        with order_lock:
            active_orders[order.id] = {
                'ticker': ticker,
                'qty': qty,
                'side': side,
                'price': limit_price,
                'client_id': client_id,
                'strategy_id': strategy_id,
                'submitted_at': datetime.now()
            }

        print(f"  📝 [{strategy_id}] {ticker} {side} {qty:.4f} @ ${limit_price:.2f}")
        return order

    except Exception as e:
        print(f"  ❌ [{strategy_id}] {ticker} failed: {e}")
        return None

def execute_orders(api, orders, side, strategy_id):
    if not orders:
        return []

    print(f"\n{'='*60}")
    print(f"📊 [{strategy_id}] EXECUTING {len(orders)} {side.upper()} ORDERS")
    print(f"{'='*60}")

    results = []

    for i, order in enumerate(orders, 1):
        ticker = order['ticker']
        target_qty = order['qty']

        if target_qty <= 0.0001:
            continue

        current_price = fetch_real_time_price(api, ticker)
        if not current_price:
            print(f"  ⚠️ {ticker}: Could not get price - skipping")
            continue

        if side == 'buy':
            limit_price = calculate_limit_price(current_price, 'buy')
        else:
            limit_price = calculate_limit_price(current_price, 'sell')

        submitted_order = submit_order_no_polling(api, ticker, target_qty, side, limit_price, strategy_id)

        if submitted_order:
            results.append({
                'ticker': ticker,
                'side': side,
                'qty': target_qty,
                'price': limit_price,
                'total': target_qty * limit_price,
                'order_id': submitted_order.id
            })

        if i < len(orders):
            time.sleep(0.5)

    print(f"\n  📊 Submitted {len(results)} of {len(orders)} orders")
    return results


# ============================================================================
# MONEY MARKET MANAGEMENT FUNCTIONS
# ============================================================================

def get_money_market_positions(portfolio, api):
    mm_positions = {}
    all_mm_tickers = set(MONEY_MARKET_SWEEP_TICKERS + MONEY_MARKET_LEVERAGE_TICKERS)

    for ticker in all_mm_tickers:
        qty = float(portfolio["other_stocks"].get(ticker, 0))
        price = fetch_real_time_price(api, ticker)
        if price:
            mm_positions[ticker] = {'qty': qty, 'price': price, 'value': qty * price}
        else:
            mm_positions[ticker] = {'qty': qty, 'price': 0, 'value': 0}

    return mm_positions

def get_total_money_market_value(portfolio, api):
    all_mm_tickers = set(MONEY_MARKET_SWEEP_TICKERS + MONEY_MARKET_LEVERAGE_TICKERS)
    total = 0
    for ticker in all_mm_tickers:
        qty = float(portfolio["other_stocks"].get(ticker, 0))
        price = fetch_real_time_price(api, ticker)
        if price:
            total += qty * price
    return total

def get_leveraged_money_market_value(portfolio, api):
    total = 0
    for ticker in MONEY_MARKET_LEVERAGE_TICKERS:
        qty = float(portfolio["other_stocks"].get(ticker, 0))
        price = fetch_real_time_price(api, ticker)
        if price:
            total += qty * price
    return total

def liquidate_all_sgov(api, portfolio, executed_trades):
    sgov_qty = float(portfolio["other_stocks"].get("SGOV", 0))
    if sgov_qty <= 0:
        return portfolio, executed_trades

    print(f"\n[HEDGE MODE] Liquidating SGOV position: {sgov_qty:.4f} shares")

    sell_orders = [{'ticker': 'SGOV', 'qty': sgov_qty}]
    results = execute_orders(api, sell_orders, 'sell', MM_STRATEGY_ID)

    for fill in results:
        portfolio["cash"] += fill['total']
        portfolio["other_stocks"]['SGOV'] -= fill['qty']
        if portfolio["other_stocks"]['SGOV'] <= 0.01:
            del portfolio["other_stocks"]['SGOV']
        executed_trades.append(fill)
        executed_trades[-1]['note'] = 'SGOV-LIQUIDATION'

    print(f"  Liquidated ${sum(f['total'] for f in results):,.2f} of SGOV")
    return portfolio, executed_trades

def manage_money_market_sweep(api, portfolio, portfolio_value, min_cash, executed_trades):
    if HEDGE_MODE_ENABLED:
        portfolio, executed_trades = liquidate_all_sgov(api, portfolio, executed_trades)
        return portfolio, executed_trades

    if not MONEY_MARKET_SWEEP_ENABLED:
        return portfolio, executed_trades

    mm_trades = []
    current_cash = float(portfolio["cash"])
    mm_positions = get_money_market_positions(portfolio, api)

    sweep_mm_value = sum(pos['value'] for ticker, pos in mm_positions.items()
                          if ticker in MONEY_MARKET_SWEEP_TICKERS)

    excess_cash = current_cash - min_cash

    target_sweep_value = excess_cash * sgov_sweep_percentage if excess_cash > 0 else 0

    if abs(sweep_mm_value - target_sweep_value) < 500 and target_sweep_value > 0:
        print(f"  MM Sweep already at target (${sweep_mm_value:,.0f} / ${target_sweep_value:,.0f}) - no action")
        return portfolio, executed_trades

    if excess_cash > 100:
        cash_to_deploy = excess_cash * sgov_sweep_percentage
        if cash_to_deploy > 100:
            if abs(sweep_mm_value - cash_to_deploy) < 100:
                print(f"  MM Sweep already at target (${sweep_mm_value:,.0f} / ${cash_to_deploy:,.0f}) - no action")
                return portfolio, executed_trades

            num_tickers = len(MONEY_MARKET_SWEEP_TICKERS)
            per_ticker_value = cash_to_deploy / num_tickers

            buy_orders = []
            for ticker in MONEY_MARKET_SWEEP_TICKERS:
                target_value = per_ticker_value
                current_value = mm_positions.get(ticker, {}).get('value', 0)
                needed_value = target_value - current_value

                if needed_value > 100:
                    price = fetch_real_time_price(api, ticker)
                    if price:
                        qty = needed_value / price
                        if qty > 0.01:
                            buy_orders.append({'ticker': ticker, 'qty': qty})
                            print(f"  Sweep MM: Buy ${needed_value:,.2f} of {ticker}")

            if buy_orders:
                results = execute_orders(api, buy_orders, 'buy', MM_STRATEGY_ID)
                for fill in results:
                    portfolio["other_stocks"][fill['ticker']] = portfolio["other_stocks"].get(fill['ticker'], 0) + fill['qty']
                    portfolio["cash"] -= fill['total']
                    executed_trades.append(fill)
                    mm_trades.append(fill)

    elif excess_cash < -100:
        cash_needed = abs(excess_cash)

        sell_orders = []
        for ticker in MONEY_MARKET_SWEEP_TICKERS:
            current_value = mm_positions.get(ticker, {}).get('value', 0)
            if current_value > 0:
                price = fetch_real_time_price(api, ticker)
                if price:
                    qty = min(current_value / price, (cash_needed / price) * (current_value / sweep_mm_value) if sweep_mm_value > 0 else 0)
                    if qty > 0.01:
                        sell_orders.append({'ticker': ticker, 'qty': qty})
                        print(f"  Sweep MM: Sell ${qty * price:,.2f} of {ticker}")

        if sell_orders:
            results = execute_orders(api, sell_orders, 'sell', MM_STRATEGY_ID)
            for fill in results:
                portfolio["cash"] += fill['total']
                portfolio["other_stocks"][fill['ticker']] -= fill['qty']
                if portfolio["other_stocks"][fill['ticker']] <= 0.01:
                    del portfolio["other_stocks"][fill['ticker']]
                executed_trades.append(fill)
                mm_trades.append(fill)

    return portfolio, executed_trades

def manage_money_market_leverage(api, portfolio, portfolio_value, executed_trades):
    if HEDGE_MODE_ENABLED:
        return portfolio, executed_trades

    if not MONEY_MARKET_LEVERAGE_ENABLED:
        return portfolio, executed_trades

    target_leveraged_value = portfolio_value * MONEY_MARKET_LEVERAGE_PERCENT

    if target_leveraged_value < MONEY_MARKET_LEVERAGE_MIN_POSITION:
        print(f"  Leveraged MM: Target ${target_leveraged_value:,.2f} below minimum ${MONEY_MARKET_LEVERAGE_MIN_POSITION:,.2f} - skipping")
        return portfolio, executed_trades

    mm_positions = get_money_market_positions(portfolio, api)
    current_leveraged_value = sum(pos['value'] for ticker, pos in mm_positions.items()
                                   if ticker in MONEY_MARKET_LEVERAGE_TICKERS)

    drift = current_leveraged_value - target_leveraged_value
    drift_pct = drift / target_leveraged_value if target_leveraged_value > 0 else 0

    print(f"\n[Leveraged MM] Target: ${target_leveraged_value:,.2f} | Current: ${current_leveraged_value:,.2f} | Drift: {drift_pct:+.1%}")

    if abs(drift_pct) <= MONEY_MARKET_LEVERAGE_REBALANCE_THRESHOLD:
        print(f"  Leveraged MM: Within {MONEY_MARKET_LEVERAGE_REBALANCE_THRESHOLD*100:.0f}% tolerance - no action")
        return portfolio, executed_trades

    num_tickers = len(MONEY_MARKET_LEVERAGE_TICKERS)
    per_ticker_target = target_leveraged_value / num_tickers

    buy_orders = []
    sell_orders = []

    for ticker in MONEY_MARKET_LEVERAGE_TICKERS:
        current_value = mm_positions.get(ticker, {}).get('value', 0)
        target_value = per_ticker_target
        needed_value = target_value - current_value

        if needed_value > 100:
            price = fetch_real_time_price(api, ticker)
            if price:
                qty = needed_value / price
                if qty > 0.01:
                    buy_orders.append({'ticker': ticker, 'qty': qty})
                    print(f"  Leveraged MM: Buy ${needed_value:,.2f} of {ticker}")

        elif needed_value < -100:
            price = fetch_real_time_price(api, ticker)
            if price:
                qty = -needed_value / price
                current_qty = mm_positions.get(ticker, {}).get('qty', 0)
                qty = min(qty, current_qty)
                if qty > 0.01:
                    sell_orders.append({'ticker': ticker, 'qty': qty})
                    print(f"  Leveraged MM: Sell ${-needed_value:,.2f} of {ticker}")

    if buy_orders:
        results = execute_orders(api, buy_orders, 'buy', MM_STRATEGY_ID)
        for fill in results:
            portfolio["other_stocks"][fill['ticker']] = portfolio["other_stocks"].get(fill['ticker'], 0) + fill['qty']
            portfolio["cash"] -= fill['total']
            executed_trades.append(fill)
            executed_trades[-1]['note'] = 'LEVERAGED-MM'

    if sell_orders:
        results = execute_orders(api, sell_orders, 'sell', MM_STRATEGY_ID)
        for fill in results:
            portfolio["cash"] += fill['total']
            portfolio["other_stocks"][fill['ticker']] -= fill['qty']
            if portfolio["other_stocks"][fill['ticker']] <= 0.01:
                del portfolio["other_stocks"][fill['ticker']]
            executed_trades.append(fill)
            executed_trades[-1]['note'] = 'LEVERAGED-MM'

    return portfolio, executed_trades


# ============================================================================
# RS STOCK SCANNING
# ============================================================================

def get_top_rs_stocks():
    if not valid_russell_tickers:
        print("No valid Russell 3000 tickers available.")
        return []

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=rs_lookback_days + rs_rsi_period)

    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"RSI OF RATIO METHOD")
    print(f"Data period: {start_date_str} to {end_date_str}")
    print(f"RSI Period: {rs_rsi_period} days")
    print(f"{'='*60}")

    print(f"\nFetching SPY benchmark data...")
    spy_df = yf.download('SPY', start=start_date_str, end=end_date_str,
                         interval="1d", auto_adjust=True, progress=False)

    if spy_df.empty:
        print("Failed to fetch SPY data")
        return []

    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_close = spy_df['Close']['SPY']
    else:
        spy_close = spy_df['Close']

    print(f"SPY data: {len(spy_close)} trading days")

    clean_tickers = []
    invalid_patterns = ['.', '^', '-', '/', '$']
    for t in valid_russell_tickers:
        if t not in rs_invalid_tickers:
            if not any(p in t for p in invalid_patterns):
                clean_tickers.append(t)

    clean_tickers = clean_tickers[:rs_max_tickers_to_process]
    print(f"\nProcessing {len(clean_tickers)} tickers sequentially...")

    results = []
    processed = 0
    start_time = time.time()

    for ticker in clean_tickers:
        processed += 1
        if processed % 200 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(clean_tickers) - processed) / rate if rate > 0 else 0
            print(f"  Progress: {processed}/{len(clean_tickers)} ({processed/len(clean_tickers)*100:.1f}%) "
                  f"- Found {len(results)} candidates - ETA: {remaining/60:.1f} min")

        try:
            df = yf.download(ticker, start=start_date_str, end=end_date_str,
                            interval="1d", auto_adjust=True, progress=False)

            if df is None or df.empty or len(df) < 60:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                if ('Close', ticker) in df.columns:
                    stock_close = df['Close'][ticker]
                else:
                    continue
            else:
                if 'Close' in df.columns:
                    stock_close = df['Close']
                else:
                    continue

            if len(stock_close) < rs_rsi_period + 20:
                continue

            common_dates = spy_close.index.intersection(stock_close.index)
            if len(common_dates) < rs_rsi_period + 10:
                continue

            spy_aligned = spy_close.loc[common_dates]
            stock_aligned = stock_close.loc[common_dates]

            ratio_series = stock_aligned / spy_aligned
            ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan).dropna()

            if len(ratio_series) < rs_rsi_period:
                continue

            rsi_series = calculate_rsi(ratio_series, period=rs_rsi_period)

            if rsi_series.empty:
                continue

            latest_rsi = float(rsi_series.iloc[-1])
            if np.isnan(latest_rsi) or latest_rsi < 0 or latest_rsi > 100:
                continue

            start_ratio = float(ratio_series.iloc[0])
            end_ratio = float(ratio_series.iloc[-1])
            ratio_change = (end_ratio - start_ratio) / start_ratio if start_ratio > 0 else 0

            results.append({
                'ticker': ticker,
                'rsi': latest_rsi,
                'ratio_change': ratio_change
            })

        except Exception:
            continue

    elapsed = time.time() - start_time
    print(f"\nScan completed in {elapsed/60:.1f} minutes")
    print(f"Total candidates: {len(results)} out of {len(clean_tickers)} processed")

    if not results:
        print("No valid RS RSI calculations returned.")
        return []

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rsi', ascending=False)
    top_stocks = results_df.head(rs_num_stocks)['ticker'].tolist()

    print(f"\nTop {len(top_stocks)} RS momentum stocks:")
    for i, ticker in enumerate(top_stocks[:20]):
        ticker_data = results_df[results_df['ticker'] == ticker].iloc[0]
        print(f"  {i+1}. {ticker}: RSI={ticker_data['rsi']:.1f}, Change={ticker_data['ratio_change']*100:.1f}%")
    if len(top_stocks) > 20:
        print(f"  ... and {len(top_stocks)-20} more")

    return top_stocks

def calculate_rsi(prices, period=50):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_rs_tickers_in_portfolio(portfolio):
    alton_tickers = set(buy_tickers.keys())
    special_tickers = set(rebalancing_tickers)
    rs_tickers = {}
    for ticker, qty in portfolio["other_stocks"].items():
        if ticker not in alton_tickers and ticker not in special_tickers:
            rs_tickers[ticker] = qty
    return rs_tickers

def liquidate_cash_equivalents_for_rs(api, portfolio, target_rs_value, executed_trades):
    rs_holdings = get_rs_tickers_in_portfolio(portfolio)
    rs_holdings_value = 0
    for ticker, qty in rs_holdings.items():
        price = fetch_real_time_price(api, ticker)
        if price:
            rs_holdings_value += qty * price

    current_cash = portfolio['cash']
    cash_needed = max(0, target_rs_value - (rs_holdings_value + current_cash))

    if cash_needed <= 0:
        return portfolio, 0

    print(f"\nLiquidating cash equivalents for RS rebalancing")
    print(f"  Need: ${cash_needed:,.2f} additional for RS target")

    cash_holdings = {}
    all_mm_tickers = set(MONEY_MARKET_SWEEP_TICKERS + MONEY_MARKET_LEVERAGE_TICKERS)
    for ticker, qty in portfolio["other_stocks"].items():
        if ticker in all_mm_tickers:
            price = fetch_real_time_price(api, ticker)
            if price:
                cash_holdings[ticker] = {'qty': qty, 'price': price, 'value': qty * price}

    if not cash_holdings:
        print("No cash equivalents to liquidate")
        return portfolio, 0

    sell_orders = []
    for ticker, holding in cash_holdings.items():
        if holding['value'] > 0:
            sell_orders.append({'ticker': ticker, 'qty': holding['qty']})

    if sell_orders:
        print(f"\n--- LIQUIDATING {len(sell_orders)} CASH EQUIVALENTS ---")
        sell_results = execute_orders(api, sell_orders, 'sell', RS_STRATEGY_ID)

        total_liquidated = 0
        for fill in sell_results:
            portfolio["cash"] += fill['total']
            portfolio["other_stocks"][fill['ticker']] -= fill['qty']
            if portfolio["other_stocks"][fill['ticker']] <= 0.01:
                del portfolio["other_stocks"][fill['ticker']]
            executed_trades.append({
                'ticker': fill['ticker'],
                'side': 'sell',
                'qty': fill['qty'],
                'price': fill['price'],
                'total': fill['total']
            })
            total_liquidated += fill['total']

        print(f"  Total liquidated: ${total_liquidated:,.2f}")
        return portfolio, total_liquidated

    return portfolio, 0

def is_rs_rebalance_day(today=None):
    if today is None:
        today = datetime.now().date()
    us_holidays_cal = holidays.US(years=today.year)
    if today.weekday() == rs_rebalance_day:
        if today in us_holidays_cal:
            print(f"RS rebalance day is a holiday, deferring")
            return False
        return True
    return False

def get_next_rs_rebalance_day():
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
    today = datetime.now().date()
    days_ahead = rs_rebalance_day - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    next_day = today + timedelta(days=days_ahead)
    return next_day.strftime('%Y-%m-%d') + f" ({day_names.get(rs_rebalance_day, 'Unknown')})"


# ============================================================================
# RS REBALANCE WITH TOLERANCE CHECK
# ============================================================================

def rebalance_rs_portfolio(api, portfolio, top_stocks, total_rs_budget, executed_trades):
    target_per_stock = total_rs_budget / rs_num_stocks
    current_rs_holdings = get_rs_tickers_in_portfolio(portfolio)

    print(f"\n{'='*60}")
    print(f"RS REBALANCING - SEQUENTIAL ORDER SUBMISSION")
    print(f"{'='*60}")
    print(f"Total RS Budget: ${total_rs_budget:,.2f} for {rs_num_stocks} stocks")
    print(f"Target per stock: ${target_per_stock:,.2f}")

    prices = {}
    for ticker in top_stocks:
        price = fetch_real_time_price(api, ticker)
        if price:
            prices[ticker] = price
    for ticker in current_rs_holdings:
        if ticker not in prices:
            price = fetch_real_time_price(api, ticker)
            if price:
                prices[ticker] = price

    sell_orders = []
    buy_orders = []

    print(f"\nChecking each stock against {RS_REBALANCE_TOLERANCE*100:.0f}% tolerance:")

    for ticker in top_stocks:
        current_qty = current_rs_holdings.get(ticker, 0)
        current_price = prices.get(ticker)
        if not current_price:
            continue

        current_value = current_qty * current_price
        target_value = target_per_stock

        if target_value > 0:
            drift_pct = (current_value - target_value) / target_value
        else:
            drift_pct = 0

        if abs(drift_pct) > RS_REBALANCE_TOLERANCE:
            target_qty = target_value / current_price
            diff = target_qty - current_qty

            if diff < -0.001:
                sell_orders.append({'ticker': ticker, 'qty': -diff})
                print(f"  SELL {ticker}: Drift {drift_pct:+.1%} (exceeds {RS_REBALANCE_TOLERANCE*100:.0f}%)")
            elif diff > 0.001:
                buy_orders.append({'ticker': ticker, 'qty': diff})
                print(f"  BUY {ticker}: Drift {drift_pct:+.1%} (exceeds {RS_REBALANCE_TOLERANCE*100:.0f}%)")
        else:
            print(f"  SKIP {ticker}: Drift {drift_pct:+.1%} (within {RS_REBALANCE_TOLERANCE*100:.0f}% tolerance)")

    for ticker, current_qty in current_rs_holdings.items():
        if ticker not in top_stocks:
            current_price = prices.get(ticker)
            if current_price:
                sell_orders.append({'ticker': ticker, 'qty': current_qty})
                print(f"  SELL {ticker}: No longer in top 50")

    print(f"\nOrder Summary: {len(sell_orders)} sells, {len(buy_orders)} buys")

    sell_results = []
    if sell_orders:
        print(f"\n--- EXECUTING {len(sell_orders)} RS SELLS ---")
        sell_results = execute_orders(api, sell_orders, 'sell', RS_STRATEGY_ID)

        for fill in sell_results:
            executed_trades.append({
                'ticker': fill['ticker'],
                'side': 'sell',
                'qty': fill['qty'],
                'price': fill['price'],
                'total': fill['total']
            })
            if fill['ticker'] in portfolio["other_stocks"]:
                portfolio["other_stocks"][fill['ticker']] -= fill['qty']
                if portfolio["other_stocks"][fill['ticker']] <= 0.01:
                    del portfolio["other_stocks"][fill['ticker']]
            portfolio["cash"] += fill['total']

    buy_results = []
    if buy_orders:
        portfolio_value = calculate_portfolio_value(portfolio, api, include_mercury=True)
        min_cash_reserve = portfolio_value * min_cash_percentage
        available_for_buys = portfolio["cash"] - min_cash_reserve

        total_buy_value = 0
        for order in buy_orders:
            price = prices.get(order['ticker'], 0)
            total_buy_value += order['qty'] * price

        if total_buy_value > available_for_buys and available_for_buys > 0:
            scale = available_for_buys / total_buy_value
            for order in buy_orders:
                order['qty'] = order['qty'] * scale
            print(f"Scaled buys by {scale:.2%} to fit available cash")

        print(f"\n--- EXECUTING {len(buy_orders)} RS BUYS ---")
        buy_results = execute_orders(api, buy_orders, 'buy', RS_STRATEGY_ID)

        for fill in buy_results:
            executed_trades.append({
                'ticker': fill['ticker'],
                'side': 'buy',
                'qty': fill['qty'],
                'price': fill['price'],
                'total': fill['total']
            })
            portfolio["other_stocks"][fill['ticker']] = portfolio["other_stocks"].get(fill['ticker'], 0) + fill['qty']
            portfolio["cash"] -= fill['total']

    summary = f"RS Rebalance: Target ${total_rs_budget:,.2f}, " \
              f"Sells: {len(sell_results)}, " \
              f"Buys: {len(buy_results)}"

    return portfolio, summary


# ============================================================================
# CREATE SIGNAL HISTORY SUMMARY
# ============================================================================

def create_signal_history_summary(signal_history):
    """Create a summary of recent signals for email."""
    recent = signal_history.get_signal_sequence(5)

    if not recent:
        return '<div class="section"><div class="section-title">Signal History</div><p>No signals recorded.</p></div>'

    rows = []
    for r in reversed(recent):
        rows.append(f"""
        <div class="metric-row">
            <div class="metric-cell metric-label">{r['date']}</div>
            <div class="metric-cell metric-value" style="color: {'#2e7d32' if r['signal'] == 'BUY' else '#c62828'}">{r['signal']}</div>
            <div class="metric-cell metric-value">{r['confidence']:.1%}</div>
        </div>
        """)

    return f"""
    <div class="section">
        <div class="section-title">Signal History (Recent)</div>
        <div class="metrics-grid">
            <div class="metric-row" style="background:#f5f5f5;font-weight:bold;">
                <div class="metric-cell metric-label">Date</div>
                <div class="metric-cell metric-value">Signal</div>
                <div class="metric-cell metric-value">Confidence</div>
            </div>
            {''.join(rows)}
        </div>
    </div>
    """


# ============================================================================
# GENERATE MODEL SIGNAL
# ============================================================================

def generate_model_signal():
    """Generate the model signal with feature importance."""

    print(f"\n{'='*60}")
    print(f"GENERATING MODEL SIGNAL")
    print(f"{'='*60}")

    data = fetch_data(TICKERS)
    combined_df = combine_data(data)

    if combined_df.empty:
        print("⚠️ No data available - defaulting to BUY")
        return 1, 0.5, {}, []

    df = combined_df.copy()

    try:
        if "SPY_Close" in df.columns:
            df["SPY_Momentum"] = df["SPY_Close"] / df["SPY_Close"].rolling(50).mean() - 1
            df["SPY_TLT_Ratio"] = df["SPY_Close"] / df["TLT_Close"].rolling(50).mean() - 1 if "TLT_Close" in df.columns else 0
        else:
            df["SPY_Momentum"] = 0
            df["SPY_TLT_Ratio"] = 0

        if "CPER_Close" in df.columns and "GLD_Close" in df.columns:
            df["Copper_Gold_Ratio"] = df["CPER_Close"] / df["GLD_Close"].rolling(50).mean() - 1
        else:
            df["Copper_Gold_Ratio"] = 0

        if "VIXY_Close" in df.columns:
            df["VIX_Change"] = df["VIXY_Close"] / df["VIXY_Close"].rolling(50).mean() - 1
        else:
            df["VIX_Change"] = 0

        if "IEF_Close" in df.columns and "VGSH_Close" in df.columns:
            df["Yield_Spread"] = (df["IEF_Close"] - df["VGSH_Close"]) / (df["IEF_Close"] - df["VGSH_Close"]).rolling(50).mean() - 1
        else:
            df["Yield_Spread"] = 0

        if "IEF_Close" in df.columns and "HYG_Close" in df.columns:
            df["Credit_Spread"] = (df["IEF_Close"] - df["HYG_Close"]) / (df["IEF_Close"] - df["HYG_Close"]).rolling(50).mean() - 1
        else:
            df["Credit_Spread"] = 0

        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        SEASONAL_PERIODS_SPY = [
            {'start': (5, 13), 'end': (8, 14)},
            {'start': (10, 3), 'end': (11, 25)},
            {'start': (3, 3), 'end': (4, 28)}
        ]

        def is_in_season(date, start, end):
            start_date = pd.Timestamp(year=date.year, month=start[0], day=start[1])
            end_date = pd.Timestamp(year=date.year, month=end[0], day=end[1])
            return start_date <= date <= end_date

        for i, period in enumerate(SEASONAL_PERIODS_SPY):
            df[f"season_{i+1}"] = df.index.to_series().apply(lambda date: int(is_in_season(date, period['start'], period['end'])))

        if "SPY_Close" in df.columns:
            df["SPY_Fwd_Return"] = df["SPY_Close"].pct_change(10).shift(-10)
            df["Target"] = (df["SPY_Fwd_Return"] > 0).astype(int)
        else:
            print("⚠️ SPY data missing - defaulting to BUY")
            return 1, 0.5, {}, []

        df.dropna(inplace=True)

        if len(df) < 100:
            print(f"⚠️ Insufficient data: {len(df)} rows - defaulting to BUY")
            return 1, 0.5, {}, []

        seasonal_features = [f"season_{i+1}" for i in range(len(SEASONAL_PERIODS_SPY))]
        feature_columns = ["SPY_Momentum", "SPY_TLT_Ratio", "Copper_Gold_Ratio", "VIX_Change",
                           "Yield_Spread", "Credit_Spread", "day_of_week", "is_weekend", "month",
                           "quarter", "sin_day_of_year", "cos_day_of_year"] + seasonal_features

        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Using features: {available_features}")

        X = df[available_features]
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        X_current = df[available_features].iloc[-1:]
        signal = model.predict(X_current)[0]
        proba = model.predict_proba(X_current)[0]
        confidence = max(proba)

        feature_importance = dict(zip(available_features, model.feature_importances_))

        # Initialize transparency for reporting
        transparency = DecisionTransparency(model, available_features)
        decision_analysis = transparency.analyze_decision(X_current)
        transparency_report_html = transparency.generate_transparency_report(decision_analysis, 'html')

        print(f"\n📊 SIGNAL GENERATED:")
        print(f"  Signal: {'BUY' if signal == 1 else 'SELL'}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Top Features:")
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_importance[:5]:
            print(f"    - {feature}: {imp:.2%}")

        # Save transparency report to audit trail
        if AUDIT_ENABLED:
            audit_file = AUDIT_DIR / f"decision_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(audit_file, 'w') as f:
                f.write(transparency_report_html)
            print(f"📊 Transparency report saved to: {audit_file}")

        return signal, confidence, feature_importance, available_features

    except Exception as e:
        print(f"⚠️ Error generating signal: {e}")
        import traceback
        traceback.print_exc()
        return 1, 0.5, {}, []


# ============================================================================
# ALTON BOOK - Alton's own shares and cash, kept apart from the rest of the account
# ============================================================================

def alton_book_tickers():
    tickers = [t for t, w in buy_tickers.items() if w > 0]
    if SH_HEDGE_ENABLED and "SH" not in tickers:
        tickers.append("SH")
    return tickers

def get_prices(api, tickers):
    prices = {}
    for ticker in tickers:
        price = fetch_real_time_price(api, ticker)
        if price:
            prices[ticker] = float(price)
    return prices

def init_alton_book(strategy_state, portfolio):
    """
    Create the Alton book the first time this version runs: the Alton ETFs now in the
    account are Alton's, fully invested, with no Alton cash yet. SH and every other
    position stay with the other strategies.
    """
    state = strategy_state.load_state()
    if 'ledger_qty' in state:
        return state

    ledger_qty = {t: float(portfolio["other_stocks"].get(t, 0)) for t, w in buy_tickers.items() if w > 0}
    holds_etfs = any(q > 0 for q in ledger_qty.values())
    progress = 100.0 if holds_etfs else 0.0
    direction = 'BUY' if holds_etfs else 'SELL'

    print(f"\n📒 CREATING ALTON BOOK from current holdings:")
    for t, q in ledger_qty.items():
        print(f"   {t}: {q:.4f} shares")
    print(f"   Alton cash: $0.00 | Position: {progress:.0f}% ({direction}, cycle complete)")

    return strategy_state.update_state({
        'version': 3,
        'ledger_qty': ledger_qty,
        'ledger_cash': 0.0,
        'direction': direction,
        'start_pct': progress,
        'progress_pct': progress,
        'current_day': execution_days,
        'cycle_complete': True,
        'cycle_start_date': None,
        'last_step_date': None,
        'capital_reset_used': None,
        'state': StrategyStateManager.STATE_FULLY_INVESTED_BUY if holds_etfs else StrategyStateManager.STATE_FULLY_INVESTED_SELL,
    })

def alton_book_value(state, prices):
    """Value of Alton's own positions + Alton's own cash."""
    positions = {t: q * prices.get(t, 0) for t, q in state['ledger_qty'].items()}
    cash = float(state.get('ledger_cash', 0))
    return positions, cash, sum(positions.values()) + cash

def apply_capital_reset(strategy_state, state, prices):
    if ALTON_CAPITAL_RESET is None or state.get('capital_reset_used') == ALTON_CAPITAL_RESET:
        return state
    positions, _, _ = alton_book_value(state, prices)
    new_cash = float(ALTON_CAPITAL_RESET) - sum(positions.values())
    print(f"\n📒 ALTON CAPITAL RESET to ${ALTON_CAPITAL_RESET:,.2f}: Alton cash set to ${new_cash:,.2f}")
    return strategy_state.update_state({'ledger_cash': new_cash, 'capital_reset_used': ALTON_CAPITAL_RESET})

def plan_cycle_step(state, today_signal, confidence, current_date):
    """
    Decide today's position in the 3-day cycle.
    A new cycle starts when the signal flips. Each following trading day moves one step
    (1/3 of the way) from where the cycle started to 100% (BUY) or 0% (SELL).
    Running again on the same day never adds a step.
    """
    direction = state.get('direction')
    day = int(state.get('current_day', execution_days))
    start_pct = float(state.get('start_pct', state.get('progress_pct', 0)))
    cycle_start_date = state.get('cycle_start_date')
    last_step_date = state.get('last_step_date')
    new_cycle = False

    if today_signal != direction and confidence >= MIN_CONFIDENCE_TO_SWITCH:
        start_pct = float(state.get('progress_pct', 0))
        direction = today_signal
        day = 1
        cycle_start_date = current_date
        last_step_date = current_date
        new_cycle = True
        note = f"Signal flipped to {today_signal} - new {execution_days}-day cycle started"
    elif today_signal != direction:
        note = (f"{today_signal} signal at {confidence:.0%} is below the {MIN_CONFIDENCE_TO_SWITCH:.0%} "
                f"needed to switch - keeping the {direction} position")
    elif day < execution_days and last_step_date != current_date:
        day += 1
        last_step_date = current_date
        note = f"{direction} cycle - day {day} of {execution_days}"
    elif day < execution_days:
        note = f"{direction} cycle - day {day} of {execution_days} (already stepped today)"
    else:
        note = f"{direction} cycle complete - holding"

    end_pct = 100.0 if direction == 'BUY' else 0.0
    progress_pct = start_pct + (end_pct - start_pct) * min(day, execution_days) / execution_days

    if day >= execution_days:
        state_name = StrategyStateManager.STATE_FULLY_INVESTED_BUY if direction == 'BUY' else StrategyStateManager.STATE_FULLY_INVESTED_SELL
    else:
        state_name = f"{'BUYING' if direction == 'BUY' else 'SELLING'}_DAY_{day}"

    return {
        'direction': direction,
        'current_day': day,
        'start_pct': start_pct,
        'progress_pct': progress_pct,
        'cycle_start_date': cycle_start_date,
        'last_step_date': last_step_date,
        'cycle_complete': day >= execution_days,
        'state': state_name,
    }, new_cycle, note

def build_alton_orders(state, prices, account_qty, progress_pct):
    """
    Target value per ticker from Alton's own value, and the orders to get there.
    A ticker is only traded when its gap is bigger than both MIN_TRADE_VALUE and
    ALTON_REBALANCE_TOLERANCE of its target, so small drifts are never chased.
    """
    positions, cash, capital = alton_book_value(state, prices)
    invest_value = capital * (1 - ALTON_CASH_RESERVE_PCT / 100) * (1 - rs_allocation_percentage)

    targets = {t: invest_value * (progress_pct / 100) * w for t, w in buy_tickers.items() if w > 0}
    sh_allocation = SH_MONTHLY_ALLOCATION.get(datetime.now().month, 0.0) if SH_HEDGE_ENABLED else 0.0
    if SH_HEDGE_ENABLED:
        # The hedge grows as Alton sells out and shrinks as it buys back in
        targets["SH"] = targets.get("SH", 0) + invest_value * sh_allocation * (1 - progress_pct / 100)

    rows, orders = [], []
    for ticker, target in targets.items():
        price = prices.get(ticker)
        held_qty = float(state['ledger_qty'].get(ticker, 0))
        value = positions.get(ticker, 0)
        gap = target - value
        threshold = max(MIN_TRADE_VALUE, target * ALTON_REBALANCE_TOLERANCE)
        action = "Hold"

        if price and abs(gap) > threshold:
            if gap > 0:
                orders.append({'ticker': ticker, 'qty': round(gap / price, 6), 'side': 'buy'})
                action = f"Buy ${gap:,.0f}"
            else:
                # Never sell more than Alton owns, nor more than the account holds
                sell_qty = held_qty if target <= 0 else abs(gap) / price
                sell_qty = round(min(sell_qty, held_qty, float(account_qty.get(ticker, 0))), 6)
                if sell_qty > 0.0001:
                    orders.append({'ticker': ticker, 'qty': sell_qty, 'side': 'sell'})
                    action = f"Sell ${sell_qty * price:,.0f}"
        elif gap != 0:
            action = f"Hold (gap ${gap:,.0f} under ${threshold:,.0f} threshold)"

        rows.append({'ticker': ticker, 'qty': held_qty, 'price': price or 0, 'value': value,
                     'target': target, 'gap': gap, 'action': action})

    return rows, orders, capital, cash, sh_allocation

def collect_alton_fills(api, since):
    """What Alton's orders since `since` actually filled (includes re-priced and partly filled orders)."""
    fills = []
    try:
        orders = api.list_orders(status='all', after=since.isoformat(), limit=500, direction='asc')
    except Exception as e:
        print(f"⚠️ Could not read Alton orders from Alpaca: {e}")
        return fills
    for o in orders:
        if not (o.client_order_id or '').startswith(f"{ALTON_STRATEGY_ID}_"):
            continue
        qty = float(o.filled_qty or 0)
        if qty <= 0:
            continue
        price = float(o.filled_avg_price)
        fills.append({'ticker': o.symbol, 'side': o.side, 'qty': qty, 'price': price, 'total': qty * price})
    return fills

def cancel_open_alton_orders(api):
    try:
        for order in api.list_orders(status='open', limit=500):
            if (order.client_order_id or '').startswith(f"{ALTON_STRATEGY_ID}_"):
                api.cancel_order(order.id)
                print(f"  Cancelled unfilled {order.symbol} {order.side} order")
    except Exception as e:
        print(f"⚠️ Could not cancel open Alton orders: {e}")

def apply_fills_to_book(state, fills):
    ledger_qty = dict(state['ledger_qty'])
    cash = float(state.get('ledger_cash', 0))
    for f in fills:
        if f['side'] == 'buy':
            ledger_qty[f['ticker']] = ledger_qty.get(f['ticker'], 0) + f['qty']
            cash -= f['total']
        else:
            ledger_qty[f['ticker']] = max(0.0, ledger_qty.get(f['ticker'], 0) - f['qty'])
            cash += f['total']
    return {'ledger_qty': ledger_qty, 'ledger_cash': cash}


# ============================================================================
# ALTON EMAIL SECTIONS
# ============================================================================

def create_alton_book_section(rows, capital, cash, progress_pct, title):
    invested = sum(r['value'] for r in rows)
    body = "".join(
        f'<tr><td class="ticker">{r["ticker"]}</td><td>{r["qty"]:.4f}</td><td>${r["value"]:,.2f}</td>'
        f'<td>${r["target"]:,.2f}</td><td class="weight">{r["action"]}</td></tr>' for r in rows)
    return f"""<div class="section"><div class="section-title">{title}</div><div class="metrics-grid">
        <div class="metric-row"><div class="metric-cell metric-label">Alton Value (own positions + own cash)</div><div class="metric-cell metric-value">${capital:,.2f}</div></div>
        <div class="metric-row"><div class="metric-cell metric-label">Alton Positions</div><div class="metric-cell metric-value">${invested:,.2f} ({invested / capital * 100 if capital else 0:.1f}%)</div></div>
        <div class="metric-row"><div class="metric-cell metric-label">Alton Cash</div><div class="metric-cell metric-value">${cash:,.2f} ({cash / capital * 100 if capital else 0:.1f}%)</div></div>
        <div class="metric-row"><div class="metric-cell metric-label">Cycle Target</div><div class="metric-cell metric-value">{progress_pct:.1f}% of {100 - ALTON_CASH_RESERVE_PCT:.1f}% invested</div></div>
        </div><table class="holdings-table"><thead><tr><th>Ticker</th><th>Shares</th><th>Value</th><th>Target</th><th style="text-align:right;">Action</th></tr></thead><tbody>{body}</tbody></table></div>"""

def create_fills_section(fills, rejected):
    if fills:
        items = "".join(
            f'<div class="trade-item {"trade-buy" if f["side"] == "buy" else "trade-sell"}"><div class="trade-ticker">{f["side"].upper()} {f["ticker"]}</div>'
            f'<div class="trade-details">{f["qty"]:.4f} shares @ ${f["price"]:.2f}</div><div class="trade-amount">${f["total"]:,.2f}</div></div>' for f in fills)
        html = f'<div class="section"><div class="section-title">Filled Today</div>{items}</div>'
    else:
        html = ''
    if rejected:
        html += create_alert_box("Orders not filled", "<br>".join(rejected), "error")
    return html

def create_account_section(account_equity, account_cash, alton_capital):
    other = account_equity - alton_capital
    return f"""<div class="section"><div class="section-title">Whole Account (information only)</div><div class="metrics-grid">
        <div class="metric-row"><div class="metric-cell metric-label">Account Equity</div><div class="metric-cell metric-value">${account_equity:,.2f}</div></div>
        <div class="metric-row"><div class="metric-cell metric-label">Account Cash (negative = margin)</div><div class="metric-cell metric-value">${account_cash:,.2f}</div></div>
        <div class="metric-row"><div class="metric-cell metric-label">Other Strategies (equity outside Alton)</div><div class="metric-cell metric-value">${other:,.2f}</div></div>
        </div><div style="font-size:12px;color:#888;margin-top:6px;">Alton never trades or sizes on positions outside its own book.</div></div>"""


# ============================================================================
# MAIN TRADING STRATEGY - ALTON BOOK WITH 3-DAY CYCLES
# ============================================================================

def trading_strategy(api, portfolio, transaction_log, api_key):
    """Run one day of the Alton strategy on Alton's own book."""

    portfolio_name = get_portfolio_name(api_key)
    current_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n{'='*60}")
    print(f"Processing: {portfolio_name}")
    print(f"  Alton Cash Reserve: {ALTON_CASH_RESERVE_PCT:.1f}% (fully invested = {100 - ALTON_CASH_RESERVE_PCT:.1f}%)")
    print(f"  Rebalance Threshold: {ALTON_REBALANCE_TOLERANCE*100:.0f}% of target and at least ${MIN_TRADE_VALUE:,.0f}")
    print(f"  RS Allocation: {rs_allocation_percentage*100:.0f}%")
    print(f"  SH Hedge: {'ON' if SH_HEDGE_ENABLED else 'OFF'}")
    print(f"{'='*60}\n")

    if not is_market_open_today():
        print("Market closed today - skipping execution")
        return None, None

    signal_history = SignalHistoryManager(api_key)
    strategy_state = StrategyStateManager(api_key)
    portfolio = fetch_current_positions(api) or portfolio_template
    account_qty = portfolio["other_stocks"]

    # ----------------------------------------------------------------
    # 1. Alton book and prices
    # ----------------------------------------------------------------
    state = init_alton_book(strategy_state, portfolio)
    prices = get_prices(api, alton_book_tickers())
    missing = [t for t, q in state['ledger_qty'].items() if q > 0 and t not in prices]
    if missing:
        msg = f"No price for {', '.join(missing)} - Alton skipped today so nothing is traded on a wrong value."
        print(f"⚠️ {msg}")
        email_body = create_email_header("Alton skipped today", "Price data missing", "error", portfolio_name)
        email_body += create_alert_box("No trading today", msg, "error") + create_email_footer()
        send_email_with_attachment(f"{portfolio_name} - Alton skipped (no prices)", email_body)
        return None, None
    state = apply_capital_reset(strategy_state, state, prices)

    for ticker, q in state['ledger_qty'].items():
        if q > float(account_qty.get(ticker, 0)) + 0.0001:
            print(f"⚠️ Alton book has {q:.4f} {ticker} but the account only holds {float(account_qty.get(ticker, 0)):.4f}")

    # ----------------------------------------------------------------
    # 2. Signal (the first signal of the day is the one used for the whole day)
    # ----------------------------------------------------------------
    stored = signal_history.get_signal_on_date(current_date)
    if stored:
        today_signal, confidence = stored['signal'], stored['confidence']
        print(f"\n📊 Using today's recorded signal: {today_signal} ({confidence:.1%})")
    else:
        signal_value, confidence, feature_importance, feature_names = generate_model_signal()
        today_signal = 'BUY' if signal_value == 1 else 'SELL'
        signal_history.add_signal(current_date, today_signal, confidence, feature_importance, feature_names)
        print(f"✅ Added signal to history: {current_date} - {today_signal} ({confidence:.1%})")
    signal_numeric = 1 if today_signal == 'BUY' else 0

    # ----------------------------------------------------------------
    # 3. Cycle step
    # ----------------------------------------------------------------
    plan, new_cycle, note = plan_cycle_step(state, today_signal, confidence, current_date)
    print(f"\n📊 CYCLE: {note}")
    print(f"  Direction: {plan['direction']} | Day {plan['current_day']}/{execution_days} | Target: {plan['progress_pct']:.1f}% invested")

    # ----------------------------------------------------------------
    # 4. Orders on Alton's own book
    # ----------------------------------------------------------------
    rows, orders, capital, cash, sh_allocation = build_alton_orders(state, prices, account_qty, plan['progress_pct'])
    print(f"\n📒 ALTON BOOK: value ${capital:,.2f} (cash ${cash:,.2f})")
    for r in rows:
        print(f"  {r['ticker']}: {r['qty']:.4f} sh = ${r['value']:,.2f} | target ${r['target']:,.2f} | {r['action']}")

    fills, rejected = [], []
    if orders:
        run_start = datetime.now(timezone('UTC')) - timedelta(minutes=1)
        monitor_thread = start_monitor_thread(api)
        for side in ('sell', 'buy'):
            side_orders = [o for o in orders if o['side'] == side]
            submitted = execute_orders(api, side_orders, side, ALTON_STRATEGY_ID)
            submitted_tickers = {s['ticker'] for s in submitted}
            rejected += [f"{o['ticker']} {side} {o['qty']:.4f} shares was rejected by the broker"
                         for o in side_orders if o['ticker'] not in submitted_tickers]

        print(f"\n📊 WAITING FOR ORDERS TO FILL (up to {ORDER_WAIT_MINUTES} min)")
        start_wait = datetime.now()
        while len(active_orders) > 0:
            if (datetime.now() - start_wait).total_seconds() / 60 > ORDER_WAIT_MINUTES:
                print(f"⚠️ Timeout - cancelling {len(active_orders)} unfilled orders")
                break
            time.sleep(10)
        stop_monitor_thread()
        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.join(timeout=10)
        cancel_open_alton_orders(api)
        active_orders.clear()
        time.sleep(2)

        fills = collect_alton_fills(api, run_start)
        filled_by_ticker = {}
        for f in fills:
            filled_by_ticker[(f['ticker'], f['side'])] = filled_by_ticker.get((f['ticker'], f['side']), 0) + f['qty']
        for o in orders:
            done = filled_by_ticker.get((o['ticker'], o['side']), 0)
            if 0 < done < o['qty'] - 0.0001:
                rejected.append(f"{o['ticker']} {o['side']}: only {done:.4f} of {o['qty']:.4f} shares filled - the rest is retried next run")
            elif done == 0 and not any(o['ticker'] in r for r in rejected):
                rejected.append(f"{o['ticker']} {o['side']}: no shares filled - retried next run")
        print(f"\n✅ Filled {len(fills)} order(s)")

    # ----------------------------------------------------------------
    # 5. Save the book and the cycle
    # ----------------------------------------------------------------
    plan.update(apply_fills_to_book(state, fills))
    plan['last_execution_date'] = current_date
    state = strategy_state.update_state(plan)

    positions, cash, capital = alton_book_value(state, prices)
    for r in rows:
        r['qty'] = float(state['ledger_qty'].get(r['ticker'], 0))
        r['value'] = positions.get(r['ticker'], r['qty'] * r['price'])

    # ----------------------------------------------------------------
    # 6. Daily email
    # ----------------------------------------------------------------
    if plan['cycle_complete']:
        status = f"Holding - {'fully invested' if plan['direction'] == 'BUY' else 'in Alton cash'}"
    else:
        status = f"{plan['direction']} day {plan['current_day']}/{execution_days}"
    subject = f"{portfolio_name} - Alton {status} ({plan['progress_pct']:.0f}%)"

    try:
        account = api.get_account()
        account_equity, account_cash = float(account.equity), float(account.cash)
    except Exception:
        account_equity, account_cash = capital, float(portfolio["cash"])

    email_body = create_email_header(f"Alton {status}", note, "success" if fills else "update", portfolio_name)
    email_body += create_signal_box(signal_numeric, confidence, new_cycle)
    email_body += create_cycle_progress(plan['current_day'], execution_days, 1 if plan['direction'] == 'BUY' else 0,
                                        cycle_status="COMPLETE" if plan['cycle_complete'] else None)
    email_body += create_fills_section(fills, rejected)
    if not orders:
        email_body += create_alert_box("No trades needed", "Alton is within its rebalance threshold - nothing was traded.", "success")
    email_body += create_alton_book_section(rows, capital, cash, plan['progress_pct'], "Alton Book")
    if SH_HEDGE_ENABLED:
        email_body += create_sh_allocation_box(MONTH_NAMES[datetime.now().month], sh_allocation)
    email_body += create_account_section(account_equity, account_cash, capital)
    email_body += create_signal_history_summary(signal_history)
    email_body += create_email_footer()
    send_email_with_attachment(subject, email_body)

    return email_body, signal_numeric


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    lock_file = "alton_rs_model.lock"
    try:
        import fcntl

        with open(lock_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            print("\n" + "="*60)
            print("ALTON RS HYBRID MODEL - ALTON BOOK & 3-DAY CYCLES")
            print(f"Alton: {(1-rs_allocation_percentage)*100:.0f}% | RS: {rs_allocation_percentage*100:.0f}%")
            print(f"Cash Reserve: {ALTON_CASH_RESERVE_PCT:.1f}% | Execution Days: {execution_days}")
            print(f"Min Confidence To Switch: {MIN_CONFIDENCE_TO_SWITCH*100:.0f}%")
            print("="*60 + "\n")

            for client in API_CLIENTS:
                api = tradeapi.REST(client["api_key"], client["secret_key"], client["base_url"])
                if not verify_api_connection(api, client["api_key"]):
                    portfolio_name = client["name"]
                    email_body = create_email_header("API Authentication Failed", "Critical Error", "error", portfolio_name)
                    email_body += create_alert_box("Authentication Error", "Please update API credentials.", "error")
                    email_body += create_email_footer()
                    send_email_with_attachment(f"{portfolio_name} - API Auth Error", email_body)
                    continue

            for client in API_CLIENTS:
                api = tradeapi.REST(client["api_key"], client["secret_key"], client["base_url"])
                api_key = client["api_key"]
                trading_strategy(api, fetch_current_positions(api), [], api_key)

            print("\n" + "="*60)
            print("ALL ACCOUNTS PROCESSED")
            print("="*60 + "\n")

        os.remove(lock_file)

    except Exception as e:
        error_msg = f"Critical error: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        for client in API_CLIENTS:
            portfolio_name = client["name"]
            email_body = create_email_header("Critical Error", "System Failure", "error", portfolio_name)
            email_body += create_alert_box("System Error", error_msg, "error")
            email_body += create_email_footer()
            send_email_with_attachment(f"{portfolio_name} - Error", email_body)
        if os.path.exists(lock_file):
            os.remove(lock_file)
        raise
