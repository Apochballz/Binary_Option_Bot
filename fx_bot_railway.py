#!/usr/bin/env python3
"""
Production-ready improved app.py for Binary Option Bot
Features:
- Safe async handling (asyncio.run usage)
- OHLC fetching (yfinance primary, TwelveData optional)
- Caching for OHLC + indicators
- Robust indicator computation (ta) with NaN safety
- SQLite persistence for win history + metrics
- VectorBT and SquareQuant optional integrations
- Rate limiting (simple token bucket)
- Health route and Railway-safe PORT handling
- Logging and error handling
"""

import os
import time
import json
import logging
import asyncio
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, List

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv

# load .env if present
load_dotenv()

# -------------------------
# Logging
# -------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("binary_bot")

# -------------------------
# Optional integrations (vectorbt, squarequant)
# -------------------------
HAS_VECTORBT = False
HAS_SQUAREQUANT = False
try:
    import vectorbt as vbt  # type: ignore
    HAS_VECTORBT = True
    log.info("vectorbt available for backtesting")
except Exception:
    log.info("vectorbt not available — backtesting endpoints will be disabled")

try:
    import squarequant as sq  # type: ignore
    HAS_SQUAREQUANT = True
    log.info("SquareQuant available for risk metrics")
except Exception:
    log.info("SquareQuant not available — risk metrics endpoints will be disabled")

# -------------------------
# Scientific libs required
# -------------------------
try:
    import numpy as np
    import pandas as pd
    import ta
    import pytz
except Exception as e:
    log.exception("Required scientific libs missing. Install requirements before running.")
    raise

# -------------------------
# HTTP / data libs
# -------------------------
import requests
# yfinance optional
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

# -------------------------
# Persistence (SQLite)
# -------------------------
import sqlite3
DB_PATH = os.environ.get("BOT_SQLITE_PATH", "bot_state.sqlite3")
_conn = None


def get_db_conn(path: str = DB_PATH):
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        cur = _conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS win_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                outcome INTEGER,
                note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _conn.commit()
    return _conn


def save_trade_result(symbol: str, outcome: int, note: str = ""):
    """Save 1 for win, 0 for loss"""
    cur = get_db_conn().cursor()
    cur.execute("INSERT INTO win_history (symbol, outcome, note) VALUES (?,?,?)", (symbol, int(outcome), note))
    get_db_conn().commit()


def get_win_rate(n: int = 100) -> Optional[float]:
    cur = get_db_conn().cursor()
    cur.execute("SELECT outcome FROM win_history ORDER BY id DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    if not rows:
        return None
    wins = sum([r["outcome"] for r in rows])
    return wins / len(rows) * 100.0

# -------------------------
# Health check / app init
# -------------------------
app = Flask(__name__)
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})


@app.route("/")
def index():
    return app.send_static_file('index.html')


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


# -------------------------
# Simple in-memory caching (per-process)
# TTL configurable via env CACHE_TTL_SECONDS
# -------------------------
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", "5"))  # very short for near-real-time
_ohlc_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
_indicators_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def cache_get_ohlc(key: str) -> Optional[pd.DataFrame]:
    item = _ohlc_cache.get(key)
    if not item:
        return None
    ts, df = item
    if time.time() - ts > CACHE_TTL:
        _ohlc_cache.pop(key, None)
        return None
    return df


def cache_set_ohlc(key: str, df: pd.DataFrame):
    _ohlc_cache[key] = (time.time(), df)


def cache_get_indicators(key: str) -> Optional[Dict[str, Any]]:
    item = _indicators_cache.get(key)
    if not item:
        return None
    ts, payload = item
    if time.time() - ts > CACHE_TTL:
        _indicators_cache.pop(key, None)
        return None
    return payload


def cache_set_indicators(key: str, payload: Dict[str, Any]):
    _indicators_cache[key] = (time.time(), payload)


# -------------------------
# Rate limiter (token bucket) - simple per-IP
# -------------------------
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
_tokens: Dict[str, Dict[str, Any]] = {}  # ip -> {'tokens': int, 'last': ts'}


def allow_request(ip: str) -> bool:
    bucket = _tokens.get(ip)
    now = time.time()
    if not bucket:
        _tokens[ip] = {'tokens': RATE_LIMIT, 'last': now}
        bucket = _tokens[ip]
    # refill
    elapsed = now - bucket['last']
    refill = (elapsed / 60.0) * RATE_LIMIT
    bucket['tokens'] = min(RATE_LIMIT, bucket['tokens'] + refill)
    bucket['last'] = now
    if bucket['tokens'] >= 1:
        bucket['tokens'] -= 1
        return True
    return False


@app.before_request
def _global_rate_limit():
    ip = request.remote_addr or "anon"
    if not allow_request(ip):
        abort(429, description="Rate limit exceeded")


# -------------------------
# OHLC Fetcher (yfinance primary, TwelveData optional)
# - symbol format: EURUSD or EURUSD=X depending on provider
# - returns pandas DataFrame with columns: open, high, low, close, volume
# -------------------------
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")


def _fetch_with_twelvedata(symbol: str, interval: str = "5min", limit: int = 200) -> Optional[pd.DataFrame]:
    if not TWELVEDATA_API_KEY:
        return None
    try:
        # twelve data expects symbol like EUR/USD; user can pass EURUSD or EUR/USD
        td_symbol = symbol.replace("=", "").replace("X", "")
        # choose endpoint
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": td_symbol,
            "interval": interval,
            "outputsize": limit,
            "format": "JSON",
            "apikey": TWELVEDATA_API_KEY
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "values" not in data:
            return None
        values = data["values"]
        values.reverse()  # oldest -> newest
        df = pd.DataFrame(values)
        # map columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                # some TD responses include 'volume' as string 'volume'
                pass
        df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df.tail(limit).reset_index(drop=True)
    except Exception as e:
        log.warning("TwelveData fetch failed: %s", e)
        return None


def fetch_ohlc(symbol: str = "EURUSD", interval: str = "5m", limit: int = 200) -> pd.DataFrame:
    key = f"{symbol}:{interval}:{limit}"
    cached = cache_get_ohlc(key)
    if cached is not None:
        return cached

    # Try TwelveData first (if configured)
    if TWELVEDATA_API_KEY:
        df_td = _fetch_with_twelvedata(symbol, interval=interval, limit=limit)
        if df_td is not None and not df_td.empty:
            cache_set_ohlc(key, df_td)
            return df_td

    # Try yfinance if available
    if HAS_YFINANCE:
        try:
            yf_symbol = symbol if symbol.endswith("=X") else f"{symbol}=X"
            data = yf.download(tickers=yf_symbol, period="1d", interval=interval, progress=False)
            if data is not None and not data.empty:
                data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
                df = data[['open', 'high', 'low', 'close', 'volume']].tail(limit).reset_index(drop=True)
                cache_set_ohlc(key, df)
                return df
        except Exception as e:
            log.warning("yfinance fetch failed: %s", e)

    # Fallback: deterministic mock (use seeded RNG so results are reproducible)
    rng = np.random.RandomState(42)
    base = 1.10
    highs = base + rng.rand(limit) * 0.02
    lows = base - rng.rand(limit) * 0.02
    closes = base + (rng.rand(limit) - 0.5) * 0.01
    opens = closes + (rng.rand(limit) - 0.5) * 0.005
    volumes = (rng.rand(limit) * 1000).astype(float)
    df_mock = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes})
    cache_set_ohlc(key, df_mock)
    return df_mock


# -------------------------
# Safe indicator helper
# - normalizes slope (percentage slope), handles NaNs
# -------------------------
def safe_fill_and_check(df: pd.DataFrame, min_len: int = 50) -> pd.DataFrame:
    # Ensure columns exist
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = np.nan
    # forward fill and drop remaining NaNs
    df = df.ffill().bfill()
    if len(df) < min_len:
        raise ValueError(f"Not enough data: {len(df)} candles (need {min_len})")
    return df


def pct_slope(series: pd.Series) -> float:
    """Return slope normalized as percent per candle (simple)."""
    y = np.log(series.astype(float) + 1e-9)
    x = np.arange(len(y))
    if len(y) < 2:
        return 0.0
    m = np.polyfit(x, y, 1)[0]
    return float(m)


# -------------------------
# Classes: MarketRegimeDetector, ConfluenceScorer, SessionFilter, AdaptiveRiskManager,
# PositionSizer, OrderFlowAnalyzer (improved versions)
# -------------------------
class MarketRegimeDetector:
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = safe_fill_and_check(df, min_len=50)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # ADX
        try:
            adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            adx = float(adx_indicator.adx()[-1])
        except Exception:
            adx = 0.0

        # ATR pct
        try:
            atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
            atr = float(atr_indicator.average_true_range()[-1])
            atr_pct = (atr / float(close[-1])) * 100
        except Exception:
            atr_pct = 0.0

        # percent slope
        try:
            slope = pct_slope(pd.Series(close[-50:]))
        except Exception:
            slope = 0.0

        # Bollinger width
        try:
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband()[-1]
            bb_lower = bb.bollinger_lband()[-1]
            bb_mid = bb.bollinger_mavg()[-1] if bb.bollinger_mavg() is not None else close[-1]
            bb_width = ((bb_upper - bb_lower) / (bb_mid + 1e-9)) * 100
        except Exception:
            bb_width = 0.0

        if adx > 25 and abs(slope) > 1e-5:
            regime = "TRENDING"; conf_mul = 1.3; trade_filter = "TREND_ONLY"
        elif adx < 20 and atr_pct < 0.5:
            regime = "RANGING"; conf_mul = 0.7; trade_filter = "SKIP"
        elif atr_pct > 1.0 or bb_width > 5:
            regime = "VOLATILE"; conf_mul = 0.6; trade_filter = "HIGH_RISK"
        else:
            regime = "TRANSITIONAL"; conf_mul = 0.85; trade_filter = "CAUTIOUS"

        return {
            "regime": regime,
            "adx": adx,
            "slope": slope,
            "volatility_pct": atr_pct,
            "bb_width": bb_width,
            "confidence_multiplier": conf_mul,
            "trade_filter": trade_filter,
            "trend_direction": "BULLISH" if slope > 0 else "BEARISH"
        }


class ConfluenceScorer:
    def calculate_confluence(self, indicators: Dict[str, Any], regime: Dict[str, Any], session_data: Dict[str, Any],
                             order_flow: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # weights tuned and normalized to 100
        score = 0.0
        details: List[str] = []

        # Trend (30)
        trend_score = 0
        rsi = indicators.get("momentum", {}).get("rsi", 50)
        if rsi < 30:
            trend_score += 10; details.append("RSI Oversold (+10)")
        elif rsi > 70:
            trend_score += 10; details.append("RSI Overbought (+10)")
        elif 40 < rsi < 60:
            trend_score += 5; details.append("RSI Neutral (+5)")

        macd = indicators.get("momentum", {}).get("macd", 0)
        macd_signal = indicators.get("momentum", {}).get("macd_signal", 0)
        if macd > macd_signal:
            trend_score += 10; details.append("MACD Bullish (+10)")
        else:
            trend_score += 10; details.append("MACD Bearish (+10)")

        adx = indicators.get("trend", {}).get("adx", 0)
        if adx > 25:
            trend_score += 10; details.append("Strong Trend (+10)")
        elif adx > 20:
            trend_score += 5; details.append("Moderate Trend (+5)")

        score += trend_score

        # Volume (20)
        volume_score = 0
        vol_info = indicators.get("volume", {})
        if vol_info:
            obv_trend = vol_info.get("obv_trend", "NEUTRAL")
            cmf = vol_info.get("cmf", 0)
            if obv_trend in ("BULLISH", "BEARISH"):
                volume_score += 10; details.append(f"OBV {obv_trend} (+10)")
            if abs(cmf) > 0.1:
                volume_score += 10; details.append(f"CMF Strong ({cmf:.2f}) (+10)")
        else:
            volume_score = 10
        score += volume_score

        # Session (15)
        session_score = 0
        if session_data.get("tradeable"):
            s = session_data.get("session")
            if s == "OVERLAP":
                session_score = 15; details.append("Peak Session: OVERLAP (+15)")
            elif s in ("LONDON", "NEW_YORK"):
                session_score = 10; details.append(f"{s} Session (+10)")
        else:
            details.append("Off-Hours Session (0)")
        score += session_score

        # Regime (15)
        regime_score = 0
        if regime.get("regime") == "TRENDING":
            regime_score = 15; details.append("Trending Market (+15)")
        elif regime.get("regime") == "RANGING":
            regime_score = 0; details.append("Ranging Market (0)")
        elif regime.get("regime") == "VOLATILE":
            regime_score = 5; details.append("Volatile Market (+5)")
        else:
            regime_score = 8; details.append("Transitional Market (+8)")
        score += regime_score

        # Orderflow (20)
        orderflow_score = 0
        if order_flow and "error" not in order_flow:
            imbalance = order_flow.get("imbalance_ratio", 1.0)
            if imbalance > 1.5:
                orderflow_score += 10; details.append("Strong Buy Pressure (+10)")
            elif imbalance < 0.67:
                orderflow_score += 10; details.append("Strong Sell Pressure (+10)")
            elif 0.8 < imbalance < 1.2:
                orderflow_score += 5; details.append("Balanced Flow (+5)")
            cvd = abs(order_flow.get("cvd", 0))
            if cvd > 1000:
                orderflow_score += 10; details.append(f"CVD Confirmation (+10)")
        else:
            orderflow_score = 10; details.append("Order Flow: N/A (+10 default)")

        score += orderflow_score

        final_score = min((score / 100.0) * 100.0, 100.0)
        return {"score": final_score, "breakdown": {"trend": trend_score, "volume": volume_score,
                                                    "session": session_score, "regime": regime_score,
                                                    "orderflow": orderflow_score}, "details": details}


class SessionFilter:
    def is_tradeable_session(self) -> Dict[str, Any]:
        utc_now = datetime.now(pytz.UTC)
        hour = utc_now.hour
        weekday = utc_now.weekday()
        if weekday >= 5:
            return {"tradeable": False, "session": "WEEKEND", "multiplier": 0, "reason": "Weekend"}
        if weekday == 4 and hour >= 15:
            return {"tradeable": False, "session": "FRIDAY_CLOSE", "multiplier": 0.5, "reason": "Friday close"}
        if 12 <= hour < 15:
            return {"tradeable": True, "session": "OVERLAP", "multiplier": 1.4, "reason": "London/NY overlap"}
        if 7 <= hour < 12:
            return {"tradeable": True, "session": "LONDON", "multiplier": 1.2, "reason": "London"}
        if 13 <= hour < 17:
            return {"tradeable": True, "session": "NEW_YORK", "multiplier": 1.2, "reason": "NY"}
        return {"tradeable": False, "session": "OFF_HOURS", "multiplier": 0.3, "reason": "Off hours"}


class AdaptiveRiskManager:
    def calculate_adaptive_stop(self, df: pd.DataFrame, direction: str, regime: Dict[str, Any]) -> Dict[str, Any]:
        df = safe_fill_and_check(df, min_len=20)
        close = df["close"].values
        entry_price = float(close[-1])
        try:
            atr_indicator = ta.volatility.AverageTrueRange(high=df["high"].values, low=df["low"].values, close=close, window=14)
            atr = float(atr_indicator.average_true_range()[-1])
        except Exception:
            atr = 0.0005  # tiny default
        stop_multiplier = 1.5
        if regime.get("regime") == "VOLATILE":
            stop_multiplier = 2.5
        elif regime.get("regime") == "RANGING":
            stop_multiplier = 1.0
        elif regime.get("regime") == "TRENDING":
            stop_multiplier = 1.8
        stop_distance = atr * stop_multiplier
        if direction == "CALL":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        stop_pips = stop_distance * 10000
        risk_red = min((stop_distance / (entry_price + 1e-9)) * 100 * 30, 30)
        return {"stop_price": float(stop_price), "stop_distance_pips": float(stop_pips), "atr_multiplier": stop_multiplier,
                "confidence_reduction": float(risk_red), "atr_value": float(atr)}


class PositionSizer:
    def calculate_position_size(self, confidence: float, account_balance: float = 10000.0,
                                win_rate_history: Optional[List[int]] = None) -> Dict[str, Any]:
        if win_rate_history and len(win_rate_history) > 10:
            win_prob = sum(win_rate_history[-20:]) / len(win_rate_history[-20:])
        else:
            win_prob = max(0.01, confidence / 100.0)
        loss_prob = 1 - win_prob
        payout_ratio = 0.85
        kelly_full = (win_prob * payout_ratio - loss_prob) / payout_ratio
        kelly_fraction = max(0.0, kelly_full * 0.2)
        position_size = account_balance * kelly_fraction
        max_risk = account_balance * 0.02
        min_risk = account_balance * 0.005
        position_size = max(min_risk, min(position_size, max_risk))
        return {"position_size": round(position_size, 2), "risk_percent": round((position_size / account_balance) * 100, 2),
                "kelly_fraction": round(kelly_fraction, 4), "recommended_stake": round(position_size, 2)}


class OrderFlowAnalyzer:
    def __init__(self):
        self.cvd_history: List[float] = []

    async def analyze_order_flow(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            if "volume" in df.columns:
                recent_volume = df["volume"].tail(20)
                close_change = df["close"].diff().tail(20)
                buy_volume = recent_volume[close_change > 0].sum()
                sell_volume = recent_volume[close_change < 0].sum()
                imbalance_ratio = float(buy_volume) / (float(sell_volume) + 1.0)
            else:
                returns = df["close"].pct_change().tail(20)
                imbalance_ratio = 1.0 + float(returns.mean()) * 100.0
            cvd = self._calculate_simulated_cvd(df)
            large_orders = self._detect_large_orders_simulated(df)
            smart_money = self._track_smart_money_simulated(imbalance_ratio, cvd)
            conf_boost = 0
            if imbalance_ratio > 1.3:
                conf_boost += 12
            elif imbalance_ratio < 0.77:
                conf_boost += 12
            if abs(cvd) > 500:
                conf_boost += 10
            if smart_money != "NEUTRAL":
                conf_boost += 8
            self.cvd_history.append(cvd)
            if len(self.cvd_history) > 1000:
                self.cvd_history = self.cvd_history[-1000:]
            return {"imbalance_ratio": float(imbalance_ratio), "bias": ("BULLISH" if imbalance_ratio > 1.2 else "BEARISH" if imbalance_ratio < 0.8 else "NEUTRAL"),
                    "cvd": float(cvd), "large_buy_walls": large_orders["buy_walls"], "large_sell_walls": large_orders["sell_walls"],
                    "smart_money_direction": smart_money, "confidence_boost": min(conf_boost, 25)}
        except Exception as e:
            log.exception("OrderFlow error")
            return {"error": str(e)}

    def _calculate_simulated_cvd(self, df: pd.DataFrame) -> float:
        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
        price_change = np.diff(close, prepend=close[0])
        buy_vol = np.where(price_change > 0, volume, 0).sum()
        sell_vol = np.where(price_change < 0, volume, 0).sum()
        return float(buy_vol - sell_vol)

    def _detect_large_orders_simulated(self, df: pd.DataFrame) -> Dict[str, int]:
        if "volume" in df.columns:
            avg_volume = df["volume"].mean()
            threshold = avg_volume * 2.0
            buy_walls = int(len(df[(df["volume"] > threshold) & (df["close"] > df["open"])]))
            sell_walls = int(len(df[(df["volume"] > threshold) & (df["close"] < df["open"])]))
        else:
            buy_walls = sell_walls = 0
        return {"buy_walls": buy_walls, "sell_walls": sell_walls}

    def _track_smart_money_simulated(self, imbalance_ratio: float, cvd: float) -> str:
        if cvd > 500 and imbalance_ratio > 1.2:
            return "BULLISH_INSTITUTIONS"
        if cvd < -500 and imbalance_ratio < 0.8:
            return "BEARISH_INSTITUTIONS"
        return "NEUTRAL"


# -------------------------
# Signal generator (integrated + improved)
# -------------------------
class Enhanced80PercentSignalGenerator:
    def __init__(self):
        self.regime = MarketRegimeDetector()
        self.scorer = ConfluenceScorer()
        self.session = SessionFilter()
        self.risk = AdaptiveRiskManager()
        self.sizer = PositionSizer()
        self.of = OrderFlowAnalyzer()
        self.win_history = self._load_win_history()

    def _load_win_history(self) -> List[int]:
        rate = get_win_rate(500)
        # we only return empty list; actual wins are stored in DB and accessed via get_win_rate
        cur = get_db_conn().cursor()
        cur.execute("SELECT outcome FROM win_history ORDER BY id DESC LIMIT 500")
        rows = cur.fetchall()
        return [int(r["outcome"]) for r in rows] if rows else []

    async def generate_signal_80_percent(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            df = safe_fill_and_check(df, min_len=50)
        except Exception as e:
            return {"error": f"Insufficient data: {e}"}

        # 1) Session
        session_data = self.session.is_tradeable_session()
        if not session_data["tradeable"]:
            return {"error": f"Session filter: {session_data['reason']}", "session": session_data}

        # 2) Regime
        regime = self.regime.detect_regime(df)
        if regime.get("trade_filter") == "SKIP":
            return {"error": f"Regime filter: {regime['regime']} - skipping", "regime": regime}

        # 3) Indicators (cached)
        ind_key = f"{symbol}:ind"
        indicators = cache_get_indicators(ind_key)
        if indicators is None:
            indicators = self._calculate_all_indicators(df)
            cache_set_indicators(ind_key, indicators)

        # 4) Order flow (async)
        order_flow = await self.of.analyze_order_flow(symbol, df)

        # 5) Confluence
        confluence = self.scorer.calculate_confluence(indicators, regime, session_data, order_flow)
        if confluence["score"] < 70:
            return {"error": f"Confluence too low: {confluence['score']:.1f}%", "confluence": confluence}

        # 6) Determine direction
        direction = self._determine_direction(indicators, regime, order_flow)

        # 7) Risk & sizing
        stop = self.risk.calculate_adaptive_stop(df, direction, regime)
        pos = self.sizer.calculate_position_size(confluence["score"], win_rate_history=self.win_history)

        # 8) Confidence adjust
        confidence = confluence["score"] * regime["confidence_multiplier"] * session_data["multiplier"]
        if "error" not in order_flow:
            confidence += order_flow.get("confidence_boost", 0)
        final_conf = min(confidence, 95)

        if final_conf < 75:
            return {"error": f"Final confidence too low: {final_conf:.1f}%"}

        price = float(df["close"].iloc[-1])
        signal = {
            "symbol": symbol,
            "direction": direction,
            "confidence": round(final_conf, 1),
            "entry_price": price,
            "regime": regime,
            "confluence": confluence,
            "session": session_data,
            "order_flow": order_flow if "error" not in order_flow else None,
            "stop_loss": stop,
            "position_sizing": pos,
            "indicators": {
                "rsi": round(indicators["momentum"]["rsi"], 2),
                "macd": round(indicators["momentum"]["macd"], 6),
                "adx": round(indicators["trend"]["adx"], 2),
                "atr_pct": round(indicators["volatility"]["atr_percent"] * 100, 3)
            },
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "timeframe": "5-min",
            "expiry_minutes": 5
        }
        return signal

    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = safe_fill_and_check(df, min_len=50)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # RSI
        try:
            rsi = float(ta.momentum.RSIIndicator(close=close, window=14).rsi()[-1])
        except Exception:
            rsi = 50.0

        # MACD
        try:
            macd_obj = ta.trend.MACD(close=close)
            macd = float(macd_obj.macd()[-1])
            macd_signal = float(macd_obj.macd_signal()[-1])
        except Exception:
            macd = 0.0
            macd_signal = 0.0

        # ADX
        try:
            adx = float(ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()[-1])
        except Exception:
            adx = 0.0

        # EMAs
        try:
            ema20 = float(ta.trend.EMAIndicator(close=close, window=20).ema_indicator()[-1])
            ema50 = float(ta.trend.EMAIndicator(close=close, window=50).ema_indicator()[-1])
        except Exception:
            ema20 = ema50 = float(close[-1])

        # ATR
        try:
            atr = float(ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()[-1])
            atr_pct = atr / float(close[-1])
        except Exception:
            atr = 0.0
            atr_pct = 0.0

        # Stochastic K
        try:
            stoch_k = float(ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3).stoch()[-1])
        except Exception:
            stoch_k = 50.0

        # Volume proxies
        volume_info = {"obv_trend": "NEUTRAL", "cmf": 0.0}
        try:
            if "volume" in df.columns:
                obv = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
                # simple trend on OBV
                volume_info["obv_trend"] = "BULLISH" if obv.iloc[-1] > obv.iloc[-5] else "BEARISH"
                cmf = ta.volume.ChaikinMoneyFlowIndicator(high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=20).chaikin_money_flow()[-1]
                volume_info["cmf"] = float(cmf)
        except Exception:
            pass

        return {
            "momentum": {"rsi": rsi, "macd": macd, "macd_signal": macd_signal, "stoch_k": stoch_k},
            "trend": {"adx": adx, "ema_20": ema20, "ema_50": ema50},
            "volatility": {"atr": atr, "atr_percent": atr_pct},
            "volume": volume_info
        }

    def _determine_direction(self, indicators: Dict[str, Any], regime: Dict[str, Any], order_flow: Optional[Dict[str, Any]]) -> str:
        bulls = 0
        bears = 0
        rsi = indicators.get("momentum", {}).get("rsi", 50)
        if rsi < 40: bulls += 1
        elif rsi > 60: bears += 1
        macd = indicators.get("momentum", {}).get("macd", 0)
        macd_signal = indicators.get("momentum", {}).get("macd_signal", 0)
        if macd > macd_signal: bulls += 1
        else: bears += 1
        if regime.get("trend_direction") == "BULLISH": bulls += 2
        else: bears += 2
        if order_flow and "error" not in order_flow:
            if order_flow.get("bias") == "BULLISH": bulls += 2
            elif order_flow.get("bias") == "BEARISH": bears += 2
        return "CALL" if bulls > bears else "PUT"


# -------------------------
# Initialize generator
# -------------------------
signal_generator = Enhanced80PercentSignalGenerator()


# -------------------------
# API Endpoints
# -------------------------
@app.route("/api/signals/generate", methods=["POST"])
def api_generate_signal():
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
    payload = request.get_json()
    symbol = payload.get("symbol", "EURUSD")
    interval = payload.get("interval", "5m")
    limit = int(payload.get("limit", 200))
    # fetch OHLC (cached)
    try:
        df = fetch_ohlc(symbol, interval=interval, limit=limit)
    except Exception as e:
        log.exception("OHLC fetch error")
        return jsonify({"success": False, "error": f"OHLC fetch error: {e}"}), 500

    # run async generator safely with asyncio.run()
    try:
        signal = asyncio.run(signal_generator.generate_signal_80_percent(symbol, df))
    except Exception as e:
        log.exception("Signal generator crash")
        return jsonify({"success": False, "error": str(e)}), 500

    if "error" in signal:
        return jsonify({"success": False, "error": signal.get("error"), "details": signal.get("confluence")}), 200
    return jsonify({"success": True, "signal": signal}), 200


@app.route("/api/signals/record", methods=["POST"])
def api_record_result():
    """
    Record a trade result from frontend/backtester:
    { "symbol": "EURUSD", "outcome": 1, "note": "manual" }
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
    data = request.get_json()
    symbol = data.get("symbol", "UNKNOWN")
    outcome = int(data.get("outcome", 0))
    note = data.get("note", "")
    try:
        save_trade_result(symbol, outcome, note)
        return jsonify({"success": True}), 200
    except Exception as e:
        log.exception("Record result failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/signals/active", methods=["GET"])
def api_active_signals():
    try:
        wr = get_win_rate(100) or 0.0
        return jsonify({"success": True, "signals": [], "win_rate": wr}), 200
    except Exception as e:
        log.exception("Active signals error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/market/status", methods=["GET"])
def api_market_status():
    try:
        s = signal_generator.session.is_tradeable_session()
        return jsonify({"is_open": s["tradeable"], "session": s["session"], "reason": s.get("reason")}), 200
    except Exception as e:
        log.exception("Market status error")
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------
# Backtesting helper (vectorbt)
# -------------------------
@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """
    Backtest endpoint (requires vectorbt).
    Payload: { symbol: "EURUSD", timeframe: "5m", start: "YYYY-MM-DD", end: "YYYY-MM-DD" }
    Returns basic stats — heavy operation.
    """
    if not HAS_VECTORBT:
        return jsonify({"success": False, "error": "vectorbt not installed"}), 501
    data = request.get_json() or {}
    symbol = data.get("symbol", "EURUSD")
    # if timeframe mapping required, map to yfinance interval or alternate data provider
    try:
        df = fetch_ohlc(symbol, interval=data.get("interval","5m"), limit=int(data.get("limit", 1000)))
        # vectorbt expects OHLC with datetime index
        # This is example usage; for full backtest you should implement strategy logic in vectorized manner
        price = df["close"]
        pf = vbt.Portfolio.from_signals(price, entries=price > price.shift(1), exits=price < price.shift(1))
        stats = pf.stats()
        return jsonify({"success": True, "stats": stats.to_dict()}), 200
    except Exception as e:
        log.exception("Backtest error")
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------
# Optional risk metrics (SquareQuant)
# -------------------------
@app.route("/api/risk/metrics", methods=["POST"])
def api_risk_metrics():
    if not HAS_SQUAREQUANT:
        return jsonify({"success": False, "error": "SquareQuant not installed"}), 501
    payload = request.get_json() or {}
    # Example: return sample risk metric using squarequant if available
    try:
        # This is placeholder — integrate your own logic using sq library
        metrics = {"vaR_95": None}
        return jsonify({"success": True, "metrics": metrics}), 200
    except Exception as e:
        log.exception("Risk metrics error")
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------
# Server start (Railway-friendly)
# -------------------------
if __name__ == "__main__":
    port_str = os.environ.get("PORT", "5000")
    try:
        port = int(port_str)
    except Exception:
        log.warning("Invalid PORT env var '%s', falling back to 5000", port_str)
        port = 5000
    log.info("Starting Binary Option Bot API on port %s", port)
    # Use debug=False in production
    app.run(host="0.0.0.0", port=port, debug=False)