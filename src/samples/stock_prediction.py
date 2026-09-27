import warnings

# Suppress urllib3 v2 NotOpenSSLWarning on macOS LibreSSL environments
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL 1\.1\.1\+.*")
try:
    import urllib3
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

import json
import os
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yfinance as yf


# 1. Define the identical structural shell (Must match your training architecture)
class StockPredictionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def __call__(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)


# 2. Reconstruct Model Structure and Load Weights
INPUT_FEATURES = 4
model = StockPredictionModel(input_dim=INPUT_FEATURES)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

WEIGHTS_FILE = str(DATA_DIR / "djia_stock_model.safetensors")
STATS_FILE = str(DATA_DIR / "normalization_djia_stats.json")

# Fallback check if in root
if not os.path.exists(WEIGHTS_FILE) and os.path.exists("djia_stock_model.safetensors"):
    WEIGHTS_FILE = "djia_stock_model.safetensors"
if not os.path.exists(STATS_FILE) and os.path.exists("normalization_djia_stats.json"):
    STATS_FILE = "normalization_djia_stats.json"

if os.path.exists(WEIGHTS_FILE):
    model.load_weights(WEIGHTS_FILE)
    mx.eval(model.parameters())
    print(f"✓ Loaded trained model weights from '{WEIGHTS_FILE}'")
else:
    print(f"⚠️ Warning: Weights file '{WEIGHTS_FILE}' not found. Please train the model first.")

norm_mean = None
norm_std = None
if os.path.exists(STATS_FILE):
    with open(STATS_FILE, "r") as f:
        stats = json.load(f)
        norm_mean = np.array(stats["mean"], dtype=np.float32)
        norm_std = np.array(stats["std"], dtype=np.float32)
        print(f"✓ Loaded training normalization stats from '{STATS_FILE}'")


# 3. Live Data Engineering Function (Strictly matching training logic)
def fetch_live_features(ticker_symbol):
    print(f"📥 Fetching data for {ticker_symbol} from Yahoo Finance...")
    # Fetch 60 days of data to safely calculate a stable 14-period RSI and 26-period MACD
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="60d")

    if df.empty:
        raise ValueError(f"Could not retrieve data for ticker: {ticker_symbol}")

    # --- Feature 1: 14-Period RSI ---
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- Feature 2: MACD (12-period EMA - 26-period EMA) ---
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # --- Feature 3: Volume Trend (Current Volume / 20-day Average) ---
    df['Volume_Trend'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # --- Feature 4: Past 24h Return (%) ---
    df['Daily_Return'] = df['Close'].pct_change() * 100

    # --- Feature Standardization ---
    features_df = df[['RSI', 'MACD', 'Volume_Trend', 'Daily_Return']].dropna()
    latest_metrics = features_df.iloc[-1].values  # Get the absolute most recent market row

    if norm_mean is not None and norm_std is not None:
        standardized_features = (latest_metrics - norm_mean) / norm_std
    else:
        # Fallback to local sample standardization if global stats are unavailable
        standardized_features = (latest_metrics - features_df.mean().values) / (features_df.std().values + 1e-9)

    return np.array([standardized_features], dtype=np.float32)


# 4. Run Execution Pipeline for DJIA Stocks
DJIA_TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
]

TRADING_THRESHOLD = 75.0
predictions = []

print("=" * 60)
print("📊 Analyzing DJIA Stocks for Next Opening Bell Breakouts")
print("=" * 60)

for ticker in DJIA_TICKERS:
    try:
        live_features = fetch_live_features(ticker)

        # Convert feature vector into an MLX Tensor
        input_tensor = mx.array(live_features)

        # Run structural inference
        raw_logits = model(input_tensor)
        probability = mx.sigmoid(raw_logits).item() * 100

        # Assign trading signal based on probabilities
        if probability >= TRADING_THRESHOLD:
            signal = "BUY 🚀"
        elif probability <= (100.0 - TRADING_THRESHOLD):
            signal = "SELL 🔻"
        elif probability >= 55.0:
            signal = "BUY (Moderate) 📈"
        elif probability <= 45.0:
            signal = "SELL (Moderate) 📉"
        else:
            signal = "HOLD / SKIP ⏸️"

        predictions.append({
            "ticker": ticker,
            "probability": probability,
            "signal": signal
        })
        print(f"  ✓ {ticker}: {probability:.2f}% ({signal})")

    except Exception as e:
        print(f"  ❌ Error fetching/analyzing {ticker}: {e}")

# Sort stocks by probability in descending order
predictions.sort(key=lambda x: x["probability"], reverse=True)

print("\n" + "=" * 60)
print("🏆 DJIA COMPLETE STOCKS RANKING (Next Opening Bell Forecast)")
print("=" * 60)
print(f"{'Rank':<6}{'Ticker':<10}{'Confidence':<16}{'Signal':<15}")
print("-" * 60)

for rank, item in enumerate(predictions, start=1):
    print(f"{rank:<6}{item['ticker']:<10}{item['probability']:>6.2f}%         {item['signal']:<15}")

# 1. Top 5 to BUY
top_5_buy = predictions[:5]
print("\n" + "=" * 60)
print("🚀 TOP 5 TO BUY (Highest Bullish Confidence):")
print("-" * 60)
for rank, pick in enumerate(top_5_buy, start=1):
    print(f"  {rank}. {pick['ticker']:<6} - {pick['probability']:>6.2f}% ({pick['signal']})")

# 2. Top 5 to SELL
top_5_sell = sorted(predictions, key=lambda x: x["probability"])[:5]
print("\n" + "=" * 60)
print("🔻 TOP 5 TO SELL (Lowest Probability / Bearish Momentum):")
print("-" * 60)
for rank, pick in enumerate(top_5_sell, start=1):
    print(f"  {rank}. {pick['ticker']:<6} - {pick['probability']:>6.2f}% ({pick['signal']})")

# 3. Top 5 to HOLD / SKIP
top_5_hold = sorted(predictions, key=lambda x: abs(x["probability"] - 50.0))[:5]
print("\n" + "=" * 60)
print("⏸️ TOP 5 TO HOLD / SKIP (Closest to Neutral 50% Momentum):")
print("-" * 60)
for rank, pick in enumerate(top_5_hold, start=1):
    print(f"  {rank}. {pick['ticker']:<6} - {pick['probability']:>6.2f}% ({pick['signal']})")
print("=" * 60)
