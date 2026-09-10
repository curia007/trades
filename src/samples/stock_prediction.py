import mlx.core as mx
import mlx.nn as nn
import numpy as nn_module
import numpy as np
import pandas as pd
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
saved_weights = mx.load("stock_model.safetensors")
model.update(saved_weights)
mx.eval(model.parameters())


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

    # --- Z-Score Feature Standardization (Simulation of production scaling) ---
    # Note: In production, always use the explicit Mean/STD calculated from your
    # original historical training set, NOT a dynamic sample mean.
    features_df = df[['RSI', 'MACD', 'Volume_Trend', 'Daily_Return']].dropna()
    latest_metrics = features_df.iloc[-1].values  # Get the absolute most recent market row

    # Simple Z-Score scale baseline matching original normalization format
    standardized_features = (latest_metrics - features_df.mean().values) / (features_df.std().values + 1e-9)

    return np.array([standardized_features], dtype=np.float32)


# 4. Run Execution Pipeline
try:
    TICKER = "AAPL"  # Change to any symbol you want to target (e.g., NVDA, SPY, TSLA)
    live_features = fetch_live_features(TICKER)

    # Convert feature vector into an MLX Tensor
    input_tensor = mx.array(live_features)

    # Run structural inference
    raw_logits = model(input_tensor)
    probability = mx.sigmoid(raw_logits).item() * 100

    print(f"\n--- 📈 {TICKER} Next-Day Forecast ---")
    print(f"Confidence score for a major breakout move: {probability:.2f}%")

    TRADING_THRESHOLD = 75.0
    if probability >= TRADING_THRESHOLD:
        print(f"🚀 SIGNAL: BUY {TICKER}. Strong breakout signature detected.")
    else:
        print(f"⚠️ SIGNAL: HOLD / SKIP {TICKER}. Below target momentum thresholds.")

except Exception as e:
    print(f"❌ Error compiling pipeline metrics: {e}")
