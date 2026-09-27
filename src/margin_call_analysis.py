import warnings

# Suppress urllib3 v2 NotOpenSSLWarning on macOS LibreSSL environments
warnings.filterwarnings("ignore", message=r".*urllib3 v2 only supports OpenSSL 1\.1\.1\+.*")
try:
    import urllib3
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

import argparse
import json
import os
import sys
from pathlib import Path
import mlx.core as mx
import numpy as np
import yfinance as yf

# Ensure project root is in sys.path when executed directly from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.train_djia_stock_model import (
        DJIA_TICKERS,
        StockPredictionModel as DjiaStockModel,
        STATS_FILE as DJIA_STATS_FILE,
        WEIGHTS_FILE as DJIA_WEIGHTS_FILE
    )
    from src.train_tech_stock_model import (
        TECH_TICKERS,
        StockPredictionModel as TechStockModel,
        STATS_FILE as TECH_STATS_FILE,
        WEIGHTS_FILE as TECH_WEIGHTS_FILE
    )
except ImportError:
    from train_djia_stock_model import (
        DJIA_TICKERS,
        StockPredictionModel as DjiaStockModel,
        STATS_FILE as DJIA_STATS_FILE,
        WEIGHTS_FILE as DJIA_WEIGHTS_FILE
    )
    from train_tech_stock_model import (
        TECH_TICKERS,
        StockPredictionModel as TechStockModel,
        STATS_FILE as TECH_STATS_FILE,
        WEIGHTS_FILE as TECH_WEIGHTS_FILE
    )


def load_model_and_stats(model_cls, weights_file, stats_file):
    """Loads a trained MLX model and its normalization statistics from data/."""
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Stats file '{stats_file}' not found. Please train the model first.")
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weights file '{weights_file}' not found. Please train the model first.")

    with open(stats_file, "r") as f:
        stats = json.load(f)
        norm_mean = np.array(stats["mean"], dtype=np.float32)
        norm_std = np.array(stats["std"], dtype=np.float32)

    model = model_cls(input_dim=4)
    model.load_weights(weights_file)
    mx.eval(model.parameters())
    return model, norm_mean, norm_std


def fetch_and_evaluate_stock(ticker, model, norm_mean, norm_std, segment="DJIA"):
    """Fetches market history for a ticker, calculates technical features,

    runs neural network inference, and computes margin risk / volatility metrics.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period="60d")
    if df.empty or len(df) < 30:
        raise ValueError(f"Insufficient historical data for {ticker}")

    # Feature 1: 14-period RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Feature 2: MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # Feature 3: Volume Trend
    df['Volume_Trend'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # Feature 4: Daily Return (%)
    df['Daily_Return'] = df['Close'].pct_change() * 100

    # Clean metrics
    valid_df = df[['Close', 'RSI', 'MACD', 'Volume_Trend', 'Daily_Return']].dropna()
    latest_row = valid_df.iloc[-1]
    latest_price = float(latest_row['Close'])
    latest_features = latest_row[['RSI', 'MACD', 'Volume_Trend', 'Daily_Return']].values

    # Model inference
    norm_features = (latest_features - norm_mean) / norm_std
    input_tensor = mx.array(norm_features.reshape(1, -1).astype(np.float32))
    raw_logits = model(input_tensor)
    bullish_prob = float(mx.sigmoid(raw_logits).item() * 100)
    bearish_prob = 100.0 - bullish_prob

    # Margin Risk & Volatility Metrics
    returns_30d = df['Close'].pct_change().dropna().tail(30)
    annualized_volatility = float(returns_30d.std() * np.sqrt(252) * 100) if len(returns_30d) > 5 else 0.0
    recent_range_pct = float(((df['High'] - df['Low']) / df['Close']).tail(14).mean() * 100)

    # Margin Call Score (Bearish Short Vulnerability):
    # Higher score = High probability of steep decline + high volatility (highest risk of margin call for longs)
    # Composite: Bearish probability weighted by volatility factor and downward MACD pressure
    macd_val = float(latest_features[1])
    vol_trend_val = float(latest_features[2])
    rsi_val = float(latest_features[0])

    # Downside acceleration factor
    downside_factor = (bearish_prob / 100.0)
    volatility_weight = min(max(annualized_volatility / 30.0, 0.5), 2.0)
    margin_short_score = float(downside_factor * 100.0 * (1.0 + (volatility_weight - 1.0) * 0.3))

    # Leveraged Margin Buy Score:
    # High bullish probability + positive momentum + manageable volatility
    upside_factor = (bullish_prob / 100.0)
    margin_buy_score = float(upside_factor * 100.0 * (1.0 + (volatility_weight - 1.0) * 0.2))

    # Categorize Action
    if margin_short_score >= 65.0 and bearish_prob >= 60.0:
        margin_action = "MARGIN SHORT 🚨"
    elif margin_buy_score >= 65.0 and bullish_prob >= 60.0:
        margin_action = "MARGIN BUY 🚀"
    elif annualized_volatility >= 45.0:
        margin_action = "HIGH RISK MARGIN ⚠️"
    else:
        margin_action = "NEUTRAL ⏸️"

    return {
        "ticker": ticker,
        "segment": segment,
        "price": latest_price,
        "bullish_prob": bullish_prob,
        "bearish_prob": bearish_prob,
        "rsi": rsi_val,
        "macd": macd_val,
        "vol_trend": vol_trend_val,
        "volatility_30d": annualized_volatility,
        "recent_range_pct": recent_range_pct,
        "margin_short_score": margin_short_score,
        "margin_buy_score": margin_buy_score,
        "margin_action": margin_action
    }


def analyze_margin_calls(period="2y", top_n=5):
    """Evaluates DJIA and High-Tech models to identify the best stocks for margin calls,

    short margin targets, and leveraged margin buys.
    """
    print("=" * 75)
    print("⚡ DOW JONES & HIGH-TECH CROSS-MODEL MARGIN CALL ANALYSIS")
    print("=" * 75)
    print("📥 Loading trained DJIA and High-Tech models from 'data/'...")

    djia_model, djia_mean, djia_std = load_model_and_stats(
        DjiaStockModel, DJIA_WEIGHTS_FILE, DJIA_STATS_FILE
    )
    tech_model, tech_mean, tech_std = load_model_and_stats(
        TechStockModel, TECH_WEIGHTS_FILE, TECH_STATS_FILE
    )

    all_results = []
    processed_tickers = set()

    # 1. Evaluate DJIA Universe
    print(f"\n🔍 Analyzing {len(DJIA_TICKERS)} DJIA component stocks...")
    for ticker in DJIA_TICKERS:
        try:
            res = fetch_and_evaluate_stock(
                ticker, djia_model, djia_mean, djia_std, segment="DJIA"
            )
            all_results.append(res)
            processed_tickers.add(ticker)
        except Exception as e:
            print(f"  ❌ Error analyzing {ticker}: {e}")

    # 2. Evaluate High-Tech Universe (deduplicating any overlap)
    tech_unique = [t for t in TECH_TICKERS if t not in processed_tickers]
    print(f"\n🔍 Analyzing {len(tech_unique)} High-Tech stocks...")
    for ticker in tech_unique:
        try:
            res = fetch_and_evaluate_stock(
                ticker, tech_model, tech_mean, tech_std, segment="Tech"
            )
            all_results.append(res)
            processed_tickers.add(ticker)
        except Exception as e:
            print(f"  ❌ Error analyzing {ticker}: {e}")

    if not all_results:
        raise RuntimeError("No stock data could be analyzed.")

    # Sort by margin call / short score descending
    short_margin_ranked = sorted(all_results, key=lambda x: x["margin_short_score"], reverse=True)
    # Sort by leveraged margin buy score descending
    buy_margin_ranked = sorted(all_results, key=lambda x: x["margin_buy_score"], reverse=True)
    # Sort by volatility descending
    volatility_ranked = sorted(all_results, key=lambda x: x["volatility_30d"], reverse=True)

    # Display Highlights
    best_short_margin = short_margin_ranked[0]
    best_buy_margin = buy_margin_ranked[0]

    print("\n" + "=" * 75)
    print("🏆 BEST STOCKS TO PLACE MARGIN CALLS / LEVERAGED POSITIONS")
    print("=" * 75)

    print(f"\n🚨 #1 BEST STOCK TO PLACE MARGIN CALLS (Bearish Short / Downside Vulnerability):")
    print(f"   Ticker:             {best_short_margin['ticker']} ({best_short_margin['segment']})")
    print(f"   Current Price:      ${best_short_margin['price']:.2f}")
    print(f"   Bearish Confidence: {best_short_margin['bearish_prob']:.2f}% (Bullish: {best_short_margin['bullish_prob']:.2f}%)")
    print(f"   Margin Short Score: {best_short_margin['margin_short_score']:.2f} / 100")
    print(f"   30-Day Volatility:  {best_short_margin['volatility_30d']:.2f}% (Annualized)")
    print(f"   Technical Profile:  RSI: {best_short_margin['rsi']:.1f} | MACD: {best_short_margin['macd']:.2f} | Action: {best_short_margin['margin_action']}")
    print(f"   💡 Rationale:        High downside probability combined with volatility creates the greatest")
    print(f"                        downward leverage potential and highest risk of triggering margin calls for longs.")

    print(f"\n🚀 #1 BEST STOCK FOR LEVERAGED MARGIN BUY (Bullish Breakout Potential):")
    print(f"   Ticker:             {best_buy_margin['ticker']} ({best_buy_margin['segment']})")
    print(f"   Current Price:      ${best_buy_margin['price']:.2f}")
    print(f"   Bullish Confidence: {best_buy_margin['bullish_prob']:.2f}%")
    print(f"   Margin Buy Score:   {best_buy_margin['margin_buy_score']:.2f} / 100")
    print(f"   30-Day Volatility:  {best_buy_margin['volatility_30d']:.2f}% (Annualized)")
    print(f"   Technical Profile:  RSI: {best_buy_margin['rsi']:.1f} | MACD: {best_buy_margin['macd']:.2f} | Action: {best_buy_margin['margin_action']}")
    print(f"   💡 Rationale:        Strongest neural momentum signal with optimal volume support for leveraged long entries.")

    # Top 5 Margin Short / Downside Margin Call Targets
    print("\n" + "=" * 75)
    print("🔻 TOP 5 MARGIN CALL / SHORT TARGETS (Highest Downside Risk):")
    print(f"{'Rank':<6}{'Segment':<9}{'Ticker':<8}{'Price':<10}{'Bearish %':<12}{'Vol 30d':<10}{'Margin Score':<14}{'Action':<15}")
    print("-" * 75)
    for rank, item in enumerate(short_margin_ranked[:top_n], start=1):
        print(f"{rank:<6}{item['segment']:<9}{item['ticker']:<8}${item['price']:<9.2f}"
              f"{item['bearish_prob']:>6.2f}%     {item['volatility_30d']:>6.1f}%    "
              f"{item['margin_short_score']:>7.2f}       {item['margin_action']}")

    # Top 5 Leveraged Margin Buys
    print("\n" + "=" * 75)
    print("🚀 TOP 5 LEVERAGED MARGIN BUYS (Highest Bullish Conviction):")
    print(f"{'Rank':<6}{'Segment':<9}{'Ticker':<8}{'Price':<10}{'Bullish %':<12}{'Vol 30d':<10}{'Margin Score':<14}{'Action':<15}")
    print("-" * 75)
    for rank, item in enumerate(buy_margin_ranked[:top_n], start=1):
        print(f"{rank:<6}{item['segment']:<9}{item['ticker']:<8}${item['price']:<9.2f}"
              f"{item['bullish_prob']:>6.2f}%     {item['volatility_30d']:>6.1f}%    "
              f"{item['margin_buy_score']:>7.2f}       {item['margin_action']}")

    # Top 5 High Volatility Margin Alert Zone
    print("\n" + "=" * 75)
    print("⚠️ TOP 5 HIGH-VOLATILITY MARGIN ALERT STOCKS (Extreme Swings):")
    print(f"{'Rank':<6}{'Segment':<9}{'Ticker':<8}{'Price':<10}{'Vol 30d':<12}{'Bullish %':<12}{'Bearish %':<12}{'Action':<15}")
    print("-" * 75)
    for rank, item in enumerate(volatility_ranked[:top_n], start=1):
        print(f"{rank:<6}{item['segment']:<9}{item['ticker']:<8}${item['price']:<9.2f}"
              f"{item['volatility_30d']:>6.1f}%      {item['bullish_prob']:>6.2f}%     "
              f"{item['bearish_prob']:>6.2f}%     {item['margin_action']}")
    print("=" * 75)

    return short_margin_ranked, buy_margin_ranked


def main():
    parser = argparse.ArgumentParser(
        description="Trades: Cross-model margin call and leveraged stock analysis combining DJIA and Tech neural networks."
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of top margin candidates to display in summary")
    args = parser.parse_args()

    analyze_margin_calls(top_n=args.top_n)


if __name__ == "__main__":
    main()
