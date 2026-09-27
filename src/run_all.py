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
import numpy as np

# Ensure project root is in sys.path when executed directly from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.train_djia_stock_model import (
        DJIA_TICKERS,
        StockPredictionModel as DjiaStockModel,
        build_historical_dataset as build_djia_dataset,
        train_model as train_djia_model,
        predict_best_stocks_for_next_open as predict_djia_stocks,
        STATS_FILE as DJIA_STATS_FILE,
        WEIGHTS_FILE as DJIA_WEIGHTS_FILE
    )
    from src.train_tech_stock_model import (
        TECH_TICKERS,
        StockPredictionModel as TechStockModel,
        build_historical_dataset as build_tech_dataset,
        train_model as train_tech_model,
        predict_best_stocks_for_next_open as predict_tech_stocks,
        STATS_FILE as TECH_STATS_FILE,
        WEIGHTS_FILE as TECH_WEIGHTS_FILE
    )
except ImportError:
    from train_djia_stock_model import (
        DJIA_TICKERS,
        StockPredictionModel as DjiaStockModel,
        build_historical_dataset as build_djia_dataset,
        train_model as train_djia_model,
        predict_best_stocks_for_next_open as predict_djia_stocks,
        STATS_FILE as DJIA_STATS_FILE,
        WEIGHTS_FILE as DJIA_WEIGHTS_FILE
    )
    from train_tech_stock_model import (
        TECH_TICKERS,
        StockPredictionModel as TechStockModel,
        build_historical_dataset as build_tech_dataset,
        train_model as train_tech_model,
        predict_best_stocks_for_next_open as predict_tech_stocks,
        STATS_FILE as TECH_STATS_FILE,
        WEIGHTS_FILE as TECH_WEIGHTS_FILE
    )


def run_pipeline(target_name, tickers, build_dataset_fn, train_model_fn, predict_fn,
                 stats_file, weights_file, model_cls, mode, period, epochs, batch_size, lr, threshold):
    print("\n" + "#" * 70)
    print(f"## RUNNING {target_name.upper()} PIPELINE (Mode: {mode})")
    print("#" * 70)

    model = model_cls(input_dim=4)
    norm_mean = None
    norm_std = None

    if mode in ["train-and-predict", "train"]:
        print(f"\n>>> [1/2] Building historical dataset & training {target_name.upper()} model...")
        X_train, y_train, X_val, y_val, norm_mean, norm_std = build_dataset_fn(
            tickers=tickers, period=period
        )
        train_model_fn(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

    if mode in ["train-and-predict", "predict"]:
        print(f"\n>>> [2/2] Generating {target_name.upper()} predictions...")
        if norm_mean is None or norm_std is None:
            if os.path.exists(stats_file):
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                    norm_mean = np.array(stats["mean"], dtype=np.float32)
                    norm_std = np.array(stats["std"], dtype=np.float32)
            else:
                raise FileNotFoundError(f"'{stats_file}' not found. Please train the model first.")

            if os.path.exists(weights_file):
                model.load_weights(weights_file)
            else:
                raise FileNotFoundError(f"'{weights_file}' not found. Please train the model first.")

        results = predict_fn(
            model=model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            tickers=tickers,
            threshold=threshold
        )
        return results

    return None


def print_combined_market_summary(djia_results, tech_results):
    if not djia_results and not tech_results:
        return

    print("\n" + "=" * 70)
    print("🌟 DOW JONES & HIGH-TECH COMBINED MARKET HIGHLIGHTS")
    print("=" * 70)

    if djia_results:
        best_djia = djia_results[0]
        print(f"🏆 Top DJIA Pick:        {best_djia['ticker']:<6} | Conf: {best_djia['probability']:>6.2f}% | "
              f"Price: ${best_djia['price']:>7.2f} | Signal: {best_djia['signal']}")

    if tech_results:
        best_tech = tech_results[0]
        print(f"🏆 Top High-Tech Pick:   {best_tech['ticker']:<6} | Conf: {best_tech['probability']:>6.2f}% | "
              f"Price: ${best_tech['price']:>7.2f} | Signal: {best_tech['signal']}")

    # Overall top 5 across both markets
    combined = []
    if djia_results:
        for item in djia_results:
            combined.append({**item, "segment": "DJIA"})
    if tech_results:
        for item in tech_results:
            combined.append({**item, "segment": "Tech"})

    if combined:
        combined.sort(key=lambda x: x["probability"], reverse=True)
        print("\n" + "-" * 70)
        print("🚀 TOP 5 PICKS ACROSS BOTH MARKETS (DJIA + HIGH-TECH):")
        print(f"{'Rank':<6}{'Segment':<10}{'Ticker':<8}{'Price':<10}{'Confidence':<14}{'Signal':<15}")
        print("-" * 70)
        for rank, item in enumerate(combined[:5], start=1):
            print(f"{rank:<6}{item['segment']:<10}{item['ticker']:<8}${item['price']:<9.2f}"
                  f"{item['probability']:>6.2f}%       {item['signal']:<15}")
        print("=" * 70)


def run_all_pipelines(mode="train-and-predict", period="2y", epochs=60,
                      batch_size=64, lr=1e-3, threshold=75.0):
    print("=" * 70)
    print("🚀 STARTING DOW JONES & HIGH-TECH DUAL STOCK PIPELINE")
    print("=" * 70)

    # 1. Run DJIA Pipeline
    djia_results = run_pipeline(
        target_name="DJIA (Dow Jones Industrial Average)",
        tickers=DJIA_TICKERS,
        build_dataset_fn=build_djia_dataset,
        train_model_fn=train_djia_model,
        predict_fn=predict_djia_stocks,
        stats_file=DJIA_STATS_FILE,
        weights_file=DJIA_WEIGHTS_FILE,
        model_cls=DjiaStockModel,
        mode=mode,
        period=period,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        threshold=threshold
    )

    # 2. Run High-Tech Pipeline
    tech_results = run_pipeline(
        target_name="High-Tech Leaders",
        tickers=TECH_TICKERS,
        build_dataset_fn=build_tech_dataset,
        train_model_fn=train_tech_model,
        predict_fn=predict_tech_stocks,
        stats_file=TECH_STATS_FILE,
        weights_file=TECH_WEIGHTS_FILE,
        model_cls=TechStockModel,
        mode=mode,
        period=period,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        threshold=threshold
    )

    # 3. Print Combined Market Summary (if in predict mode)
    if mode in ["train-and-predict", "predict"]:
        print_combined_market_summary(djia_results, tech_results)

    return djia_results, tech_results


def main():
    parser = argparse.ArgumentParser(
        description="Trades: Unified runner to execute both DJIA and High-Tech stock prediction pipelines."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        help="Target market segment (default: 'all' to run both DJIA and Tech)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train-and-predict", "train", "predict"],
        default="train-and-predict",
        help="Execution mode for both models: train-and-predict (default), train, or predict"
    )
    parser.add_argument("--period", type=str, default="2y", help="Historical data retrieval period (e.g., 1y, 2y, 5y)")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=75.0, help="Buy signal probability threshold percentage")

    args = parser.parse_args()

    run_all_pipelines(
        mode=args.mode,
        period=args.period,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
