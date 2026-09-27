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
import numpy as np

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


def main():
    parser = argparse.ArgumentParser(
        description="Trades: Retrieve stock history, train MLX neural network, and forecast next opening day gainers."
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["djia", "tech", "all"],
        default="djia",
        help="Stock market segment to model: 'djia' (Dow Jones 30, default), 'tech' (High-Tech 30), or 'all' (both)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train-and-predict", "train", "predict"],
        default="train-and-predict",
        help="Execution mode: train-and-predict (default), train only, or predict only"
    )
    parser.add_argument("--period", type=str, default="2y", help="Historical data retrieval period (e.g., 1y, 2y, 5y)")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=75.0, help="Buy signal probability threshold percentage")

    args = parser.parse_args()

    if args.target == "all":
        from src.run_all import run_all_pipelines
        run_all_pipelines(
            mode=args.mode,
            period=args.period,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            threshold=args.threshold
        )
        return

    if args.target == "tech":
        tickers = TECH_TICKERS
        build_dataset_fn = build_tech_dataset
        train_model_fn = train_tech_model
        predict_fn = predict_tech_stocks
        stats_file = TECH_STATS_FILE
        weights_file = TECH_WEIGHTS_FILE
        model = TechStockModel(input_dim=4)
    else:
        tickers = DJIA_TICKERS
        build_dataset_fn = build_djia_dataset
        train_model_fn = train_djia_model
        predict_fn = predict_djia_stocks
        stats_file = DJIA_STATS_FILE
        weights_file = DJIA_WEIGHTS_FILE
        model = DjiaStockModel(input_dim=4)

    norm_mean = None
    norm_std = None

    if args.mode in ["train-and-predict", "train"]:
        # 1. Retrieve Historical Data & Build Dataset
        X_train, y_train, X_val, y_val, norm_mean, norm_std = build_dataset_fn(
            tickers=tickers, period=args.period
        )

        # 2. Train Model with MLX
        train_model_fn(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

    if args.mode in ["train-and-predict", "predict"]:
        # Load weights and stats if not already trained in memory
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

        # 3. Predict & Rank Stocks for Next Opening Bell
        predict_fn(
            model=model,
            norm_mean=norm_mean,
            norm_std=norm_std,
            tickers=tickers,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
