"""Package the two trained MLX models with named config files for reuse.

Creates shareable model directories:
  - top_tech_trades_01
  - top_djia_trades_01

Each directory contains:
  - config.json
  - model.safetensors
  - normalization_stats.json

Others can load a packaged model with:
  from src.export_named_models import load_named_model
  model, config, stats = load_named_model("top_djia_trades_01")
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"

TECH_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "AMD", "QCOM", "INTC", "CSCO", "TXN", "INTU", "NOW", "AMAT", "MU",
    "LRCX", "ADI", "PANW", "SNPS", "CDNS", "KLAC", "CRWD", "PLTR", "MRVL", "ANET",
]
DJIA_TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]


class StockPredictionModel(nn.Module):
    """MLP matching the training architecture: 4 -> 16 -> 8 -> 1."""

    def __init__(self, input_dim=4, hidden1=16, hidden2=8, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()

    def __call__(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)


MODEL_SPECS = {
    "top_tech_trades_01": {
        "name": "top_tech_trades_01",
        "display_name": "Top Tech Trades v01",
        "universe": "tech",
        "description": "MLX MLP trained to forecast next-open bullish probability for leading tech stocks.",
        "source_weights": DATA_DIR / "tech_stock_model.safetensors",
        "source_stats": DATA_DIR / "normalization_tech_stats.json",
        "tickers": TECH_TICKERS,
    },
    "top_djia_trades_01": {
        "name": "top_djia_trades_01",
        "display_name": "Top DJIA Trades v01",
        "universe": "djia",
        "description": "MLX MLP trained to forecast next-open bullish probability for DJIA component stocks.",
        "source_weights": DATA_DIR / "djia_stock_model.safetensors",
        "source_stats": DATA_DIR / "normalization_djia_stats.json",
        "tickers": DJIA_TICKERS,
    },
}

ARCHITECTURE = {
    "framework": "mlx",
    "model_type": "stock_prediction_mlp",
    "class_name": "StockPredictionModel",
    "input_dim": 4,
    "hidden1": 16,
    "hidden2": 8,
    "output_dim": 1,
    "activation": "relu",
    "output_activation": "sigmoid",
    "features": ["RSI", "MACD", "Volume_Trend", "Daily_Return"],
    "task": "binary_classification",
    "target": "next_day_positive_return",
}


def _model_dir(model_name: str) -> Path:
    return MODELS_DIR / model_name


def create_model_config(model_name: str) -> dict:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model '{model_name}'. Known: {list(MODEL_SPECS)}")

    spec = MODEL_SPECS[model_name]
    stats_path = spec["source_stats"]
    stats = {}
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)

    config = {
        "model_name": spec["name"],
        "display_name": spec["display_name"],
        "universe": spec["universe"],
        "description": spec["description"],
        "architecture": ARCHITECTURE,
        "weights_file": "model.safetensors",
        "normalization_file": "normalization_stats.json",
        "tickers": spec["tickers"],
        "trading_threshold": 75.0,
        "normalization": {
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "features": stats.get("features", ARCHITECTURE["features"]),
        },
    }
    return config


def export_named_model(model_name: str) -> Path:
    spec = MODEL_SPECS[model_name]
    out_dir = _model_dir(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_src = spec["source_weights"]
    stats_src = spec["source_stats"]
    if not weights_src.exists():
        raise FileNotFoundError(f"Missing weights for {model_name}: {weights_src}")
    if not stats_src.exists():
        raise FileNotFoundError(f"Missing normalization stats for {model_name}: {stats_src}")

    shutil.copy2(weights_src, out_dir / "model.safetensors")
    shutil.copy2(stats_src, out_dir / "normalization_stats.json")

    config = create_model_config(model_name)
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Packaged '{model_name}' -> {out_dir}")
    print(f"    config.json, model.safetensors, normalization_stats.json")
    return out_dir


def export_all_named_models() -> list[Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    exported = []
    for name in MODEL_SPECS:
        exported.append(export_named_model(name))
    return exported


def load_named_model(model_name: str):
    """Load a packaged model so others can run inference.

    Returns (model, config, stats) where stats has numpy mean/std arrays.
    """
    out_dir = _model_dir(model_name)
    config_path = out_dir / "config.json"
    if not config_path.exists():
        export_named_model(model_name)

    with open(config_path, "r") as f:
        config = json.load(f)

    arch = config["architecture"]
    model = StockPredictionModel(
        input_dim=arch["input_dim"],
        hidden1=arch["hidden1"],
        hidden2=arch["hidden2"],
        output_dim=arch["output_dim"],
    )
    weights_path = out_dir / config["weights_file"]
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())

    stats_path = out_dir / config["normalization_file"]
    with open(stats_path, "r") as f:
        raw_stats = json.load(f)
    stats = {
        "mean": np.array(raw_stats["mean"], dtype=np.float32),
        "std": np.array(raw_stats["std"], dtype=np.float32),
        "features": raw_stats.get("features", arch["features"]),
    }
    return model, config, stats


def list_named_models() -> list[str]:
    return list(MODEL_SPECS.keys())


def main():
    print("=" * 60)
    print("Packaging MLX models for reuse")
    print("=" * 60)
    paths = export_all_named_models()
    print("-" * 60)
    for path in paths:
        print(f"  {path.name}: {path}")
    print("-" * 60)
    print("Load example:")
    print("  from src.export_named_models import load_named_model")
    print('  model, config, stats = load_named_model("top_tech_trades_01")')
    print('  model, config, stats = load_named_model("top_djia_trades_01")')
    print("=" * 60)


if __name__ == "__main__":
    main()
