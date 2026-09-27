# Trades

A machine learning-driven framework for retrieving historical Dow Jones Industrial Average (DJIA) and High-Tech stock data, training deep neural network models with Apple's MLX framework, and forecasting the best stocks to buy for the next opening trading day.

---

## 📌 Overview

**Trades** provides end-to-end pipelines to evaluate stock momentum and next-day price gains across major market segments:
- **DJIA 30 Component Stocks**
- **High-Tech Leaders** (semiconductors, cloud, AI, enterprise software, and megacap tech)

It combines historical and live technical indicator engineering via Yahoo Finance (`yfinance`) with deep learning training and inference accelerated on Apple Silicon using MLX.

---

## ✨ Features

- **Historical & Live Data Retrieval**:
  - Automatically fetches multi-year historical OHLCV data for DJIA or High-Tech stocks via `yfinance`.
  - Calculates key technical indicators:
    - **14-Period RSI** (Relative Strength Index) for momentum and overbought/oversold levels.
    - **MACD** (Moving Average Convergence Divergence) using 12- and 26-period EMAs.
    - **Volume Trend** comparing current volume to a 20-day rolling average.
    - **Daily Return (%)** capturing past 24-hour price momentum.
  - Standardizes metrics using global training distribution statistics (Z-score normalization).

- **MLX Deep Learning Model & Training**:
  - Built with Apple's `mlx.core`, `mlx.nn`, and `mlx.optimizers` for native Apple Silicon acceleration.
  - Multi-Layer Perceptron (MLP) architecture:
    $$\text{Input (4)} \to \text{Linear(16)} \to \text{ReLU} \to \text{Linear(8)} \to \text{ReLU} \to \text{Linear(1)} \to \text{Sigmoid}$$
  - Trains using Binary Cross-Entropy loss with Adam optimizer to predict next-day positive price movement.
  - Serializes distinct models to `data/`:
    - DJIA: `djia_stock_model.safetensors` and `normalization_djia_stats.json`.
    - Tech: `tech_stock_model.safetensors` and `normalization_tech_stats.json`.

- **Next Opening Trading Day Forecasting**:
  - Fetches the latest live market data for tracked stocks.
  - Computes next-day gain probability and assigns trading signals:
    - 🚀 **BUY**: High breakout confidence.
    - 🔻 **SELL**: Low gain probability / high downward momentum.
    - ⏸️ **HOLD / SKIP**: Neutral / sideways momentum.
  - Ranks all stocks and highlights the **Top 5 to BUY**, **Top 5 to SELL**, and **Top 5 to HOLD/SKIP** for the next opening bell.

---

## 📂 Project Structure

```text
Trades/
├── README.md                          # Project documentation
├── LICENSE                            # License information
├── main.py                            # Unified CLI entry point (DJIA, Tech, or All)
├── data/                              # Serialized model weights & normalization metadata
│   ├── normalization_djia_stats.json  # DJIA Z-score normalization statistics
│   ├── djia_stock_model.safetensors   # Trained DJIA MLX neural network weights
│   ├── normalization_tech_stats.json  # Tech Z-score normalization statistics
│   └── tech_stock_model.safetensors   # Trained Tech MLX neural network weights
└── src/
    ├── run_all.py                     # Dual-market runner (calls both DJIA and Tech pipelines)
    ├── train_djia_stock_model.py      # DJIA historical retrieval, dataset construction & training
    ├── train_tech_stock_model.py      # High-Tech historical retrieval, dataset construction & training
    └── samples/
        └── stock_prediction.py        # Live feature fetching and MLX inference sample
```

---

## 🛠 Prerequisites & Installation

### Requirements
- **macOS** with Apple Silicon (recommended for MLX acceleration)
- **Python 3.9+**

### Install Dependencies

```bash
pip install mlx numpy pandas yfinance
```

> **Note on LibreSSL / urllib3**: On macOS systems where Python was compiled with LibreSSL instead of OpenSSL 1.1.1+, `urllib3` v2 may emit a `NotOpenSSLWarning`. The codebase automatically suppresses this warning, or you can optionally install `pip install "urllib3<2.0.0"`.

---

## 🚀 Getting Started

### 1. Train Model and Predict Best Stock (End-to-End)

Run both DJIA and High-Tech pipelines together:
```bash
python src/run_all.py
# or via main.py:
python main.py --target all
```

Run the DJIA stock pipeline only:
```bash
python main.py --target djia
# or directly execute the DJIA script:
python src/train_djia_stock_model.py
```

Run the High-Tech stock pipeline only:
```bash
python main.py --target tech
# or directly execute the tech script:
python src/train_tech_stock_model.py
```

Optional CLI flags:
```bash
python src/run_all.py --mode train-and-predict --period 2y --epochs 60 --lr 0.001 --threshold 75.0
```

### 2. Run Training Only

```bash
# Both DJIA & Tech model training
python src/run_all.py --mode train --period 2y --epochs 60

# DJIA model training
python main.py --target djia --mode train --period 2y --epochs 60
python src/train_djia_stock_model.py --period 2y --epochs 60

# Tech model training
python main.py --target tech --mode train --period 2y --epochs 60
python src/train_tech_stock_model.py --period 2y --epochs 60
```

### 3. Run Inference on Existing Trained Weights

```bash
# Run predictions for both DJIA and High-Tech
python src/run_all.py --mode predict

# DJIA prediction only
python main.py --target djia --mode predict

# Tech prediction only
python main.py --target tech --mode predict

# Standalone sample prediction for DJIA
python src/samples/stock_prediction.py
```

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is not financial advice. Always perform your own due diligence before making investment decisions.
