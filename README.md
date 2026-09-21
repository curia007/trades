# Trades

A machine learning-driven framework for predicting stock breakouts and generating algorithmic trading signals on the NYSE and broader markets using Apple's MLX framework.

---

## 📌 Overview

**Trades** provides tools and pipelines to evaluate stock momentum and next-day breakout potential. It combines live technical indicator engineering via Yahoo Finance with deep learning inference accelerated on Apple Silicon using MLX.

---

## ✨ Features

- **Live Data & Feature Engineering**:
  - Automatically fetches historical market data via `yfinance`.
  - Calculates key technical indicators:
    - **14-Period RSI** (Relative Strength Index) for momentum and overbought/oversold levels.
    - **MACD** (Moving Average Convergence Divergence) using 12- and 26-period EMAs.
    - **Volume Trend** comparing current volume to a 20-day moving average.
    - **Daily Return (%)** capturing past 24-hour price momentum.
  - Standardizes metrics using Z-score normalization for neural network input.

- **MLX Deep Learning Architecture**:
  - Built with Apple's `mlx.core` and `mlx.nn` for native hardware acceleration.
  - Multi-Layer Perceptron (MLP) architecture:
    $$\text{Input (4)} \to \text{Linear(16)} \to \text{ReLU} \to \text{Linear(8)} \to \text{ReLU} \to \text{Linear(1)} \to \text{Sigmoid}$$
  - Seamless weights loading from `.safetensors` model format.

- **Automated Trading Signals**:
  - Computes a confidence percentage score for next-day breakout moves.
  - Evaluates scores against configurable confidence thresholds (e.g. 75%) to issue actionable signals:
    - 🚀 **BUY**: Strong breakout signature detected.
    - ⚠️ **HOLD / SKIP**: Below target momentum threshold.

---

## 📂 Project Structure

```text
Trades/
├── README.md                          # Project documentation
├── LICENSE                            # License information
├── main.py                            # Project entry point
└── src/
    └── samples/
        └── stock_prediction.py        # Live feature fetching and MLX inference pipeline
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

---

## 🚀 Getting Started

### Running the Stock Prediction Pipeline

1. Ensure your trained model weights (`stock_model.safetensors`) are present in the working directory.
2. Run the sample prediction script:

```bash
python src/samples/stock_prediction.py
```

3. To analyze different tickers or adjust the breakout threshold, edit the ticker symbol or threshold parameters in `src/samples/stock_prediction.py`:

```python
TICKER = "AAPL"          # Target ticker (e.g., NVDA, SPY, TSLA)
TRADING_THRESHOLD = 75.0 # Minimum confidence percentage for BUY signals
```

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is not financial advice. Always perform your own due diligence before making investment decisions.
