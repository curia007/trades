"""Expose packaged named models as a local OpenAI-compatible LLM for agents.

Default model: top_tech_trades_01

Start:
  python src/local_llm.py --model top_tech_trades_01 --port 11434

Agents should use:
  base_url = http://127.0.0.1:11434/v1
  model    = top_tech_trades_01
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlx.core as mx
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_named_models import load_named_model, write_agent_files, MODELS_DIR
from stock_llama import StockPredictionLlama

DEFAULT_MODEL = "top_tech_trades_01"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")


def _signal(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "BUY"
    if probability <= (100.0 - threshold):
        return "SELL"
    if probability >= 55.0:
        return "BUY_MODERATE"
    if probability <= 45.0:
        return "SELL_MODERATE"
    return "HOLD"


def fetch_live_features(ticker: str, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    df = yf.Ticker(ticker).history(period="60d")
    if df.empty:
        raise ValueError(f"Could not retrieve data for ticker: {ticker}")

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Volume_Trend"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
    df["Daily_Return"] = df["Close"].pct_change() * 100
    latest = df[["RSI", "MACD", "Volume_Trend", "Daily_Return"]].dropna().iloc[-1].values
    return ((latest - mean) / std).astype(np.float32)


class LocalStockLLM:
    """Llama wrapper served as a local chat model for agents."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model, self.config, self.stats = load_named_model(model_name)
        if not isinstance(self.model, StockPredictionLlama):
            raise TypeError("Loaded weights are not a StockPredictionLlama")
        self.tickers = set(self.config.get("tickers", []))
        self.threshold = float(self.config.get("trading_threshold", 75.0))

    def list_models(self) -> dict[str, Any]:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "local",
                    "permission": [],
                    "root": self.model_name,
                    "parent": None,
                }
            ],
        }

    def predict_ticker(self, ticker: str) -> dict[str, Any]:
        ticker = ticker.upper()
        features = fetch_live_features(
            ticker,
            self.stats["mean"],
            self.stats["std"],
        )
        logits = self.model(mx.array(features.reshape(1, -1)))
        probability = float(mx.sigmoid(logits).item() * 100)
        return {
            "ticker": ticker,
            "probability": round(probability, 2),
            "signal": _signal(probability, self.threshold),
            "in_universe": ticker in self.tickers,
            "universe": self.config.get("universe"),
            "model": self.model_name,
        }

    def extract_tickers(self, text: str) -> list[str]:
        found = []
        for match in TICKER_RE.findall(text.upper()):
            if match in self.tickers and match not in found:
                found.append(match)
        return found

    def chat(self, messages: list[dict], max_tickers: int = 5) -> str:
        user_text = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") in {"user", "system"}
        )
        tickers = self.extract_tickers(user_text)
        if not tickers:
            listed = ", ".join(sorted(self.tickers))
            return (
                f"I am {self.config.get('display_name', self.model_name)}, a local Llama "
                f"stock-forecast model ({self.config.get('universe')} universe). "
                f"Ask about one or more tickers from: {listed}"
            )

        results = []
        errors = []
        for ticker in tickers[:max_tickers]:
            try:
                results.append(self.predict_ticker(ticker))
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")

        lines = [
            f"Local LLM: {self.model_name}",
            f"Task: next-open bullish probability (threshold {self.threshold:.0f}%)",
            "",
        ]
        for item in results:
            lines.append(
                f"- {item['ticker']}: {item['probability']:.2f}% -> {item['signal']}"
            )
        if errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(f"- {err}" for err in errors)
        lines.append("")
        lines.append("Not financial advice. Educational / research use only.")
        return "\n".join(lines)

    def chat_completion(self, body: dict) -> dict[str, Any]:
        messages = body.get("messages") or []
        content = self.chat(messages)
        created = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": created,
            "model": body.get("model") or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def make_handler(llm: LocalStockLLM):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict | list, extra_headers: dict | None = None) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in {"/v1/models", "/models"}:
                self._send(200, llm.list_models())
                return
            if path in {"/", "/health"}:
                self._send(200, {"status": "ok", "model": llm.model_name, "local_llm": True})
                return
            self._send(404, {"error": {"message": f"Unknown path {path}", "type": "not_found"}})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/")
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send(400, {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}})
                return
            if path in {"/v1/chat/completions", "/chat/completions"}:
                requested = body.get("model") or llm.model_name
                if requested != llm.model_name:
                    self._send(
                        404,
                        {"error": {"message": f"Model '{requested}' not found", "type": "invalid_request_error"}},
                    )
                    return
                self._send(200, llm.chat_completion(body))
                return
            self._send(404, {"error": {"message": f"Unknown path {path}", "type": "not_found"}})

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def serve(model_name: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    llm = LocalStockLLM(model_name)
    write_agent_files(MODELS_DIR / model_name, model_name, host, port)
    handler = make_handler(llm)
    httpd = ThreadingHTTPServer((host, port), handler)
    print("=" * 60)
    print(f"Local LLM ready: {model_name}")
    print(f"  GET  http://{host}:{port}/v1/models")
    print(f"  POST http://{host}:{port}/v1/chat/completions")
    print("Agents: set OpenAI base_url to the /v1 URL above, model id =", model_name)
    print("=" * 60)
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a named Trades Llama as a local LLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Named model directory (default: top_tech_trades_01)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    serve(args.model, args.host, args.port)


if __name__ == "__main__":
    main()
