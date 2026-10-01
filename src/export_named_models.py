"""Package the two trained MLX Llama models with named config files for reuse.

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
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stock_llama import DEFAULT_LLAMA_CONFIG, StockPredictionLlama


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




MODEL_SPECS = {
    "top_tech_trades_01": {
        "name": "top_tech_trades_01",
        "display_name": "Top Tech Trades v01",
        "universe": "tech",
        "description": "MLX Llama transformer wrapper trained to forecast next-open bullish probability for leading tech stocks.",
        "source_weights": DATA_DIR / "tech_stock_model.safetensors",
        "source_stats": DATA_DIR / "normalization_tech_stats.json",
        "tickers": TECH_TICKERS,
    },
    "top_djia_trades_01": {
        "name": "top_djia_trades_01",
        "display_name": "Top DJIA Trades v01",
        "universe": "djia",
        "description": "MLX Llama transformer wrapper trained to forecast next-open bullish probability for DJIA component stocks.",
        "source_weights": DATA_DIR / "djia_stock_model.safetensors",
        "source_stats": DATA_DIR / "normalization_djia_stats.json",
        "tickers": DJIA_TICKERS,
    },
}

ARCHITECTURE = dict(DEFAULT_LLAMA_CONFIG)


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
        "model_type": "llama",
        "model_file": "model.py",
        "architectures": ["StockPredictionLlama"],
        "hidden_size": ARCHITECTURE.get("hidden_size", 64),
        "num_hidden_layers": ARCHITECTURE.get("num_hidden_layers", 2),
        "intermediate_size": ARCHITECTURE.get("intermediate_size", 128),
        "num_attention_heads": ARCHITECTURE.get("num_attention_heads", 4),
        "rms_norm_eps": ARCHITECTURE.get("rms_norm_eps", 1e-5),
        "vocab_size": ARCHITECTURE.get("vocab_size", ARCHITECTURE.get("input_dim", 4)),
        "rope_theta": ARCHITECTURE.get("rope_theta", 10000.0),
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


def write_agent_files(model_dir: Path, model_name: str, host: str = "127.0.0.1", port: int = 11434) -> None:
    """Write files agents use to discover a local OpenAI-compatible LLM."""
    model_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{host}:{port}/v1"
    agent = {
        "name": model_name,
        "kind": "local_llm",
        "protocol": "openai",
        "base_url": base_url,
        "model": model_name,
        "api_key": "local",
        "endpoints": {
            "models": f"{base_url}/models",
            "chat_completions": f"{base_url}/chat/completions",
        },
        "start_command": f"python src/local_llm.py --model {model_name} --host {host} --port {port}",
    }
    (model_dir / "agent.json").write_text(json.dumps(agent, indent=2) + "\n")
    generation = {
        "model_type": "llama",
        "architectures": ["StockPredictionLlama"],
        "local_llm": True,
        "openai_compatible": True,
        "base_url": base_url,
    }
    (model_dir / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n")
    modelfile = (
        f"# Local agent LLM wrapper for {model_name}\n"
        f"FROM {model_dir}\n"
        f"PARAMETER temperature 0\n"
        f'SYSTEM "You are {model_name}, a local Llama stock forecast model. '
        f'Reply with next-open bullish probabilities for requested tickers."\n'
    )
    (model_dir / "Modelfile").write_text(modelfile)


def write_tokenizer_files(model_dir: Path) -> None:
    """Write lightweight tokenizer files for mlx_lm compatibility."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from transformers import PreTrainedTokenizerFast

        vocab = {
            "[UNK]": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<pad>": 4,
        }
        for i in range(256):
            vocab[chr(i)] = len(vocab)

        chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

        tok = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
        fast_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            unk_token="[UNK]",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            chat_template=chat_template,
        )
        fast_tok.save_pretrained(model_dir)
    except Exception:
        pass


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

    stock_llama_src = Path(__file__).resolve().parent / "stock_llama.py"
    if stock_llama_src.exists():
        shutil.copy2(stock_llama_src, out_dir / "model.py")

    config = create_model_config(model_name)
    config["local_llm"] = {
        "enabled": True,
        "protocol": "openai",
        "model_id": model_name,
        "start_command": f"python src/local_llm.py --model {model_name}",
        "base_url": "http://127.0.0.1:11434/v1",
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    write_agent_files(out_dir, model_name, "127.0.0.1", 11434)
    write_tokenizer_files(out_dir)

    print(f"✓ Packaged '{model_name}' -> {out_dir}")
    print(f"    config.json, model.safetensors, normalization_stats.json, agent.json, model.py")
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
    model = StockPredictionLlama(arch)
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
    print("Packaging MLX Llama transformer models for reuse")
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
