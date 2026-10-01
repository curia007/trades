"""Llama-style transformer wrapper for tabular stock features.

Treats each input feature as a token, runs Llama decoder layers, then
pools the sequence into a single next-day bullish logit.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


DEFAULT_LLAMA_CONFIG = {
    "framework": "mlx",
    "model_type": "llama",
    "class_name": "StockPredictionLlama",
    "wrapper": "transformer",
    "input_dim": 4,
    "vocab_size": 4,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "output_dim": 1,
    "output_activation": "sigmoid",
    "features": ["RSI", "MACD", "Volume_Trend", "Daily_Return"],
    "task": "binary_classification",
    "target": "next_day_positive_return",
}


@dataclass
class LlamaWrapperConfig:
    model_type: str = "llama"
    input_dim: int = 4
    vocab_size: int = 4
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    output_dim: int = 1

    @classmethod
    def from_dict(cls, data: dict | None = None) -> "LlamaWrapperConfig":
        if not data:
            return cls()
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        if "vocab_size" in data and "input_dim" not in data:
            fields["input_dim"] = data["vocab_size"]
        if "input_dim" in data and "vocab_size" not in data:
            fields["vocab_size"] = data["input_dim"]
        return cls(**fields)

    def to_architecture_dict(self) -> dict:
        arch = dict(DEFAULT_LLAMA_CONFIG)
        arch.update(asdict(self))
        return arch


ModelArgs = LlamaWrapperConfig


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)
        norm = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + self.eps)
        return (x32 * norm * self.weight).astype(orig_dtype)


class LlamaAttention(nn.Module):
    def __init__(self, cfg: LlamaWrapperConfig):
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.rope_theta = cfg.rope_theta
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def _rope(self, x: mx.array) -> mx.array:
        # x: (B, n_heads, L, head_dim)
        seq_len = x.shape[2]
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (mx.arange(0, half, dtype=mx.float32) / half)
        )
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).reshape(1, 1, seq_len, self.head_dim)
        sin = mx.sin(emb).reshape(1, 1, seq_len, self.head_dim)
        x32 = x.astype(mx.float32)
        return (x32 * cos + _rotate_half(x32) * sin).astype(x.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self._rope(q)
        k = self._rope(k)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaWrapperConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, cfg: LlamaWrapperConfig):
        super().__init__()
        self.self_attn = LlamaAttention(cfg)
        self.mlp = LlamaMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class StockPredictionLlama(nn.Module):
    """Transformer wrapper: feature tokens -> Llama layers -> binary logit."""

    def __init__(self, config: LlamaWrapperConfig | dict | None = None, input_dim: int | None = None, args: LlamaWrapperConfig | dict | None = None):
        super().__init__()
        cfg_input = args if args is not None else config
        if isinstance(cfg_input, dict):
            cfg = LlamaWrapperConfig.from_dict(cfg_input)
        elif cfg_input is None:
            cfg = LlamaWrapperConfig()
        else:
            cfg = cfg_input
        if input_dim is not None:
            cfg.input_dim = input_dim
        self.cfg = cfg
        self.args = cfg
        self.feature_proj = nn.Linear(1, cfg.hidden_size, bias=False)
        self.feature_embed = nn.Embedding(cfg.input_dim, cfg.hidden_size)
        self.layers = [LlamaDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.output = nn.Linear(cfg.hidden_size, cfg.output_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        bsz, n_features = x.shape
        tokens = self.feature_proj(x.reshape(bsz, n_features, 1))
        pos = mx.arange(n_features)
        tokens = tokens + self.feature_embed(pos)
        for layer in self.layers:
            tokens = layer(tokens)
        hidden = self.norm(tokens)
        pooled = mx.mean(hidden, axis=1)
        return self.output(pooled)


# Backward-compatible and custom model aliases
Model = StockPredictionLlama
StockPredictionModel = StockPredictionLlama
