import math

import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.layers import Linear
from cs336_basics.nn_utils import softmax


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, d_k / 2]

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        x_even, x_odd = einx.id("... (d (1 + 1)) -> ... d, ... d", x_fp32)
        cos = self.cos_cached[token_positions]  # [..., in_seq_len, d_k / 2]
        sin = self.sin_cached[token_positions]  # [..., in_seq_len, d_k / 2]
        while cos.ndim < x_even.ndim:
            cos.unsqueeze_(-3)
            sin.unsqueeze_(-3)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out_pair = einx.id("..., ... -> ... (1 + 1)", out_even, out_odd)
        out = einx.id("... d two -> ... (d two)", out_pair)
        return out.to(in_dtype)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    d_k = Q.shape[-1]
    scores = einx.dot("... queries [d_k], ... keys [d_k] -> ... queries keys", Q, K) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return einx.dot("... queries [keys], ... [keys] d_v -> ... queries d_v", attn, V)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.max_seq_len = max_seq_len
        self.rope = None
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = x.shape[-2]
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        head_split_desc = r"... l (h d) -> ... h l d"
        Q = einx.id(head_split_desc, Q, h=self.num_heads)
        K = einx.id(head_split_desc, K, h=self.num_heads)
        V = einx.id(head_split_desc, V, h=self.num_heads)

        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        mha = scaled_dot_product_attention(Q, K, V, causal_mask)
        mha = einx.id("... h l d -> ... l (h d)", mha, h=self.num_heads)
        return self.output_proj(mha)
