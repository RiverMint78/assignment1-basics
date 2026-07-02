import math

import einx
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from cs336_basics.nn_utils import silu


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        init.trunc_normal_(self.weight, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

#     def reset_parameters(self) -> None:
#         init.ones_(self.weight)

#     def forward(self, x: Tensor) -> Tensor:
#         in_dtype = x.dtype
#         x_fp32 = x.to(torch.float32)
#         rms = torch.rsqrt(einx.mean("... d -> ... 1", x_fp32.square()) + self.eps)
#         out = einx.multiply("... d, ... 1, d -> ... d", x_fp32, rms, self.weight)
#         return out.to(in_dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + self.eps)
        out = x_fp32 * rms * self.weight
        return out.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return silu(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        if d_ff is None:
            d_ff = int((8 * d_model) / 3)
            d_ff = 64 * ((d_ff + 63) // 64)

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
