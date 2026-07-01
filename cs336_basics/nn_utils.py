import einx
import torch
from torch import Tensor


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    names = [f"a{i}" for i in range(x.ndim)]
    names[dim] = f"[{names[dim]}]"
    return einx.softmax(f"{' '.join(names)} -> {' '.join(names)}", x)
