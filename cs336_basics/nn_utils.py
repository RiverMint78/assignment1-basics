import einx
import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    names = [f"a{i}" for i in range(x.ndim)]
    names[dim] = f"[{names[dim]}]"
    return einx.softmax(f"{' '.join(names)} -> {' '.join(names)}", x)
