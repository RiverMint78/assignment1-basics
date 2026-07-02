import einx
import torch
from jaxtyping import Float, Int
from torch import Tensor


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


# def softmax(x: Tensor, dim: int = -1) -> Tensor:
#     names = [f"a{i}" for i in range(x.ndim)]
#     names[dim] = f"[{names[dim]}]"
#     return einx.softmax(f"{' '.join(names)} -> {' '.join(names)}", x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    return torch.softmax(x, dim=dim)


def cross_entropy(inputs: Float[Tensor, "... classes"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    max_logits = einx.max("... [classes]", inputs)
    shifted = einx.subtract("... classes, ... -> ... classes", inputs, max_logits)
    logsumexp = einx.sum("... [classes]", shifted.exp()).log()
    target_logits = einx.get_at("... [classes], ... -> ...", shifted, targets)
    losses = logsumexp - target_logits
    return losses.mean()


def perplexity(losses: Float[Tensor, "... seq_len"]) -> Float[Tensor, "..."]:
    return einx.mean("... [seq_len]", losses).exp()
