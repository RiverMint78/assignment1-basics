import os
import typing

import torch


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint_state = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if isinstance(out, (str, os.PathLike)):
        parent_dir = os.path.dirname(os.path.abspath(out))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    torch.save(checkpoint_state, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    target_device = next(model.parameters()).device
    checkpoint_state = torch.load(src, map_location=target_device)
    model.load_state_dict(checkpoint_state["model_state_dict"])
    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    return checkpoint_state["iteration"]
