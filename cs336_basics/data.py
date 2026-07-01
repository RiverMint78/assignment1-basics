import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    max_idx = len(dataset) - context_length - 1
    if max_idx < 0:
        raise ValueError(f"Dataset length ({len(dataset)}) is too short for context_length ({context_length})")
    start_indices = np.random.randint(low=0, high=max_idx + 1, size=(batch_size,))
    x_lst = [dataset[i : i + context_length] for i in start_indices]
    y_lst = [dataset[i + 1 : i + context_length + 1] for i in start_indices]
    x_np = np.stack(x_lst).astype(np.int64)
    y_np = np.stack(y_lst).astype(np.int64)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return x, y
