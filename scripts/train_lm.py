from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.transformer import TransformerLM


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_dtype(name: str) -> torch.dtype:
    match name:
        case "float32" | "fp32":
            return torch.float32
        case "float16" | "fp16":
            return torch.float16
        case "bfloat16" | "bf16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Unknown dtype: {name}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    valid_data: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()

    losses: list[float] = []

    for _ in range(eval_iters):
        x, y = get_batch(
            valid_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        losses.append(float(loss.item()))

    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimizer"]
    train_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]

    set_seed(int(run_cfg.get("seed", 42)))

    device = run_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_torch_dtype(run_cfg.get("dtype", "float32"))

    if run_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_data = np.load(data_cfg["train_path"], mmap_mode="r")
    valid_data = np.load(data_cfg["valid_path"], mmap_mode="r")

    model = TransformerLM(
        vocab_size=int(model_cfg["vocab_size"]),
        context_length=int(model_cfg["context_length"]),
        d_model=int(model_cfg["d_model"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        rope_theta=float(model_cfg["rope_theta"]),
        device=torch.device(device),
        dtype=dtype,
    )

    if run_cfg.get("compile", False):
        model = torch.compile(model)

    optimizer = AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        betas=(float(opt_cfg["beta1"]), float(opt_cfg["beta2"])),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )

    start_step = 0
    resume_from = ckpt_cfg.get("resume_from")
    if resume_from:
        start_step = load_checkpoint(
            resume_from,
            model,
            optimizer,
        )
        print(f"Resumed from {resume_from} at step {start_step}")

    output_dir = Path(ckpt_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=run_cfg["log_dir"])

    batch_size = int(train_cfg["batch_size"])
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    context_length = int(model_cfg["context_length"])

    max_steps = int(train_cfg["max_steps"])
    log_interval = int(train_cfg["log_interval"])
    eval_interval = int(train_cfg["eval_interval"])
    eval_iters = int(train_cfg["eval_iters"])
    checkpoint_interval = int(train_cfg["checkpoint_interval"])

    max_lr = float(opt_cfg["lr"])
    min_lr = float(opt_cfg["min_lr"])
    warmup_iters = int(opt_cfg["warmup_iters"])
    cosine_cycle_iters = int(opt_cfg["cosine_cycle_iters"])
    max_grad_norm = float(opt_cfg["max_grad_norm"])

    model.train()

    for step in range(start_step, max_steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        set_optimizer_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        for _ in range(grad_accum_steps):
            x, y = get_batch(
                train_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )

            logits = model(x)
            loss = cross_entropy(logits, y)

            # Keep effective gradient scale independent of grad_accum_steps.
            (loss / grad_accum_steps).backward()

            total_loss += float(loss.item())

        train_loss = total_loss / grad_accum_steps

        grad_norm = gradient_clipping(
            model.parameters(),
            max_l2_norm=max_grad_norm,
        )

        optimizer.step()

        if step % log_interval == 0:
            train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")

            print(f"step={step:>6d} lr={lr:.4e} train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} grad_norm={float(grad_norm):.4f}")

            writer.add_scalar("train/loss", train_loss, step)
            writer.add_scalar("train/perplexity", train_ppl, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)

        if step % eval_interval == 0 and step > 0:
            val_loss = estimate_loss(
                model,
                valid_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                eval_iters=eval_iters,
            )
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

            print(f"[eval] step={step:>6d} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

            writer.add_scalar("valid/loss", val_loss, step)
            writer.add_scalar("valid/perplexity", val_ppl, step)

        if step % checkpoint_interval == 0 and step > 0:
            ckpt_path = output_dir / f"ckpt_step_{step}.pt"
            latest_path = output_dir / "latest.pt"

            save_checkpoint(model, optimizer, step, ckpt_path)
            save_checkpoint(model, optimizer, step, latest_path)

            print(f"Saved checkpoint to {ckpt_path}")

    final_path = output_dir / "final.pt"
    save_checkpoint(model, optimizer, max_steps, final_path)
    print(f"Saved final checkpoint to {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
