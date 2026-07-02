from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml
from jaxtyping import Int
from torch import Tensor

from cs336_basics.tokenizer.bpe_tokenizer import BPETokenizer
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


def load_tokenizer(path: str | os.PathLike[str], special_tokens: list[str] | None) -> BPETokenizer:
    return BPETokenizer.from_file(path, special_tokens)


def clean_state_dict_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        cleaned[k] = v
    return cleaned


def load_model_from_checkpoint(
    model: TransformerLM,
    checkpoint_path: str | os.PathLike[str],
    device: str | torch.device,
) -> int | None:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        iteration = ckpt.get("iteration", ckpt.get("step", None))
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        iteration = ckpt.get("iteration", ckpt.get("step", None))
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        iteration = None
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    model.load_state_dict(clean_state_dict_keys(state_dict))
    return iteration


@dataclass
class GenerationStats:
    prompt_tokens: int
    total_tokens: int
    new_tokens: int
    stopped_at_eos: bool
    elapsed_s: float
    tokens_per_s: float
    ms_per_token: float
    max_memory_allocated_mib: float | None = None
    max_memory_reserved_mib: float | None = None


@torch.inference_mode()
def sample_next_token(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> int:
    """
    logits: [1, vocab_size]
    """
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        values, _ = torch.topk(logits, k=k, dim=-1)
        cutoff = values[:, [-1]]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def generate_ids(
    model: TransformerLM,
    prompt_ids: list[int],
    *,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eos_id: int | None,
    stop_at_eos: bool,
    device: str | torch.device,
) -> tuple[list[int], GenerationStats]:
    model.eval()

    token_ids = list(prompt_ids)
    prompt_tokens = len(prompt_ids)
    stopped_at_eos = False

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    for _ in range(max_new_tokens):
        context_ids = token_ids[-context_length:]

        x: Int[Tensor, "1 seq"] = torch.as_tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)
        next_logits = logits[:, -1, :]
        next_id = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
        )

        token_ids.append(next_id)

        if stop_at_eos and eos_id is not None and next_id == eos_id:
            stopped_at_eos = True
            break

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()

    elapsed_s = time.perf_counter() - start_time
    new_tokens = len(token_ids) - prompt_tokens

    tokens_per_s = new_tokens / elapsed_s if elapsed_s > 0 else float("inf")
    ms_per_token = 1000 * elapsed_s / new_tokens if new_tokens > 0 else float("inf")

    max_alloc = None
    max_reserved = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        max_alloc = torch.cuda.max_memory_allocated() / 1024**2
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2

    stats = GenerationStats(
        prompt_tokens=prompt_tokens,
        total_tokens=len(token_ids),
        new_tokens=new_tokens,
        stopped_at_eos=stopped_at_eos,
        elapsed_s=elapsed_s,
        tokens_per_s=tokens_per_s,
        ms_per_token=ms_per_token,
        max_memory_allocated_mib=max_alloc,
        max_memory_reserved_mib=max_reserved,
    )

    return token_ids, stats


def get_eos_id(tokenizer: BPETokenizer, eos_token: str | None) -> int | None:
    if eos_token is None:
        return None

    eos_ids = list(tokenizer.encode(eos_token))
    if len(eos_ids) != 1:
        print(f"[warning] eos_token={eos_token!r} encoded to {eos_ids}, not a single token; EOS stopping disabled.")
        return None

    return eos_ids[0]


def print_stats(stats: GenerationStats) -> None:
    print()
    print("-" * 80)
    print(
        f"prompt_tokens={stats.prompt_tokens} | "
        f"new_tokens={stats.new_tokens} | "
        f"total_tokens={stats.total_tokens} | "
        f"stopped_at_eos={stats.stopped_at_eos}"
    )
    print(f"elapsed={stats.elapsed_s:.3f}s | speed={stats.tokens_per_s:.2f} tok/s | latency={stats.ms_per_token:.2f} ms/token")

    if stats.max_memory_allocated_mib is not None:
        print(f"cuda_peak_allocated={stats.max_memory_allocated_mib:.0f} MiB | cuda_peak_reserved={stats.max_memory_reserved_mib:.0f} MiB")

    print("-" * 80)


def build_model(
    model_cfg: dict[str, Any],
    *,
    device: str,
    dtype: torch.dtype,
) -> TransformerLM:
    return TransformerLM(
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


def run_once(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    *,
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eos_id: int | None,
    stop_at_eos: bool,
    device: str,
) -> None:
    prompt_ids = list(tokenizer.encode(prompt))

    if len(prompt_ids) == 0:
        print("[warning] prompt encoded to empty token sequence; skip.")
        return

    output_ids, stats = generate_ids(
        model,
        prompt_ids,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
        stop_at_eos=stop_at_eos,
        device=device,
    )

    text = tokenizer.decode(output_ids)

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)
    print_stats(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    run_cfg = cfg["run"]
    model_cfg = cfg["model"]
    ckpt_cfg = cfg["checkpoint"]
    tok_cfg = cfg["tokenizer"]
    gen_cfg = cfg["generation"]

    set_seed(int(run_cfg.get("seed", 42)))

    device = run_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_torch_dtype(run_cfg.get("dtype", "float32"))

    if run_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    special_tokens = tok_cfg.get("special_tokens", None)
    if special_tokens is None and tok_cfg.get("eos_token", None) is not None:
        special_tokens = [tok_cfg["eos_token"]]

    tokenizer = load_tokenizer(tok_cfg["path"], special_tokens=special_tokens)
    eos_id = get_eos_id(tokenizer, tok_cfg.get("eos_token", None))

    model = build_model(model_cfg, device=device, dtype=dtype)

    iteration = load_model_from_checkpoint(
        model,
        checkpoint_path=ckpt_cfg["path"],
        device=device,
    )

    model.eval()

    if iteration is not None:
        print(f"Loaded checkpoint from step {iteration}: {ckpt_cfg['path']}")
    else:
        print(f"Loaded checkpoint: {ckpt_cfg['path']}")

    context_length = int(model_cfg["context_length"])
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(gen_cfg["max_new_tokens"])
    temperature = args.temperature if args.temperature is not None else float(gen_cfg.get("temperature", 1.0))
    top_k = args.top_k if args.top_k is not None else gen_cfg.get("top_k", None)
    stop_at_eos = bool(gen_cfg.get("stop_at_eos", True))

    print(
        f"Generation config: context_length={context_length}, "
        f"max_new_tokens={max_new_tokens}, temperature={temperature}, "
        f"top_k={top_k}, eos_id={eos_id}, stop_at_eos={stop_at_eos}"
    )

    if args.interactive:
        print()
        print("Interactive mode. Type /q, /quit, or /exit to leave.")
        print("You can also use:")
        print("  /set temperature 0.8")
        print("  /set top_k 50")
        print("  /set max_new_tokens 200")
        print()

        while True:
            prompt = input("prompt> ")

            if prompt.strip().lower() in {"/q", "/quit", "/exit"}:
                break

            if prompt.strip().startswith("/set "):
                parts = prompt.strip().split()
                if len(parts) != 3:
                    print("Usage: /set temperature 0.8 | /set top_k 50 | /set max_new_tokens 200")
                    continue

                key, value = parts[1], parts[2]

                try:
                    if key == "temperature":
                        temperature = float(value)
                    elif key == "top_k":
                        top_k = None if value.lower() == "none" else int(value)
                    elif key == "max_new_tokens":
                        max_new_tokens = int(value)
                    else:
                        print(f"Unknown setting: {key}")
                        continue
                except ValueError as e:
                    print(f"Bad value: {e}")
                    continue

                print(f"Updated: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")
                continue

            run_once(
                model,
                tokenizer,
                prompt,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_id=eos_id,
                stop_at_eos=stop_at_eos,
                device=device,
            )

    else:
        prompt = args.prompt if args.prompt is not None else gen_cfg["prompt"]
        run_once(
            model,
            tokenizer,
            prompt,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
            stop_at_eos=stop_at_eos,
            device=device,
        )


if __name__ == "__main__":
    main()
