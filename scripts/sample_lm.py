import argparse
import os
import pickle
import random
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

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


def load_tokenizer(path: str | os.PathLike[str]):
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_state_dict_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """
    If a model was saved after torch.compile, keys may be prefixed with '_orig_mod.'.
    Strip that prefix if present.
    """
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

    # Be tolerant to slightly different checkpoint formats.
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        iteration = ckpt.get("iteration", None)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        iteration = ckpt.get("iteration", ckpt.get("step", None))
    elif isinstance(ckpt, dict):
        # Maybe the file itself is a raw state_dict.
        state_dict = ckpt
        iteration = None
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    state_dict = clean_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    return iteration


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    *,
    context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token: str | None = None,
    stop_at_eos: bool = True,
    device: str | torch.device = "cpu",
) -> str:
    model.eval()

    token_ids = list(tokenizer.encode(prompt))

    if len(token_ids) == 0:
        raise ValueError("Prompt encoded to an empty token sequence.")

    eos_id: int | None = None
    if eos_token is not None:
        eos_encoded = list(tokenizer.encode(eos_token))
        if len(eos_encoded) == 1:
            eos_id = eos_encoded[0]

    for _ in range(max_new_tokens):
        # Use only the last context_length tokens.
        context_ids = token_ids[-context_length:]

        x = torch.tensor([context_ids], dtype=torch.long, device=device)

        logits = model(x)  # [1, seq_len, vocab_size]
        next_logits = logits[:, -1, :]  # [1, vocab_size]

        if temperature <= 0:
            # Greedy decoding.
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            next_logits = next_logits / temperature

            if top_k is not None:
                k = min(top_k, next_logits.shape[-1])
                values, _ = torch.topk(next_logits, k=k, dim=-1)
                cutoff = values[:, [-1]]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))

            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        token_ids.append(next_id)

        if stop_at_eos and eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
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

    tokenizer = load_tokenizer(tok_cfg["path"])

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

    iteration = load_model_from_checkpoint(
        model,
        checkpoint_path=ckpt_cfg["path"],
        device=device,
    )

    model.to(device)
    model.eval()

    if iteration is not None:
        print(f"Loaded checkpoint from step {iteration}: {ckpt_cfg['path']}")
    else:
        print(f"Loaded checkpoint: {ckpt_cfg['path']}")

    prompt = args.prompt if args.prompt is not None else gen_cfg["prompt"]

    text = generate(
        model,
        tokenizer,
        prompt,
        context_length=int(model_cfg["context_length"]),
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        temperature=float(gen_cfg.get("temperature", 1.0)),
        top_k=gen_cfg.get("top_k", None),
        eos_token=tok_cfg.get("eos_token", None),
        stop_at_eos=bool(gen_cfg.get("stop_at_eos", True)),
        device=device,
    )

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


if __name__ == "__main__":
    main()
