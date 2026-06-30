from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer.bpe_tokenizer import DEFAULT_END_TOKEN, BPETokenizer


def iter_text_lines(path: str | os.PathLike) -> Iterable[str]:
    """Stream a UTF-8 text file line by line.

    Keeping newlines in each yielded line is important because they are part of
    the training corpus.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        yield from f


def encode_to_uint16_npy(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    token_buffer_size: int = 1_000_000,
    copy_chunk_tokens: int = 8_000_000,
    force: bool = False,
) -> int:
    """Encode a text dataset into a uint16 NumPy .npy file.

    This function avoids materializing the full token list in RAM. It first
    writes a temporary raw uint16 stream, counts tokens, then wraps it into a
    standard .npy file using open_memmap.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists. Use --force to overwrite.")

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.uint16.bin")
    if tmp_path.exists():
        tmp_path.unlink()

    total_tokens = 0
    buf: list[int] = []

    print(f"Encoding: {input_path}")
    print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    print(f"Temporary raw file: {tmp_path}")

    with open(tmp_path, "wb") as raw_f:
        for token_id in tokenizer.encode_iterable(iter_text_lines(input_path)):
            if token_id < 0 or token_id > np.iinfo(np.uint16).max:
                raise ValueError(f"Token id {token_id} cannot be represented as uint16. Use uint32 instead.")

            buf.append(token_id)

            if len(buf) >= token_buffer_size:
                arr = np.asarray(buf, dtype=np.uint16)
                arr.tofile(raw_f)
                total_tokens += arr.size
                buf.clear()

                if total_tokens % 10_000_000 < token_buffer_size:
                    print(f"  encoded {total_tokens:,} tokens...")

        if buf:
            arr = np.asarray(buf, dtype=np.uint16)
            arr.tofile(raw_f)
            total_tokens += arr.size
            buf.clear()

    print(f"Total tokens: {total_tokens:,}")
    print(f"Writing NumPy array: {output_path}")

    if output_path.exists():
        output_path.unlink()

    out_arr = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    pos = 0
    with open(tmp_path, "rb") as raw_f:
        while True:
            chunk = np.fromfile(raw_f, dtype=np.uint16, count=copy_chunk_tokens)
            if chunk.size == 0:
                break
            out_arr[pos : pos + chunk.size] = chunk
            pos += chunk.size

    out_arr.flush()
    del out_arr

    if pos != total_tokens:
        raise RuntimeError(f"Copied {pos} tokens, expected {total_tokens}.")

    tmp_path.unlink()

    print("Done.")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MiB")

    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer .pkl file.")
    parser.add_argument("--input", required=True, help="Path to input .txt dataset.")
    parser.add_argument("--output", required=True, help="Path to output .npy file.")
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token. Can be passed multiple times. Defaults to <|endoftext|>.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument(
        "--token-buffer-size",
        type=int,
        default=1_000_000,
        help="Number of token ids buffered before flushing to disk.",
    )
    args = parser.parse_args()

    special_tokens = args.special_token or [DEFAULT_END_TOKEN]

    tokenizer = BPETokenizer.from_file(args.tokenizer, special_tokens=special_tokens)

    encode_to_uint16_npy(
        tokenizer=tokenizer,
        input_path=args.input,
        output_path=args.output,
        token_buffer_size=args.token_buffer_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
# python ./scripts/encode_dataset.py --tokenizer data/owt.tokenizer.pkl --input data/owt_valid.txt  --output data/owt_valid.uint16.npy