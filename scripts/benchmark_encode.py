from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from cs336_basics.tokenizer.bpe_tokenizer import DEFAULT_END_TOKEN, BPETokenizer


def iter_limited_text(path: str | os.PathLike, max_bytes: int):
    seen = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            b = len(line.encode("utf-8"))
            if seen + b > max_bytes:
                remain = max_bytes - seen
                if remain > 0:
                    yield line.encode("utf-8")[:remain].decode("utf-8", errors="ignore")
                break
            yield line
            seen += b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mb", type=int, default=100)
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_file(args.tokenizer, [DEFAULT_END_TOKEN])
    max_bytes = args.mb * 1024 * 1024

    start = time.time()
    n_tokens = 0
    for _ in tokenizer.encode_iterable(iter_limited_text(args.input, max_bytes)):
        n_tokens += 1
    elapsed = time.time() - start

    mbps = args.mb / elapsed
    tokps = n_tokens / elapsed

    file_size = Path(args.input).stat().st_size
    est_time = file_size / max_bytes * elapsed
    est_tokens = file_size / max_bytes * n_tokens

    print(f"sample bytes: {args.mb} MiB")
    print(f"sample tokens: {n_tokens:,}")
    print(f"time: {elapsed:.2f}s")
    print(f"throughput: {mbps:.2f} MiB/s, {tokps:,.0f} tokens/s")
    print(f"estimated full tokens: {est_tokens:,.0f}")
    print(f"estimated full time: {est_time/60:.1f} min = {est_time/3600:.2f} h")


if __name__ == "__main__":
    main()
# python scripts/benchmark_encode.py --tokenizer data/TinyStoriesV2.tokenizer.pkl  --input data/TinyStoriesV2-GPT4-train.txt --mb 10