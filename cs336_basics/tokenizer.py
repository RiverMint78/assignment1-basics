import os
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

_PRETOKEN_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
_END_TOKEN = "<|endoftext|>"


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def chunk_iterator(file_path: str, end_token: str, chunk_cnt: int):
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_cnt, end_token.encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk.replace("\r\n", "\n")  # Windows mismatch


def _pretokenize_worker(chunk: str, end_token: str) -> Counter:
    res_cnter = Counter()
    for mini_chunk in chunk.split(end_token):
        res_cnter += Counter(tuple(bytes([b]) for b in match.group().encode("utf-8")) for match in _PRETOKEN_PAT.finditer(mini_chunk))
    return res_cnter


def pretokenize(file_path: str, end_token: str, chunk_cnt: int | None = None) -> Counter:
    if os.path.getsize(file_path) < 10_000_000:
        total = Counter()
        for chunk in chunk_iterator(file_path, end_token, 1):
            total += _pretokenize_worker(chunk, end_token)
        return total
    chunk_cnt = chunk_cnt or min(os.cpu_count() or 1, 8)
    worker = partial(_pretokenize_worker, end_token=end_token)
    with Pool(processes=chunk_cnt) as pool:
        return sum(pool.map(worker, chunk_iterator(file_path, end_token, chunk_cnt)), start=Counter())


def _merge_tuple(token_tuple: tuple[bytes, ...], merge_pair: tuple[bytes, bytes], merged: bytes) -> tuple[bytes, ...]:
    out = []
    i = 0
    while i < len(token_tuple):
        if i + 1 < len(token_tuple) and token_tuple[i] == merge_pair[0] and token_tuple[i + 1] == merge_pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(token_tuple[i])
            i += 1
    return tuple(out)


def bpe_merge(word_cnter: Counter, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    bpe_merges: list[tuple[bytes, bytes]] = []
    words = list(word_cnter.keys())
    word_cnts = list(word_cnter.values())

    # preprocess pairs
    pair_cnts = Counter()
    pair_to_words = defaultdict(set)

    for word_id, tokens in enumerate(words):
        cnt = word_cnts[word_id]
        for pair in zip(tokens, tokens[1:]):
            pair_cnts[pair] += cnt
            pair_to_words[pair].add(word_id)

    vocab_lst = [bytes([i]) for i in range(256)]

    while (len(vocab_lst) + len(special_tokens)) < vocab_size:
        merge_pair: tuple[bytes, bytes] = max(pair_cnts.items(), key=lambda x: (x[1], x[0]))[0]
        merged: bytes = merge_pair[0] + merge_pair[1]
        bpe_merges.append(merge_pair)
        vocab_lst.append(merged)

        affected_word_ids = pair_to_words[merge_pair].copy()

        for word_id in affected_word_ids:
            old_word = words[word_id]
            cnt = word_cnts[word_id]

            if merge_pair not in zip(old_word, old_word[1:]):
                continue

            # subtract old pairs
            old_pairs = zip(old_word, old_word[1:])
            for pair in old_pairs:
                pair_cnts[pair] -= cnt
                if pair_cnts[pair] <= 0:
                    pair_cnts.pop(pair, None)

            # remove id from rev idx
            for pair in set(old_pairs):
                word_set = pair_to_words.get(pair)
                if word_set is not None:
                    word_set.discard(word_id)

            # merge
            new_tuple = _merge_tuple(old_word, merge_pair, merged)
            words[word_id] = new_tuple

            # add new pairs
            for pair in zip(new_tuple, new_tuple[1:]):
                pair_cnts[pair] += cnt
                pair_to_words[pair].add(word_id)

    for special_token in special_tokens:
        vocab_lst.append(special_token.encode("utf-8"))

    return {i: token for i, token in enumerate(vocab_lst)}, bpe_merges


def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return bpe_merge(pretokenize(input_path, end_token=special_tokens[0]), vocab_size, special_tokens)


if __name__ == "__main__":
    word_cnter = pretokenize(r"data/TinyStoriesV2-GPT4-valid.txt", _END_TOKEN)
    bpe_merge(word_cnter, 1024, [_END_TOKEN])
