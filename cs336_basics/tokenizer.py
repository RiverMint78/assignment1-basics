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


def _pretokenize_worker(chunk: str, end_token: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    if special_tokens:
        pattern = "|".join(re.escape(token) for token in special_tokens)
        text = re.sub(pattern, end_token, chunk)
    else:
        text = chunk
    counter = Counter()
    for mini_chunk in text.split(end_token):
        for match in _PRETOKEN_PAT.finditer(mini_chunk):
            counter[tuple(match.group().encode("utf-8"))] += 1
    return counter


def pretokenize(file_path: str, end_token: str, special_tokens: list[str], chunk_cnt: int | None = None) -> Counter[tuple[int, ...]]:
    if os.path.getsize(file_path) < 10_000_000:
        total = Counter()
        for chunk in chunk_iterator(file_path, end_token, 1):
            total += _pretokenize_worker(chunk, end_token, special_tokens)
        return total
    chunk_cnt = chunk_cnt or min(os.cpu_count() or 1, 8)
    worker = partial(_pretokenize_worker, end_token=end_token, special_tokens=special_tokens)
    with Pool(processes=chunk_cnt) as pool:
        total = Counter()
        for c in pool.imap_unordered(worker, chunk_iterator(file_path, end_token, chunk_cnt), chunksize=1):
            total.update(c)
        return total


def _merge_tuple_ids(token_tuple: tuple[int, ...], merge_pair: tuple[int, int], merged_id: int) -> tuple[int, ...]:
    out = []
    i = 0
    while i < len(token_tuple):
        if i + 1 < len(token_tuple) and token_tuple[i] == merge_pair[0] and token_tuple[i + 1] == merge_pair[1]:
            out.append(merged_id)
            i += 2
        else:
            out.append(token_tuple[i])
            i += 1
    return tuple(out)


def bpe_merge(word_cnter: Counter[tuple[int, ...]], vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    words = list(word_cnter.keys())
    word_cnts = list(word_cnter.values())

    vocab_lst: list[bytes] = [bytes([i]) for i in range(256)]
    merges_ids: list[tuple[int, int]] = []

    pair_cnts: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

    for word_id, word in enumerate(words):
        cnt = word_cnts[word_id]
        for pair in zip(word, word[1:]):
            pair_cnts[pair] += cnt
            pair_to_words[pair].add(word_id)

    while (len(vocab_lst) + len(special_tokens)) < vocab_size:
        if not pair_cnts:
            break

        merge_pair = max(pair_cnts, key=lambda p: (pair_cnts[p], vocab_lst[p[0]], vocab_lst[p[1]]))

        merged_id = len(vocab_lst)
        vocab_lst.append(vocab_lst[merge_pair[0]] + vocab_lst[merge_pair[1]])
        merges_ids.append(merge_pair)

        affected_word_ids = list(pair_to_words.get(merge_pair, set()))

        for word_id in affected_word_ids:
            old_word = words[word_id]

            if merge_pair not in zip(old_word, old_word[1:]):
                continue

            cnt = word_cnts[word_id]
            old_pairs = list(zip(old_word, old_word[1:]))

            # subtract old pairs
            for pair in old_pairs:
                pair_cnts[pair] -= cnt
                if pair_cnts[pair] <= 0:
                    pair_cnts.pop(pair, None)

            # remove id from rev idx
            for pair in set(old_pairs):
                word_set = pair_to_words.get(pair)
                if word_set is not None:
                    word_set.discard(word_id)
                    if not word_set:
                        pair_to_words.pop(pair, None)

            new_word = _merge_tuple_ids(old_word, merge_pair, merged_id)
            words[word_id] = new_word

            # add new pairs
            for pair in zip(new_word, new_word[1:]):
                pair_cnts[pair] += cnt
                pair_to_words[pair].add(word_id)

    # append special tokens
    for token in special_tokens:
        vocab_lst.append(token.encode("utf-8"))

    merges = [(vocab_lst[a], vocab_lst[b]) for a, b in merges_ids]
    return {i: token for i, token in enumerate(vocab_lst)}, merges


def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return bpe_merge(pretokenize(input_path, end_token=special_tokens[0], special_tokens=special_tokens), vocab_size, special_tokens)


def save_vocab_csv(vocab: dict[int, bytes], output_path: str):
    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "bytes"])
        for token_id in sorted(vocab.keys()):
            writer.writerow([token_id, vocab[token_id]])
    print(f"Vocab saved to {output_path} ({len(vocab)} tokens)")


def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_path: str):
    import pickle

    with open(output_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


if __name__ == "__main__":
    vocab, merges = bpe_train(r"data/TinyStoriesV2-GPT4-valid.txt", 32000, [_END_TOKEN])
    save_vocab_csv(vocab, "tmp.vocab.csv")
