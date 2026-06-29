import os
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool

from .utils import DEFAULT_END_TOKEN, PRETOKEN_PAT, find_chunk_boundaries, save_tokenizer, save_vocab_csv, special_tokens_to_end_token


def chunk_iterator(file_path: str, end_token: str, chunk_cnt: int):
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_cnt, end_token.encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk.replace("\r\n", "\n")  # Windows mismatch


def _pretokenize_worker(chunk: str, end_token: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    text = special_tokens_to_end_token(chunk, end_token, special_tokens)
    counter = Counter()
    for text_chunk in text.split(end_token):
        for match in PRETOKEN_PAT.finditer(text_chunk):
            counter[tuple(match.group().encode("utf-8"))] += 1
    return counter


def get_pretoken_cnter(file_path: str, end_token: str, special_tokens: list[str], chunk_cnt: int | None = None) -> Counter[tuple[int, ...]]:
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


def bpe_merge_cnter(
    word_cnter: Counter[tuple[int, ...]], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    return bpe_merge_cnter(get_pretoken_cnter(input_path, end_token=special_tokens[0], special_tokens=special_tokens), vocab_size, special_tokens)


if __name__ == "__main__":
    vocab, merges = bpe_train(r"data/TinyStoriesV2-GPT4-valid.txt", 32000, [DEFAULT_END_TOKEN])
    save_vocab_csv(vocab, "tmp.vocab.csv")
