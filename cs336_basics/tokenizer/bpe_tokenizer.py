from collections.abc import Iterable, Iterator

import regex as re

from .utils import DEFAULT_END_TOKEN, PRETOKEN_PAT, load_tokenizer


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        self.vocab: dict[int, bytes] = vocab
        self.rev_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.bpe_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []
        self.special_token_to_id = {}
        if special_tokens:
            max_id = max(vocab.keys())
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in vocab.values():
                    max_id += 1
                    vocab[max_id] = token_bytes
                    self.special_token_to_id[token] = max_id
                else:
                    self.special_token_to_id[token] = self.rev_vocab[token_bytes]
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
        else:
            self.special_pattern = None

    @classmethod
    def from_file(cls, tokenizer_filepath: str, special_tokens: list[str] | None) -> "BPETokenizer":
        vocab, merges = load_tokenizer(tokenizer_filepath)
        return BPETokenizer(vocab, merges, special_tokens)

    def _bpe_encode_bytes(self, token_bytes: tuple[bytes, ...]) -> list[int]:
        while len(token_bytes) > 1:
            pairs = list(zip(token_bytes, token_bytes[1:]))
            best_pair = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if best_pair not in self.bpe_ranks:
                break
            new_bytes = []
            i = 0
            while i < len(token_bytes):
                if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i + 1]) == best_pair:
                    new_bytes.append(token_bytes[i] + token_bytes[i + 1])
                    i += 2
                else:
                    new_bytes.append(token_bytes[i])
                    i += 1
            token_bytes = new_bytes
        return [self.rev_vocab[token] for token in token_bytes]

    def _encode_normal_text(self, text: str) -> list[int]:
        tokens_ids = []
        for match in PRETOKEN_PAT.finditer(text):
            token_bytes = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            tokens_ids.extend(self._bpe_encode_bytes(token_bytes))
        return tokens_ids

    def encode(self, text: str) -> list[int]:
        tokens_ids = []
        if self.special_pattern is None:
            return self._encode_normal_text(text)
        text_chunks = self.special_pattern.split(text)
        for chunk in text_chunks:
            if chunk == "":
                continue
            if chunk in self.special_token_to_id:
                tokens_ids.append(self.special_token_to_id[chunk])
            else:
                tokens_ids.extend(self._encode_normal_text(chunk))
        return tokens_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab.get(token_id, b"") for token_id in ids)
        return raw_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = BPETokenizer.from_file("data/TinyStoriesV2.tokenizer.pkl", [DEFAULT_END_TOKEN])
    encoded = tokenizer.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>")
    print(encoded)
    print(tokenizer.decode(encoded))
