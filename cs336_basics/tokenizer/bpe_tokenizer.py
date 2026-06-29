from collections.abc import Iterable, Iterator

from .utils import DEFAULT_END_TOKEN, PRETOKEN_PAT, load_tokenizer, special_tokens_to_end_token


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        self.vocab: dict[int, bytes] = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens:
            max_id = max(vocab.keys()) if vocab else -1
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in vocab.values():
                    max_id += 1
                    vocab[max_id] = token_bytes
        self.rev_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.bpe_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_file(cls, tokenizer_filepath: str, special_tokens: list[str] | None) -> "BPETokenizer":
        vocab, merges = load_tokenizer(tokenizer_filepath)
        return BPETokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pretokenize
        words = []
        for text_chunk in text.split(DEFAULT_END_TOKEN):
            for match in PRETOKEN_PAT.finditer(text_chunk):
                words.append(tuple(bytes([b]) for b in match.group().encode("utf-8")))
        # merge
        tokens = []
        for word in words:
            while len(word) > 1:
                pairs = list(zip(word, word[1:]))
                best_pair = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
                if best_pair not in self.bpe_ranks:
                    break
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            tokens.extend([self.rev_vocab[token] for token in word])
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str: ...


if __name__ == "__main__":
    tokenizer = BPETokenizer.from_file("data/TinyStoriesV2.tokenizer.pkl", [DEFAULT_END_TOKEN])
    print(tokenizer.encode(f"Last one.\n{DEFAULT_END_TOKEN}This is a tokenizer test."))
