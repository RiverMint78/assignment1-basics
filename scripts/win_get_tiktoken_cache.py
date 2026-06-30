
import hashlib
import pathlib
import shutil

cache = pathlib.Path(r"./tiktoken_cache")
fixture = pathlib.Path(r"./tests/fixtures")
cache.mkdir(parents=True, exist_ok=True)

urls = {
    "vocab_bpe": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
    "encoder_json": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
}

vocab_bpe_cache = cache / hashlib.sha1(urls["vocab_bpe"].encode()).hexdigest()
encoder_json_cache = cache / hashlib.sha1(urls["encoder_json"].encode()).hexdigest()

shutil.copyfile(fixture / "gpt2_vocab.json", encoder_json_cache)

merges_path = fixture / "gpt2_merges.txt"
raw = merges_path.read_text(encoding="utf-8")

raw = raw.replace("\r\n", "\n").replace("\r", "\n")

if not raw.startswith("#version:"):
    raw = "#version: 0.2\n" + raw

if not raw.endswith("\n"):
    raw += "\n"

vocab_bpe_cache.write_bytes(raw.encode("utf-8"))

print("cache dir:", cache)
print("vocab.bpe cache:", vocab_bpe_cache)
print("encoder.json cache:", encoder_json_cache)
print("vocab.bpe sha256:", hashlib.sha256(vocab_bpe_cache.read_bytes()).hexdigest())
print("encoder.json sha256:", hashlib.sha256(encoder_json_cache.read_bytes()).hexdigest())
print("files:", list(cache.iterdir()))