import regex as re
from collections import Counter

PRETOKEN_PAT = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

def read_file_bytes(input_path: str) -> bytes:
    with open(input_path, "rb") as f:
        data = f.read()
    return data

def split_on_specials(data: bytes, specials: list[str]) -> list[bytes]:
    toks = sorted({s.encode("utf-8") for s in specials}, key=len, reverse=True)
    if not toks:
        return [data]
    pat = re.compile(b"|".join(re.escape(t) for t in toks))
    return pat.split(data)

def pretokenize_span(span: bytes) -> list[bytes]:
    text = span.decode("utf-8", errors="ignore")
    return [match.encode("utf-8") for match in PRETOKEN_PAT.findall(text)]

def pretokenize_corpus(path: str, specials: list[str]) -> list[bytes]:
    data = read_file_bytes(path)
    spans = split_on_specials(data, specials)
    return [tok for span in spans for tok in pretokenize_span(span)]

def count_pretokens(data: bytes, specials: list[str]) -> Counter:
    counts = Counter()
    for span in split_on_specials(data, specials):
        counts.update(pretokenize_span(span))
    return counts


def count_corpus(path: str, specials: list[str]) -> Counter:
    return count_pretokens(read_file_bytes(path), specials)

def build_initial_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], int]:
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    next_id = len(vocab)
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    return vocab, next_id

def count_pairs(word_counts: Counter) -> Counter:
    pairs: Counter = Counter()
    for word, freq in word_counts.items():
        ids = tuple(word)
        for left, right in zip(ids, ids[1:]):
            pairs[(left, right)] += freq
    return pairs

def apply_merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i + 1 < n and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

def select_best_pair(pair_counts: Counter, vocab: dict[int, bytes]) -> tuple[int, int]:
    return max(
        pair_counts,
        key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]),
    )

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    raw_counts = count_pretokens(read_file_bytes(input_path), special_tokens)

    vocab, next_id = build_initial_vocab(special_tokens)

    word_counts = Counter({tuple(w): f for w, f in raw_counts.items()})

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(word_counts)

        if not pair_counts:
            break

        best = select_best_pair(pair_counts, vocab)

        merges.append((vocab[best[0]], vocab[best[1]]))

        new_id = next_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        next_id += 1

        new_counts: Counter = Counter()
        for word, freq in word_counts.items():
            new_counts[apply_merge(word, best, new_id)] += freq
        word_counts = new_counts

    return vocab, merges

if __name__ == "__main__":
    import os
    import tempfile

    # --- 1. pattern sanity: the required hand-picked sentence ---
    sentence = "Hello world, don't stop!"
    expected = ["Hello", " world", ",", " don", "'t", " stop", "!"]
    assert PRETOKEN_PAT.findall(sentence) == expected, PRETOKEN_PAT.findall(sentence)
    print("findall :", PRETOKEN_PAT.findall(sentence))

    # --- 2. bytes -> str -> bytes boundary ---
    assert pretokenize_span(sentence.encode("utf-8")) == [t.encode("utf-8") for t in expected]
    print("span    :", pretokenize_span(sentence.encode("utf-8")))

    # --- 3. literal special-token splitting ---
    assert split_on_specials(b"A<|endoftext|>B", ["<|endoftext|>"]) == [b"A", b"B"]
    assert split_on_specials(b"<|endoftext|>A", ["<|endoftext|>"]) == [b"", b"A"]   # leading empty
    assert split_on_specials(b"A<|endoftext|>", ["<|endoftext|>"]) == [b"A", b""]   # trailing empty
    assert split_on_specials(b"A<|e|>B<|z|>C", ["<|e|>", "<|z|>"]) == [b"A", b"B", b"C"]
    assert split_on_specials(b"hello", []) == [b"hello"]                            # no specials -> one span
    print("split   : OK")

    # --- 4. counts + the no-drop/no-dup invariant ---
    data = b"Hello world, don't stop! Hello world, don't stop! Done."
    flat = [tok for span in split_on_specials(data, ["<|endoftext|>"])
                 for tok in pretokenize_span(span)]
    counts = count_pretokens(data, ["<|endoftext|>"])
    assert sum(counts.values()) == len(flat), (sum(counts.values()), len(flat))
    assert Counter(flat) == counts
    print("counts  :", counts)
    print("invariant:", sum(counts.values()), "==", len(flat))

    # --- 5. end-to-end file round trip (temp file, cleaned up) ---
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"Hello<|endoftext|>it's fine Hello")
        path = tf.name
    try:
        assert pretokenize_corpus(path, ["<|endoftext|>"]) == [b"Hello", b"it", b"'s", b' fine', b' Hello']
        assert count_corpus(path, ["<|endoftext|>"]) == Counter(
            {b"Hello": 1, b"it": 1, b"'s": 1, b" fine": 1, b" Hello": 1}
        )
        print("file    : OK")
    finally:
        os.remove(path)
    vocab, next_id = build_initial_vocab(["<|endoftext|>"])
    assert len(vocab) == 257
    assert vocab[65] == b"A"
    assert vocab[256] == b"<|endoftext|>"
    assert next_id == 257
    print("\nall tests passed")