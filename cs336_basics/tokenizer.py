import cs336_basics.bpe as bpe
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        self.byte_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        self.pair_to_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        self.special_to_id: dict[str, int] = {}
        self._special_bytes: dict[bytes, int] = {}
        next_id = (max(self.vocab) + 1) if self.vocab else 0
        for tok in self.special_tokens:
            if tok == "":
                raise ValueError("empty special token not allowed")
            b = tok.encode("utf-8")
            if b not in self.byte_to_id:
                self.vocab[next_id] = b
                self.byte_to_id[b] = next_id
                next_id += 1
            self.special_to_id[tok] = self.byte_to_id[b]
            self._special_bytes[b] = self.byte_to_id[b]

        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self.special_pat = (
            re.compile(b"(" + b"|".join(re.escape(t.encode("utf-8")) for t in ordered) + b")")
            if ordered
            else None
        )

    def _lowest_rank_pair(self, word: tuple[int, ...]) -> tuple[int, int] | None:
        best_pair, best_rank = None, None
        for a, b in zip(word, word[1:]):
            rank = self.pair_to_rank.get((self.vocab[a], self.vocab[b]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank, best_pair = rank, (a, b)
        return best_pair

    def _merge_word(self, word: tuple[int, ...]) -> tuple[int, ...]:
        while len(word) >= 2:
            pair = self._lowest_rank_pair(word)
            if pair is None:
                break
            new_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            word = bpe.apply_merge(word, pair, self.byte_to_id[new_bytes])
        return word

    def _encode_ordinary(self, piece: bytes) -> list[int]:
        ids: list[int] = []
        for pretoken in bpe.pretokenize_span(piece):
            word = tuple(self.byte_to_id[bytes([b])] for b in pretoken)
            ids.extend(self._merge_word(word))
        return ids

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        pieces = self.special_pat.split(data) if self.special_pat else [data]
        ids: list[int] = []
        for piece in pieces:
            if piece in self._special_bytes:
                ids.append(self._special_bytes[piece])
            elif piece:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")