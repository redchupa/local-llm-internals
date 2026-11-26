"""Byte-level BPE (Byte Pair Encoding) Tokenizer implemented from scratch.

GPT-2 / GPT-4 / Llama 계열 LLM이 사용하는 Byte-level BPE의 핵심 원리를
외부 라이브러리 없이 순수 Python으로 구현합니다.

핵심 동작:
1. 모든 문자열을 UTF-8 바이트(0~255) 시퀀스로 변환합니다.
   -> 256개의 기본 토큰으로 세상의 모든 문자(영어, 한글, 이모지 등)를 <UNK> 없이 표현합니다.
2. 학습(train): 가장 자주 연속 등장하는 토큰 쌍(pair)을 찾아 새 토큰 ID(256, 257, ...)로 병합합니다.
3. 인코딩(encode): 학습된 병합 규칙을 우선순위대로 적용해 바이트 시퀀스를 압축합니다.
4. 디코딩(decode): 각 토큰 ID를 원래 바이트 열로 복원한 뒤 UTF-8 문자열로 디코딩합니다.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def get_pair_counts(ids: list[int]) -> Counter[tuple[int, int]]:
    """인접한 토큰 ID 쌍의 등장 빈도를 계산합니다."""
    counts: Counter[tuple[int, int]] = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """시퀀스 `ids`에서 `pair`가 등장하는 모든 위치를 `new_id`로 치환합니다."""
    merged: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class ByteBPETokenizer:
    """순수 Python 기반 Byte-level BPE 토크나이저."""

    def __init__(self) -> None:
        # merges: (token_a, token_b) -> new_token_id (학습 순서 = 병합 우선순위)
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: token_id -> bytes
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """주어진 텍스트로부터 `vocab_size` 크기의 BPE 어휘 사전을 학습합니다."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 (got {vocab_size})")

        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

        for step in range(num_merges):
            counts = get_pair_counts(ids)
            if not counts:
                break

            # 가장 빈도가 높은 인접 쌍 선택 (동점 시 결정적(deterministic) 선택)
            best_pair, best_freq = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_freq < 2:
                # 2회 미만 등장한 쌍은 압축 효과가 없으므로 조기 종료
                break

            new_id = 256 + step
            ids = merge_pair(ids, best_pair, new_id)
            merges[best_pair] = new_id
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

            if verbose:
                token_repr = vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"[merge {step + 1:03d}/{num_merges}] "
                    f"{best_pair} -> {new_id} (freq={best_freq}, token={token_repr!r})"
                )

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        """문자열을 BPE 토큰 ID 리스트로 인코딩합니다."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # 학습 시 가장 먼저 생성된(가장 낮은 new_id를 가진) 병합 규칙부터 적용
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            ids = merge_pair(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """토큰 ID 리스트를 원래 UTF-8 문자열로 디코딩합니다."""
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """학습된 병합 규칙을 JSON 파일로 저장합니다."""
        payload = {
            "version": "byte-bpe-v1",
            "merges": [
                {"pair": [p[0], p[1]], "new_id": new_id}
                for p, new_id in sorted(self.merges.items(), key=lambda kv: kv[1])
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        """저장된 JSON 파일에서 BPE 토크나이저를 복원합니다."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        for item in payload["merges"]:
            p0, p1 = item["pair"]
            new_id = item["new_id"]
            tokenizer.merges[(p0, p1)] = new_id
            tokenizer.vocab[new_id] = tokenizer.vocab[p0] + tokenizer.vocab[p1]
        return tokenizer


def _demo() -> None:
    sample_text = (
        "Local LLM internals: tokenization turns raw text into token IDs. "
        "토크나이저는 자연어를 숫자 시퀀스로 변환합니다. "
        "Tokenization and attention are the foundation of modern LLMs. "
        "토크나이저는 반복되는 바이트 쌍을 병합해 시퀀스 길이를 줄입니다."
    )
    tok = ByteBPETokenizer()
    tok.train(sample_text, vocab_size=300, verbose=True)

    encoded = tok.encode(sample_text)
    decoded = tok.decode(encoded)
    raw_len = len(sample_text.encode("utf-8"))

    print("\n--- Summary ---")
    print(f"Vocab size      : {tok.vocab_size}")
    print(f"Raw UTF-8 bytes : {raw_len}")
    print(f"Encoded tokens  : {len(encoded)}")
    print(f"Compression     : {raw_len / len(encoded):.2f}x")
    print(f"Exact roundtrip : {decoded == sample_text}")


if __name__ == "__main__":
    _demo()
