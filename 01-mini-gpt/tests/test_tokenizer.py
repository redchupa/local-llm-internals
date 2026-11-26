"""Unit tests for ByteBPETokenizer."""

import sys
import tempfile
import unittest
from pathlib import Path

# Allow running directly via `python -m unittest discover -s 01-mini-gpt/tests`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizer import ByteBPETokenizer, get_pair_counts, merge_pair  # noqa: E402


class TestByteBPETokenizer(unittest.TestCase):
    def test_get_pair_counts_and_merge_pair(self) -> None:
        ids = [1, 2, 3, 1, 2]
        counts = get_pair_counts(ids)
        self.assertEqual(counts[(1, 2)], 2)
        self.assertEqual(counts[(2, 3)], 1)
        self.assertEqual(merge_pair(ids, (1, 2), 256), [256, 3, 256])

    def test_roundtrip_multilingual_and_emoji(self) -> None:
        corpus = (
            "Hello world! Hello LLM! "
            "안녕하세요 로컬 LLM 연구실입니다. 안녕하세요! "
            "🚀🔥 Transformer RoPE GQA"
        )
        tok = ByteBPETokenizer()
        tok.train(corpus, vocab_size=285)

        encoded = tok.encode(corpus)
        decoded = tok.decode(encoded)

        self.assertEqual(decoded, corpus)
        self.assertLess(len(encoded), len(corpus.encode("utf-8")))

    def test_unseen_text_without_unk(self) -> None:
        tok = ByteBPETokenizer()
        tok.train("aaa bbb aaa bbb", vocab_size=260)

        unseen = "새로운 한글 문장과 특수문자 @#$% ✨"
        self.assertEqual(tok.decode(tok.encode(unseen)), unseen)

    def test_empty_string(self) -> None:
        tok = ByteBPETokenizer()
        tok.train("hello hello", vocab_size=260)
        self.assertEqual(tok.encode(""), [])
        self.assertEqual(tok.decode([]), "")

    def test_invalid_vocab_size_raises(self) -> None:
        tok = ByteBPETokenizer()
        with self.assertRaises(ValueError):
            tok.train("hello", vocab_size=100)

    def test_save_and_load_consistency(self) -> None:
        corpus = "transformer attention attention transformer rope gqa"
        tok = ByteBPETokenizer()
        tok.train(corpus, vocab_size=275)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bpe_vocab.json"
            tok.save(path)
            loaded = ByteBPETokenizer.load(path)

        self.assertEqual(tok.merges, loaded.merges)
        self.assertEqual(tok.vocab, loaded.vocab)
        self.assertEqual(tok.encode(corpus), loaded.encode(corpus))
        self.assertEqual(loaded.decode(loaded.encode(corpus)), corpus)


if __name__ == "__main__":
    unittest.main()
