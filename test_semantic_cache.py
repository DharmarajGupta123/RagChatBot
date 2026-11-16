"""
test_semantic_cache.py -- Unit tests for semantic_cache.py.

Run with plain Python, no API key and no LangChain install needed:

    python test_semantic_cache.py

A fake in-memory store stands in for ChromaDB, and a bag-of-words
embedder stands in for Gemini embeddings. Word overlap is a crude proxy
for real semantic similarity, but it is deterministic, which is exactly
what these tests need: they verify the cache's own logic (threshold
math, doc-version filtering, refusal filtering, eviction, stats), not
the quality of any embedding model. Real-embedding behaviour is measured
by evaluate_cache.py instead.
"""

import math
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")  # so `import semantic_cache` works from the repo root

from semantic_cache import (  # noqa: E402
    CacheHit,
    SemanticCache,
    compute_doc_version,
    normalize_question,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEmbedder:
    """Bag-of-words vectors over a fixed vocabulary, L2-normalised.

    Cosine similarity then equals word overlap, which lets the tests use
    exact expected values, e.g. "what is the refund policy" vs
    "what is the refund process" share 4 of 5 words -> similarity 0.8.
    """

    def __init__(self, vocab):
        self.index = {word: i for i, word in enumerate(sorted(vocab))}

    def embed(self, text):
        vec = [0.0] * len(self.index)
        for token in _tokens(text):
            if token in self.index:
                vec[self.index[token]] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


class FakeStore:
    """Minimal in-memory stand-in for langchain_community Chroma.

    Implements only the methods SemanticCache uses, with the same
    signatures: add_texts, similarity_search_with_score (returning cosine
    DISTANCE, like real ChromaDB), get, delete, persist.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.entries = {}  # id -> {"text", "vector", "metadata"}

    def add_texts(self, texts, metadatas=None, ids=None):
        for i, text in enumerate(texts):
            entry_id = ids[i] if ids else f"id-{len(self.entries)}"
            self.entries[entry_id] = {
                "text": text,
                "vector": self.embedder.embed(text),
                "metadata": metadatas[i] if metadatas else {},
            }
        return ids or []

    def similarity_search_with_score(self, query, k=1, filter=None):
        query_vec = self.embedder.embed(query)
        scored = []
        for entry in self.entries.values():
            meta = entry["metadata"]
            if filter and any(meta.get(key) != val for key, val in filter.items()):
                continue
            distance = 1.0 - _cosine(query_vec, entry["vector"])
            doc = SimpleNamespace(page_content=entry["text"], metadata=meta)
            scored.append((doc, distance))
        scored.sort(key=lambda item: item[1])
        return scored[:k]

    def get(self, where=None):
        ids, metadatas = [], []
        for entry_id, entry in self.entries.items():
            meta = entry["metadata"]
            if where and any(meta.get(key) != val for key, val in where.items()):
                continue
            ids.append(entry_id)
            metadatas.append(meta)
        return {"ids": ids, "metadatas": metadatas}

    def delete(self, ids):
        for entry_id in ids:
            self.entries.pop(entry_id, None)

    def persist(self):
        pass


# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------

QUESTIONS = [
    "How do I reset my password?",
    "How do I reset password?",
    "What is the refund policy?",
    "What is the refund process?",
    "How do I track my delivery order?",
    "What are the delivery charges?",
    "How do I cancel my subscription?",
    "Where is my order right now?",
]

VOCAB = {token for q in QUESTIONS for token in _tokens(q)}
V1 = compute_doc_version([("guide.pdf", b"fake-pdf-bytes-v1")])
V2 = compute_doc_version([("guide.pdf", b"fake-pdf-bytes-v2")])


def make_cache(**kwargs):
    kwargs.setdefault("similarity_threshold", 0.85)
    kwargs.setdefault("max_entries", 500)
    return SemanticCache(store=FakeStore(FakeEmbedder(VOCAB)), **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_exact_match_returns_cached_answer():
    cache = make_cache()
    assert cache.add("How do I reset my password?", "Go to Settings > Security.", V1) is True
    hit = cache.lookup("How do I reset my password?", V1)
    assert isinstance(hit, CacheHit)
    assert hit.answer == "Go to Settings > Security."
    assert hit.source == "exact"
    assert hit.similarity == 1.0


def test_exact_match_ignores_case_and_spacing():
    cache = make_cache()
    cache.add("How do I reset my password?", "Go to Settings > Security.", V1)
    hit = cache.lookup("  HOW do I   reset my password? ", V1)
    assert hit is not None and hit.source == "exact"


def test_paraphrase_hits_and_similarity_is_correct():
    # "How do I reset my password?" (6 words) vs "How do I reset
    # password?" (5 words): 5 shared words -> 5/sqrt(6*5) = 0.913.
    cache = make_cache()
    cache.add("How do I reset my password?", "Go to Settings > Security.", V1)
    hit = cache.lookup("How do I reset password?", V1)
    assert hit is not None, "paraphrase should hit at threshold 0.85"
    assert hit.source == "semantic"
    assert abs(hit.similarity - 0.913) < 0.01, f"got {hit.similarity}"


def test_threshold_is_respected():
    # "What is the refund policy?" vs "What is the refund process?":
    # 4 shared of 5 words -> similarity 0.8. Hits at 0.75, misses at 0.85.
    strict = make_cache(similarity_threshold=0.85)
    strict.add("What is the refund policy?", "Refunds within 30 days.", V1)
    assert strict.lookup("What is the refund process?", V1) is None

    loose = make_cache(similarity_threshold=0.75)
    loose.add("What is the refund policy?", "Refunds within 30 days.", V1)
    hit = loose.lookup("What is the refund process?", V1)
    assert hit is not None and abs(hit.similarity - 0.8) < 0.01


def test_unrelated_question_misses():
    cache = make_cache()
    cache.add("What is the refund policy?", "Refunds within 30 days.", V1)
    assert cache.lookup("How do I track my delivery order?", V1) is None


def test_empty_cache_misses():
    assert make_cache().lookup("What is the refund policy?", V1) is None


def test_doc_version_isolates_entries():
    cache = make_cache()
    cache.add("What is the refund policy?", "Refunds within 30 days.", V1)
    assert cache.lookup("What is the refund policy?", V2) is None, \
        "answers from other documents must never be served"
    assert cache.lookup("What is the refund policy?", V1) is not None


def test_refusal_answers_are_never_cached():
    cache = make_cache()
    added = cache.add(
        "What is the meaning of life?",
        "I cannot answer this based on the provided context.",
        V1,
    )
    assert added is False
    assert cache.lookup("What is the meaning of life?", V1) is None


def test_error_and_empty_answers_are_never_cached():
    cache = make_cache()
    assert cache.add("Q1?", "Error generating response: boom", V1) is False
    assert cache.add("Q2?", "   ", V1) is False
    assert cache.summary()["entries"] == 0


def test_duplicate_add_is_rejected():
    cache = make_cache()
    assert cache.add("What is the refund policy?", "Refunds within 30 days.", V1) is True
    assert cache.add("What is the refund policy?", "Refunds within 30 days.", V1) is False
    assert cache.summary()["entries"] == 1


def test_fifo_eviction_at_capacity():
    cache = make_cache(max_entries=3)
    cache.add("What is the refund policy?", "A1.", V1)
    cache.add("How do I track my delivery order?", "A2.", V1)
    cache.add("What are the delivery charges?", "A3.", V1)
    cache.add("How do I cancel my subscription?", "A4.", V1)
    assert cache.summary()["entries"] == 3
    assert cache.lookup("What is the refund policy?", V1) is None, \
        "oldest entry should have been evicted (store and memory)"
    assert cache.lookup("How do I cancel my subscription?", V1) is not None


def test_stats_accounting():
    cache = make_cache()
    cache.add("What is the refund policy?", "Refunds within 30 days.", V1)
    cache.lookup("What is the refund policy?", V1)   # hit
    cache.lookup("How do I track my delivery order?", V1)  # miss
    cache.record_miss_latency(2.5)
    stats = cache.summary()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["entries"] == 1
    assert stats["avg_miss_latency_ms"] == 2500.0
    assert stats["avg_hit_latency_ms"] is not None
    assert stats["approx_output_tokens_saved"] > 0


def test_clear_removes_only_the_given_version():
    cache = make_cache()
    cache.add("What is the refund policy?", "Refunds within 30 days.", V1)
    cache.add("What is the refund policy?", "Refunds within 14 days.", V2)
    assert cache.clear(doc_version=V1) == 1
    assert cache.lookup("What is the refund policy?", V1) is None
    hit = cache.lookup("What is the refund policy?", V2)
    assert hit is not None and hit.answer == "Refunds within 14 days."


def test_doc_version_changes_when_files_change():
    same = compute_doc_version([("guide.pdf", b"fake-pdf-bytes-v1")])
    assert same == V1
    assert V1 != V2
    assert compute_doc_version([("guide.pdf", b"fake-pdf-bytes-v1"),
                                ("extra.pdf", b"more")]) != V1


def test_normalize_question():
    assert normalize_question("  WHAT   is   X? ") == "what is x?"


# ---------------------------------------------------------------------------
# Runner (no pytest needed)
# ---------------------------------------------------------------------------

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except AssertionError as exc:
            failed.append((name, str(exc) or "assertion failed"))
            print(f"FAIL {name}: {exc or 'assertion failed'}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed} passed, {len(failed)} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
