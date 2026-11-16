"""
evaluate_cache.py -- Measure the semantic cache honestly.

Two modes:

  1. Generate a test set with Gemini (needs an API key):
       python evaluate_cache.py generate --api-key YOUR_KEY --out test_set.json
       python evaluate_cache.py generate --api-key YOUR_KEY --seeds my_seeds.json

     Writes one entry per seed question: 3 paraphrases (same meaning,
     different wording -- these SHOULD hit the cache) and 2 traps
     (share keywords but mean something different -- these should MISS).
     Review the generated file by hand before trusting it: an
     LLM-written paraphrase can drift in meaning, and a drifted
     "paraphrase" would punish the cache unfairly.

  2. Run the evaluation:
       python evaluate_cache.py run --local
       python evaluate_cache.py run --local --test-set test_set.json
       python evaluate_cache.py run --api-key YOUR_KEY --test-set test_set.json
       python evaluate_cache.py run --smoke        (no key: plumbing check only)

     --local is the recommended mode: it uses the same free local
     bge-base embeddings the app's cache uses (no API key, no credits),
     with an in-memory store doing exact cosine search -- the same math
     ChromaDB does, so the threshold numbers transfer to the real app.
     --api-key runs the same sweep against Gemini embeddings in a real
     ChromaDB collection instead. --smoke uses bag-of-words embeddings:
     word overlap cannot do semantics, so its hit rates look bad on
     purpose; it only proves the harness itself runs.

     For each threshold, the script seeds a fresh cache with the seed
     questions, then queries every paraphrase and every trap and reports:
       * paraphrase hit rate  (higher is better -- the cache is useful)
       * trap false-positive rate (lower is better -- the cache is safe)
       * average lookup latency
     The recommended threshold is the one with the highest paraphrase
     hit rate among thresholds with zero trap false positives (serving
     a wrong answer is worse than a cache miss); if no threshold is
     clean, it falls back to maximising (hit rate - false-positive
     rate).

The answers stored during evaluation are canned placeholders. That is
deliberate: this script measures cache retrieval behaviour (does the
right question match?), not answer quality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, ".")  # so `import semantic_cache` works from the repo root

from semantic_cache import SemanticCache, init_cache_embeddings  # noqa: E402

DEFAULT_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]

# Generic default test set. It measures the cache mechanics fine, but for
# numbers you can quote on your resume, generate a set from questions
# about YOUR document instead (see `generate` mode).
DEFAULT_TEST_SET = [
    {
        "seed": "What is the refund policy?",
        "paraphrases": [
            "How do refunds work?",
            "Can I get my money back if I return something?",
            "What are the rules for getting a refund?",
        ],
        "traps": [
            "What is the refund policy for sale items?",
            "Can I get a refund on my subscription?",
        ],
    },
    {
        "seed": "How do I reset my password?",
        "paraphrases": [
            "I forgot my password, what should I do?",
            "How can I change my password?",
            "What are the steps to recover my password?",
        ],
        "traps": [
            "How do I reset my device to factory settings?",
            "How do I change my username?",
        ],
    },
    {
        "seed": "How long does delivery take?",
        "paraphrases": [
            "When will my order arrive?",
            "What is the expected delivery time?",
            "How many days until my package gets here?",
        ],
        "traps": [
            "How long does a refund take to reach my account?",
            "When will the sale end?",
        ],
    },
    {
        "seed": "What are the delivery charges?",
        "paraphrases": [
            "How much does shipping cost?",
            "Is delivery free?",
            "What do I pay for shipping?",
        ],
        "traps": [
            "What are the charges for returning an item?",
            "How much does express delivery cost?",
        ],
    },
    {
        "seed": "How do I cancel my subscription?",
        "paraphrases": [
            "How can I stop my subscription?",
            "What is the process to unsubscribe?",
            "I want to end my subscription, how?",
        ],
        "traps": [
            "How do I pause my subscription?",
            "How do I cancel my order?",
        ],
    },
    {
        "seed": "How do I track my order?",
        "paraphrases": [
            "Where is my order right now?",
            "Can I see the status of my delivery?",
            "How do I check where my package is?",
        ],
        "traps": [
            "How do I track my refund?",
            "How do I change my delivery address?",
        ],
    },
    {
        "seed": "What payment methods do you accept?",
        "paraphrases": [
            "How can I pay for my order?",
            "Which payment options are available?",
            "Can I pay with a credit card?",
        ],
        "traps": [
            "How do I update my saved payment method?",
            "When will my payment be charged?",
        ],
    },
    {
        "seed": "How do I contact customer support?",
        "paraphrases": [
            "What is the support email address?",
            "How can I reach the help team?",
            "Where do I raise a complaint?",
        ],
        "traps": [
            "How do I contact the sales team?",
            "What are the customer support timings?",
        ],
    },
]

GENERATE_PROMPT = """You are building a test set for a semantic cache. For the question
below, write:
- 3 paraphrases: the SAME meaning, genuinely different wording.
- 2 traps: questions that share keywords with it but mean something
  DIFFERENT, so answering them with this question's answer would be wrong.

Question: "{seed}"

Reply with JSON only, exactly in this shape:
{{"paraphrases": ["...", "...", "..."], "traps": ["...", "..."]}}"""


# ---------------------------------------------------------------------------
# Fake embeddings for --smoke (mirrors the fakes in test_semantic_cache.py)
# ---------------------------------------------------------------------------

def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class _FakeEmbedder:
    def __init__(self, vocab):
        self.index = {word: i for i, word in enumerate(sorted(vocab))}

    def embed(self, text):
        vec = [0.0] * len(self.index)
        for token in _tokens(text):
            if token in self.index:
                vec[self.index[token]] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class _LocalEmbedderAdapter:
    """Adapts LocalEmbeddings (embed_query) to the .embed() interface
    _FakeStore expects, so --local mode runs real bge-base embeddings
    through the same in-memory cosine search."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def embed(self, text):
        return self._embeddings.embed_query(text)


class _FakeStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.entries = {}

    def add_texts(self, texts, metadatas=None, ids=None):
        for i, text in enumerate(texts):
            entry_id = ids[i] if ids else f"id-{len(self.entries)}"
            self.entries[entry_id] = {
                "text": text,
                "vector": self.embedder.embed(text),
                "metadata": metadatas[i] if metadatas else {},
            }

    def similarity_search_with_score(self, query, k=1, filter=None):
        from types import SimpleNamespace

        query_vec = self.embedder.embed(query)
        scored = []
        for entry in self.entries.values():
            meta = entry["metadata"]
            if filter and any(meta.get(key) != val for key, val in filter.items()):
                continue
            similarity = sum(x * y for x, y in zip(query_vec, entry["vector"]))
            doc = SimpleNamespace(page_content=entry["text"], metadata=meta)
            scored.append((doc, 1.0 - similarity))  # ChromaDB returns distance
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
# Generate mode
# ---------------------------------------------------------------------------

def cmd_generate(args):
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("generate mode needs a Gemini API key: pass --api-key or set GOOGLE_API_KEY.")
        return 1
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("langchain_google_genai is not installed. Run: pip install -r requirements.txt")
        return 1

    if args.seeds:
        with open(args.seeds, encoding="utf-8") as f:
            seeds = json.load(f)
    else:
        seeds = [entry["seed"] for entry in DEFAULT_TEST_SET]

    llm = ChatGoogleGenerativeAI(model=args.model, temperature=0.3)
    test_set = []
    for seed in seeds:
        print(f"Generating variants for: {seed}")
        response = llm.invoke(GENERATE_PROMPT.format(seed=seed)).content
        cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            variants = json.loads(cleaned)
            test_set.append({"seed": seed, **variants})
        except json.JSONDecodeError:
            print(f"  WARNING: could not parse Gemini's reply for {seed!r}; skipped.")
            print(f"  Raw reply: {response[:300]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(test_set)} entries to {args.out}.")
    print("Review it by hand before running the evaluation: a paraphrase that")
    print("drifted in meaning, or a trap that is actually a paraphrase, will")
    print("make the numbers lie.")
    return 0


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------

def _load_test_set(path):
    if not path:
        return DEFAULT_TEST_SET
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _make_cache(threshold, fake_store, embedder, embedding_model):
    if fake_store:
        return SemanticCache(
            similarity_threshold=threshold, store=_FakeStore(embedder)
        )
    return SemanticCache(
        embedding_function=embedding_model,
        persist_directory=tempfile.mkdtemp(prefix="cache_eval_"),
        similarity_threshold=threshold,
    )


def _run_one_threshold(test_set, threshold, fake_store, embedder, embedding_model):
    cache = _make_cache(threshold, fake_store, embedder, embedding_model)
    doc_version = "eval-v1"
    for entry in test_set:
        cache.add(entry["seed"], f"Canned answer for: {entry['seed']}", doc_version)

    paraphrase_hits, paraphrase_total = 0, 0
    trap_hits, trap_total = 0, 0
    missed_paraphrases, false_positives = [], []
    latencies = []
    for entry in test_set:
        for question in entry["paraphrases"]:
            paraphrase_total += 1
            started = time.perf_counter()
            hit = cache.lookup(question, doc_version)
            latencies.append(time.perf_counter() - started)
            if hit:
                paraphrase_hits += 1
            else:
                missed_paraphrases.append((question, entry["seed"]))
        for question in entry["traps"]:
            trap_total += 1
            started = time.perf_counter()
            hit = cache.lookup(question, doc_version)
            latencies.append(time.perf_counter() - started)
            if hit:
                trap_hits += 1
                false_positives.append((question, hit.matched_question, hit.similarity))
    return {
        "threshold": threshold,
        "hit_rate": paraphrase_hits / paraphrase_total if paraphrase_total else 0.0,
        "false_positive_rate": trap_hits / trap_total if trap_total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 1) if latencies else 0.0,
        "missed_paraphrases": missed_paraphrases,
        "false_positives": false_positives,
    }


def cmd_run(args):
    test_set = _load_test_set(args.test_set)
    thresholds = args.thresholds or DEFAULT_THRESHOLDS

    embedder = embedding_model = None
    if args.local:
        try:
            embedder = _LocalEmbedderAdapter(init_cache_embeddings())
        except ImportError:
            print("sentence-transformers is not installed.")
            print("Run: pip install -r requirements.txt")
            return 1
        print("LOCAL MODE: real bge-base embeddings, in-memory cosine search.")
        print("Same model the app's cache uses, so these numbers transfer.\n")
    elif args.smoke:
        vocab = set()
        for entry in test_set:
            for question in [entry["seed"]] + entry["paraphrases"] + entry["traps"]:
                vocab.update(_tokens(question))
        embedder = _FakeEmbedder(vocab)
        print("SMOKE MODE: bag-of-words embeddings. Hit rates will look bad --")
        print("word overlap cannot do semantics. This only proves the harness runs.\n")
    else:
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("run mode needs an embedding source: use --local (free, no key),")
            print("pass --api-key / set GOOGLE_API_KEY for Gemini embeddings,")
            print("or use --smoke for a no-key plumbing check.")
            return 1
        try:
            from session_4_rag_backend import init_embedding_model, setup_api_key
        except ImportError as exc:
            print(f"Could not import session_4_rag_backend ({exc}).")
            print("Run this script from the RagChatBot repo root with the")
            print("requirements installed: pip install -r requirements.txt")
            return 1
        setup_api_key(api_key)
        embedding_model = init_embedding_model()
        if embedding_model is None:
            print("Failed to initialise the embedding model. Check your API key.")
            return 1

    print(f"Test set: {len(test_set)} seeds, "
          f"{sum(len(e['paraphrases']) for e in test_set)} paraphrases, "
          f"{sum(len(e['traps']) for e in test_set)} traps.\n")
    print(f"{'threshold':>9}  {'paraphrase hit rate':>19}  {'trap false-pos rate':>19}  {'avg lookup':>10}")
    results = []
    use_fake_store = args.smoke or args.local
    for threshold in thresholds:
        result = _run_one_threshold(test_set, threshold, use_fake_store, embedder, embedding_model)
        results.append(result)
        print(f"{threshold:>9.2f}  {result['hit_rate']:>19.2%}  "
              f"{result['false_positive_rate']:>19.2%}  {result['avg_latency_ms']:>8.1f} ms")

    # Best operating point: a cache that serves a wrong answer is worse
    # than a cache miss, so prefer the highest hit rate among thresholds
    # with zero false positives; fall back to maximising
    # (hit rate - false-positive rate) if no threshold is clean.
    zero_fp = [r for r in results if r["false_positive_rate"] == 0]
    if zero_fp:
        best = max(zero_fp, key=lambda r: (r["hit_rate"], r["threshold"]))
    else:
        best = max(results, key=lambda r: (r["hit_rate"] - r["false_positive_rate"], r["threshold"]))
    print(f"\nRecommended threshold: {best['threshold']:.2f} "
          f"(hit rate {best['hit_rate']:.0%}, false-positive rate "
          f"{best['false_positive_rate']:.0%}).")
    if best["false_positives"]:
        print("\nFalse positives at this threshold (inspect these!):")
        for question, matched, similarity in best["false_positives"]:
            print(f"  - {question!r} matched {matched!r} (sim {similarity:.2f})")
    if best["missed_paraphrases"]:
        print("\nMissed paraphrases at this threshold:")
        for question, seed in best["missed_paraphrases"]:
            print(f"  - {question!r} (seed: {seed!r})")
    print("\nThese numbers come from THIS test set. Quote them with the test")
    print("set size and threshold, e.g. '0.90 threshold, 24 paraphrase queries'.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a test set and evaluate the semantic cache."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a test set with Gemini")
    gen.add_argument("--api-key", default=None)
    gen.add_argument("--seeds", default=None,
                     help="JSON file with a list of seed questions")
    gen.add_argument("--model", default="gemini-2.0-flash")
    gen.add_argument("--out", default="test_set.json")
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="run the threshold evaluation")
    run.add_argument("--api-key", default=None)
    run.add_argument("--test-set", default=None,
                     help="test set JSON (default: built-in generic set)")
    run.add_argument("--thresholds", nargs="+", type=float, default=None)
    run.add_argument("--local", action="store_true",
                     help="free local bge-base embeddings (same model the app uses)")
    run.add_argument("--smoke", action="store_true",
                     help="fake embeddings; plumbing check only, no API key")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
