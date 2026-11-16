# Semantic cache — integration record

Status: **wired in.** The cache is live in `rag_streamlit.py`; this
document records what changed, how to verify it, and how to re-measure
it. The numbers in `README.md` come from `evaluate_cache.py run --local`.

## New files

- `semantic_cache.py` — the cache module (local bge-base embeddings,
  separate `semantic_cache` ChromaDB collection, doc-version filtering,
  refusal filtering, FIFO eviction, fail-open).
- `evaluate_cache.py` — test-set generator + threshold sweep.
- `test_semantic_cache.py` — 15 unit tests, no API key needed.

No new dependencies: everything reuses `requirements.txt`
(`langchain`, `langchain-google-genai`, `langchain_community`,
`chromadb`, `sentence-transformers`).

## What changed in `rag_streamlit.py`

1. **Imports** — `time`, plus `SemanticCache, compute_doc_version`
   from `semantic_cache`.
2. **Session state** — new keys `semantic_cache` and `doc_version`.
3. **`process_documents()`** — after the embedding-model setup, the
   uploaded files are hashed into a `doc_version` and the cache is
   created once (`SemanticCache(persist_directory="./cache_chroma_db")`).
   The cache uses free local embeddings, so this step needs no API key.
   First run downloads the ~440 MB bge-base model.
4. **`handle_user_query()`** — the cache is checked before RAG. On a
   hit, the stored answer is shown with a ⚡ badge (similarity + the
   matched question). On a miss, RAG runs as before, its latency is
   recorded, and the new Q&A pair is added to the cache.
5. **`display_chat()`** — cached answers show the badge; they have no
   fresh source context, so no context expander appears for them.
6. **Sidebar** — a "Semantic Cache" section (hit rate, entries,
   hits/misses, avg hit vs miss latency, approx. tokens saved, clear
   button) and a similarity-threshold slider under Advanced Options.

## Bug fixed along the way

`process_documents()` built a `metadatas` list (each chunk's source
file) but never passed it to `store_embeddings`, so the source was
dropped. It now passes `metadatas=metadatas`.

## Verify

```bash
# 1. Unit tests — no API key needed, should print 15 passed
python test_semantic_cache.py

# 2. Eval harness plumbing check — no API key needed
python evaluate_cache.py run --smoke

# 3. Real numbers with local embeddings — no API key needed
python evaluate_cache.py run --local

# 4. Run the app and try it: ask a question, then ask it again in
#    different words. The second answer should show the ⚡ badge.
streamlit run rag_streamlit.py
```

## Re-measuring with your own questions

```bash
# Generate a test set about YOUR document (needs a Gemini API key;
# review the output by hand first!)
python evaluate_cache.py generate --api-key YOUR_KEY --out test_set.json

# Sweep thresholds with local embeddings (no key needed)
python evaluate_cache.py run --local --test-set test_set.json
```

Prefer the `GOOGLE_API_KEY` environment variable over `--api-key` so
the key never lands in your shell history. Never paste the key into
any source file.

## Troubleshooting

- **"No module named langchain_community"** (or
  `sentence_transformers`) — install the repo deps:
  `pip install -r requirements.txt`.
- **Cache never hits** — check the threshold slider (try 0.80), and
  confirm you did not re-process different PDFs (a new `doc_version`
  correctly isolates old entries).
- **ChromaDB deprecation warnings about `persist()`** — harmless; the
  module guards the call for old and new versions.
- **Eviction prints a warning** — your installed ChromaDB version may
  not support the listing call the eviction uses. Safe to ignore: the
  cache just grows past `max_entries` instead of evicting.
- **"Clear cache" shows 0 removed** — the listing call failed (see
  above). Delete the `./cache_chroma_db` folder while the app is
  stopped to wipe it manually.
