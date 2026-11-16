# RagChatBot — RAG chatbot over PDFs, with a semantic cache

Upload PDFs, ask questions, get answers grounded in your documents —
with a **semantic cache** that recognises reworded repeat questions and
answers them from the cache instead of calling the LLM again.

## How it works

**RAG pipeline** (`session_4_rag_backend.py` + `rag_streamlit.py`)

1. PDFs are parsed with pdfplumber and split into 1,000-character
   chunks (200-character overlap).
2. Chunks are embedded with Gemini `text-embedding-004` and stored in
   ChromaDB.
3. Each question retrieves the top-3 chunks; Gemini answers from that
   context only (temperature 0.2), and refuses when the context does
   not contain the answer.

**Semantic cache** (`semantic_cache.py`)

1. Every question is embedded with a free local embedding model and
   checked against a separate ChromaDB collection of past questions.
2. If a past question is similar enough (cosine similarity above the
   threshold), the stored answer is served immediately — **zero LLM
   calls, ~60 ms lookup**.
3. Otherwise the normal RAG pipeline runs, and the new Q&A pair is
   added to the cache.

Design decisions that matter:

- **Separate collection** — cached Q&A pairs live in their own
  collection, so they never pollute document retrieval.
- **Document versioning** — every cached answer is tagged with a hash
  of the documents it came from. Re-upload different PDFs and old
  answers are never served.
- **Refusals are never cached** — "I cannot answer this based on the
  provided context." is not stored, so a question that was once
  unanswerable can be answered later.
- **Fail-open** — if the cache errors, the app falls back to normal
  RAG. The cache can never break answering.
- **FIFO eviction** — the cache caps at 500 entries (oldest first).

The Streamlit sidebar shows live cache stats (hit rate, entries, hit
vs miss latency, approx. tokens saved), a threshold slider, and a
clear-cache button. Cached answers are marked with a ⚡ badge showing
the similarity and the matched question.

## Measured results

Evaluated with `evaluate_cache.py` on a fixed hand-written test set:
**8 seed questions, 24 paraphrases, 16 traps** (questions that look
similar but need a different answer — e.g. "pause my subscription" vs
"cancel my subscription"). Three embedding models were compared on
this same test set; **BAAI/bge-base-en-v1.5** was selected.

| threshold | paraphrase hit rate (n=24) | trap false-positive rate (n=16) | avg lookup |
|---:|---:|---:|---:|
| 0.75 | 95.8% | 62.5% | 59.7 ms |
| 0.80 | 58.3% | 37.5% | 62.8 ms |
| 0.85 | 41.7% | 12.5% | 47.3 ms |
| **0.90** | **16.7%** | **0.0%** | **60.3 ms** |
| 0.95 | 0.0% | 0.0% | 39.9 ms |

Lookup latency was measured on a shared 1.5-vCPU VM; the 0.90 row was
measured twice (45.3 ms and 75.3 ms, mean 60.3 ms). The other two
models, measured on the same test set: all-MiniLM-L6-v2 reached 12.5%
hits at 0% false positives (threshold 0.85), and bge-small-en-v1.5
reached 25% hits but with a 6.25% false-positive rate (threshold 0.90).
bge-base at 0.90 was chosen as the highest hit rate with zero false
positives.

Recommended threshold: **0.90**.

Known limitation, found by the evaluation: embedding similarity alone
cannot reliably distinguish questions that differ by a single word
("pause" vs "cancel"). The recommended threshold is the operating
point that keeps this false-positive rate low while still catching
reworded questions. A cross-encoder verification step would be the
next improvement.

## RAG security testing

Documents a RAG chatbot retrieves are untrusted input: a PDF can hide
instructions ("ignore previous instructions", "you are now...") that
the model may follow. `evaluate_security.py` measures that risk on a
fixed test set of 12 injection attacks (each embedded in a realistic
policy/manual chunk, each asking the model to output a unique canary
word) plus 12 benign chunks, then re-measures with the guardrails on.

Guardrails (`rag_guardrails.py`): a transparent rule-based detector
flags injection-like imperatives aimed at the assistant, flagged
chunks are dropped before they reach the prompt, and a hardened prompt
wraps the context in delimiters and states the documents are data, not
instructions. The app's query path uses them by default
(`use_guardrails=True` in `query_with_full_context`).

Measured 2026-11-17 (victim LLM: Pollinations text API, model=openai;
retrieval: local bge-base embeddings, top-3, 12 attacks):

| metric | result |
|---|---:|
| Retrieval exposure (injected chunk in top-3) | 11/12 (92%) |
| Attack success rate BEFORE guardrails | 9/12 (75%) |
| Attack success rate BEFORE, retrieved attacks only | 9/11 (82%) |
| Attack success rate AFTER guardrails | 0/12 (0%) |
| Filter: injected chunks flagged | 12/12 |
| Filter: benign chunks flagged (false positives) | 0/12 (0%) |
| Scan latency overhead | 0.16 ms per chunk |

The unguarded model complied with 9 of the 11 retrieved attacks --
its answers literally start with the injected canary word -- including
a system-prompt extraction attack that got it to output ~1,200
characters of system-prompt-like text. The two attacks it resisted
were verbatim context exfiltration and a pirate role-play. With the
guardrails on, the filter dropped the injected chunk from every
retrieved set (1 of 3 chunks each time) and no attack succeeded; the
remaining benign chunks still answered every question correctly.

Limitations: the victim LLM is Pollinations' free text API
(model=openai), measured 2026-11-17 -- a different model may comply
more or less, and Pollinations responses are non-deterministic, so
re-runs may differ by an attack or two. The detector is rule-based,
not ML: a paraphrased injection that avoids every pattern will pass
the filter (the hardened prompt is the second layer). Canary presence
is the attack-success signal; a model quoting the injection rather
than following it would also trip it -- every answer is saved in the
JSONL file for audit.

## Run it

```bash
pip install -r requirements.txt
streamlit run rag_streamlit.py
```

You need a **Gemini API key** for document embedding and answering
(paste it in the app's sidebar). The cache itself uses free local
embeddings and needs no key — first run downloads the model (~440 MB).

## Files

- `rag_streamlit.py` — Streamlit app (RAG + cache wiring)
- `session_4_rag_backend.py` — PDF parsing, chunking, embeddings, RAG
- `evaluate_cache.py` — test-set generator + threshold sweep
- `test_semantic_cache.py` — 15 unit tests (no API key needed)
- `rag_guardrails.py` — prompt-injection detector, sanitizer, hardened prompt
- `security_test_set.py` — 12 injection attacks + 12 benign chunks
- `evaluate_security.py` — before/after security evaluation harness
- `test_guardrails.py` — 13 unit tests (no API calls)
- `INTEGRATION.md` — what changed where, and how to verify it

## Re-measure it

```bash
# Unit tests
python test_semantic_cache.py

# Threshold sweep with local embeddings (no API key needed)
python evaluate_cache.py run --local

# Generate a test set about YOUR document (needs a Gemini API key)
python evaluate_cache.py generate --api-key YOUR_KEY --out test_set.json
python evaluate_cache.py run --local --test-set test_set.json

# Security evaluation: injection attacks before/after guardrails
python evaluate_security.py run --skip-llm   # retrieval + filter only, no LLM calls
python evaluate_security.py run              # full run (needs internet, ~10 min)
python test_guardrails.py                    # guardrail unit tests, no API calls
```
