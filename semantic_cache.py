"""
semantic_cache.py -- Semantic cache layer for RagChatBot.

Wired into rag_streamlit.py (see INTEGRATION.md for what changed where,
and README.md for the measured results).

What it does
------------
Before a question goes through the normal RAG pipeline (retrieve chunks,
call Gemini), the cache checks whether a semantically similar question has
already been answered:

    new question -> exact-match check (free)
                 -> embed + similarity search over cached questions
                 -> similar enough? return the stored answer, no LLM call
                 -> no match? run RAG as normal, then store the new Q&A pair

Design decisions (the ones an interviewer will ask about)
----------------------------------------------------------
1.  Local embeddings for the cache. The cache uses a local model
    (BAAI/bge-base-en-v1.5 via sentence-transformers, see
    LocalEmbeddings) instead of the Gemini embeddings the RAG pipeline
    uses. That is deliberate: cache similarity is question-to-question,
    so it only needs its own embedding space to be internally
    consistent -- it never compares a question embedding with a
    document-chunk embedding. Local embeddings make every cache lookup
    free (zero API calls) and keep the cache working even when the
    Gemini API key has no credits left. bge-base was chosen after
    measuring three local models on a fixed 40-question test set (see
    README.md): at threshold 0.90 it was the only one with zero false
    positives. A Gemini `GoogleGenerativeAIEmbeddings` instance can
    still be passed as `embedding_function` if you prefer one model
    everywhere.
2.  Separate ChromaDB collection ("semantic_cache"). Cache entries never
    mix with document chunks, so retrieval over the documents is
    untouched.
3.  ChromaDB returns a cosine DISTANCE (0 = identical), not a similarity.
    The threshold is stored as a similarity (0.90) and converted with
    similarity = 1 - distance. Getting this backwards is the classic bug.
4.  Document-version filtering. Every entry is tagged with a hash of the
    PDFs it was answered from. A cached answer is only served when the
    current documents have the same version hash, so uploading a new PDF
    can never serve a stale answer.
5.  Refusals are never cached. If RAG answered "I cannot answer this
    based on the provided context", there is nothing worth reusing.
6.  Fail-open. If the cache itself errors (corrupt collection, missing
    dependency), lookup returns None and add returns False, so the
    chatbot keeps working through plain RAG.
7.  FIFO eviction. When the collection reaches max_entries, the oldest
    entry (by creation time) is deleted. LRU would need a metadata update
    on every hit; FIFO is simpler and good enough at this scale.

Only the standard library is imported at module level. LangChain/ChromaDB
and sentence-transformers are imported lazily, so this file can be
imported (and unit-tested with a fake store) without the full RAG stack
installed.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_COLLECTION = "semantic_cache"
DEFAULT_PERSIST_DIR = "./cache_chroma_db"
DEFAULT_THRESHOLD = 0.90
DEFAULT_MAX_ENTRIES = 500

# Answers that must never be cached: the RAG prompt's refusal sentence
# (see the prompt in query_with_full_context) and backend error strings.
REFUSAL_MARKERS = (
    "i cannot answer this based on the provided context",
    "error generating response",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_question(question: str) -> str:
    """Lowercase, strip, collapse whitespace.

    Makes the exact-match fast path insensitive to trivial differences:
    "  What is X? " and "what is x?" hit the same cache entry.
    """
    return re.sub(r"\s+", " ", question.strip().lower())


def compute_doc_version(files: List[Tuple[str, bytes]]) -> str:
    """Short hash identifying one exact set of uploaded documents.

    Args:
        files: list of (file name, file bytes) pairs, e.g.
            [(f.name, f.getvalue()) for f in st_uploaded_files]

    The hash covers each file's name, size and content, sorted by name,
    so the same set of PDFs always produces the same version and any
    change (new file, edited file, removed file) produces a new one.
    """
    digest = hashlib.sha256()
    for name, data in sorted(files, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(len(data)).encode("utf-8"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Local embeddings (no API key, no credits)
# ---------------------------------------------------------------------------

LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


class LocalEmbeddings:
    """LangChain-compatible wrapper around a local sentence-transformers
    model (BAAI/bge-base-en-v1.5, 768 dimensions, ~440 MB download on
    first use).

    Implements the two methods the Chroma wrapper needs: embed_documents
    and embed_query. Embeddings are L2-normalised, so cosine similarity
    is a plain dot product.

    Why local embeddings for the cache: the cache only ever compares
    questions with questions, so it does not need the Gemini embedding
    space the document chunks live in -- and a local model makes every
    lookup free and independent of API credits.
    """

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        # Lazy import: sentence-transformers (and torch) are only needed
        # when the cache actually runs, not when this module is imported
        # for tests.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(
            [text], normalize_embeddings=True
        )[0].tolist()


def init_cache_embeddings(
    model_name: str = LOCAL_EMBEDDING_MODEL,
) -> LocalEmbeddings:
    """Build the default local embedding model for the cache."""
    print(f"Loading local embedding model ({model_name})...")
    embeddings = LocalEmbeddings(model_name)
    print("Local embedding model ready")
    return embeddings


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CacheHit:
    answer: str
    similarity: float        # 0.0 - 1.0; 1.0 for an exact match
    matched_question: str    # the cached question that matched
    source: str              # "exact" or "semantic"


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Semantic cache backed by a dedicated ChromaDB collection.

    Args:
        embedding_function: embeddings for cache questions. Defaults to
            None, which builds a LocalEmbeddings (free, local bge-base).
            Pass the GoogleGenerativeAIEmbeddings instance from
            init_embedding_model() to use Gemini embeddings instead, or
            inject `store` (tests pass a fake store).
        persist_directory: where ChromaDB keeps the cache collection.
        collection_name: ChromaDB collection for cached Q&A pairs.
        similarity_threshold: minimum cosine similarity (0-1) for a
            cached question to count as "the same question". Start at
            0.90 (measured: zero false positives at 0.90; see README.md)
            and tune with evaluate_cache.py.
        max_entries: cap on stored Q&A pairs; oldest entries are evicted
            first (FIFO) once the cap is reached.
        store: optional pre-built vector store (same interface as
            langchain_community.vectorstores.Chroma). Used by tests.
    """

    def __init__(
        self,
        embedding_function: Any = None,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        store: Any = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries

        # In-memory exact-match fast path: (normalised question, doc
        # version) -> hit. Rebuilt naturally as questions arrive; a
        # restart just means the first repeat goes through vector search
        # (distance ~0 for an identical question) instead.
        self._exact: Dict[Tuple[str, str], CacheHit] = {}

        # Session stats (in-memory only; reset on restart).
        self.hits = 0
        self.misses = 0
        self._hit_latencies: List[float] = []
        self._miss_latencies: List[float] = []
        self._approx_tokens_saved = 0  # rough chars/4 estimate, see below

        if store is not None:
            self.store = store
        else:
            if embedding_function is None:
                # Default: free local embeddings (no API key, no credits).
                embedding_function = init_cache_embeddings()
            # Lazy import: keeps this module importable without LangChain.
            from langchain_community.vectorstores import Chroma

            self.store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=persist_directory,
            )

    # -- lookup ----------------------------------------------------------

    def lookup(self, question: str, doc_version: str) -> Optional[CacheHit]:
        """Return a cached answer for `question`, or None on a miss.

        Only entries tagged with the same `doc_version` are considered,
        so answers from a previous set of documents are never served.
        """
        started = time.perf_counter()
        norm = normalize_question(question)
        key = (norm, doc_version)

        # Fast path: exact repeat, no embedding call needed.
        if key in self._exact:
            hit = self._exact[key]
            self._record_hit(hit, time.perf_counter() - started)
            return hit

        try:
            # The where-filter runs inside ChromaDB, so the top result is
            # the most similar question *for these documents*.
            results = self.store.similarity_search_with_score(
                question, k=1, filter={"doc_version": doc_version}
            )
        except Exception as exc:  # fail-open: a cache error must not break RAG
            print(f"Semantic cache lookup failed ({exc}); falling back to RAG.")
            self.misses += 1
            return None

        if not results:
            self.misses += 1
            return None

        doc, distance = results[0]
        # ChromaDB returns cosine DISTANCE (0 = identical). Convert it.
        similarity = 1.0 - float(distance)
        if similarity < self.similarity_threshold:
            self.misses += 1
            return None

        hit = CacheHit(
            answer=doc.metadata["answer"],
            similarity=round(similarity, 3),
            matched_question=doc.page_content,
            source="semantic",
        )
        self._exact[key] = hit
        self._record_hit(hit, time.perf_counter() - started)
        return hit

    # -- write -----------------------------------------------------------

    def add(self, question: str, answer: str, doc_version: str) -> bool:
        """Store a freshly generated Q&A pair. Returns False if skipped.

        Skipped when the answer is empty, a refusal, or a backend error,
        or when the exact question is already cached for these documents.
        """
        if not answer or not answer.strip():
            return False
        lowered = answer.strip().lower()
        if any(marker in lowered for marker in REFUSAL_MARKERS):
            return False

        norm = normalize_question(question)
        if (norm, doc_version) in self._exact:
            return False

        self._evict_if_full()
        metadata = {
            "answer": answer,
            "doc_version": doc_version,
            "question": question,
            "created_at": time.time(),  # unix timestamp (float) for FIFO eviction
        }
        try:
            self.store.add_texts(
                texts=[question],
                metadatas=[metadata],
                ids=[f"cache-{uuid.uuid4().hex}"],
            )
        except Exception as exc:  # fail-open
            print(f"Semantic cache write failed ({exc}); answer not cached.")
            return False
        self._persist()
        self._exact[(norm, doc_version)] = CacheHit(
            answer=answer,
            similarity=1.0,
            matched_question=question,
            source="exact",
        )
        return True

    # -- caller-reported timing ------------------------------------------

    def record_miss_latency(self, seconds: float) -> None:
        """Record how long the full RAG pipeline took for a cache miss.

        Called by the Streamlit app after query_with_full_context runs,
        so the dashboard can compare hit latency vs miss latency.
        """
        self._miss_latencies.append(seconds)

    # -- maintenance ------------------------------------------------------

    def clear(self, doc_version: Optional[str] = None) -> int:
        """Delete cached entries; all of them, or only one doc version.

        Returns the number of entries deleted (0 if the store could not
        be listed). Also clears the in-memory exact-match index.
        """
        self._exact = {
            key: hit
            for key, hit in self._exact.items()
            if doc_version is not None and key[1] != doc_version
        } if doc_version is not None else {}
        try:
            where = {"doc_version": doc_version} if doc_version else None
            data = self.store.get(where=where) if where else self.store.get()
            ids = data.get("ids", [])
            if ids:
                self.store.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            print(f"Semantic cache clear failed ({exc}).")
            return 0

    def summary(self) -> Dict[str, Any]:
        """Session stats for the Streamlit sidebar dashboard."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "entries": self._entry_count(),
            "avg_hit_latency_ms": self._avg_ms(self._hit_latencies),
            "avg_miss_latency_ms": self._avg_ms(self._miss_latencies),
            # Rough estimate only: cached answers are ~len/4 tokens that
            # did not need generating. Not a billing figure.
            "approx_output_tokens_saved": self._approx_tokens_saved,
        }

    # -- internals --------------------------------------------------------

    def _record_hit(self, hit: CacheHit, latency: float) -> None:
        self.hits += 1
        self._hit_latencies.append(latency)
        # Output tokens the LLM did not have to generate this time.
        self._approx_tokens_saved += max(1, len(hit.answer) // 4)

    def _evict_if_full(self) -> None:
        """FIFO eviction: delete the oldest entry when at capacity."""
        try:
            data = self.store.get()
            ids = data.get("ids", [])
            if len(ids) < self.max_entries:
                return
            metadatas = data.get("metadatas", [])
            oldest = min(
                range(len(ids)),
                key=lambda i: metadatas[i].get("created_at", 0),
            )
            self.store.delete(ids=[ids[oldest]])
            # Keep the in-memory exact-match index consistent, or an
            # evicted entry would still be served from memory.
            evicted = metadatas[oldest]
            self._exact.pop(
                (normalize_question(evicted.get("question", "")),
                 evicted.get("doc_version", "")),
                None,
            )
        except Exception as exc:
            # Safe failure: the cache just grows past max_entries.
            print(f"Cache eviction skipped ({exc}).")

    def _entry_count(self) -> Optional[int]:
        try:
            return len(self.store.get().get("ids", []))
        except Exception:
            return None  # unknown; the dashboard shows a dash

    def _persist(self) -> None:
        """Persist the collection. Newer ChromaDB auto-persists and the
        call is a no-op; older versions need it. Either way, never crash."""
        try:
            self.store.persist()
        except (AttributeError, TypeError):
            pass
        except Exception as exc:
            print(f"Cache persist warning ({exc}).")

    @staticmethod
    def _avg_ms(latencies: List[float]) -> Optional[float]:
        if not latencies:
            return None
        return round(sum(latencies) / len(latencies) * 1000, 1)
