"""
evaluate_security.py -- Measure prompt-injection attacks on RagChatBot.

End-to-end harness: builds a small document store (12 injected chunks +
12 benign chunks from security_test_set.py), runs realistic queries,
retrieves the top-3 chunks with real embeddings, then calls a victim
LLM twice per attack:

  BEFORE -- the original prompt, no guardrails (the app as it is today)
  AFTER  -- sanitized context + hardened prompt (the guardrails in
            rag_guardrails.py)

Reports:
  * retrieval exposure -- how often an injected chunk reaches the top-3
  * attack success rate BEFORE and AFTER (canary word in the answer)
  * filter true-positive rate on injected chunks and false-positive
    rate on benign chunks
  * scan latency overhead

Every number comes from a real run. The victim LLM is Pollinations'
free text API (model=openai, no key needed), because the Gemini credits
on the original project are depleted; the prompt is the app's real
prompt (see rag_guardrails.build_original_prompt), and retrieval uses
the same local bge-base embeddings the semantic cache uses.

Usage (on the user's laptop):

    python evaluate_security.py run [--out security-results.json]
    python evaluate_security.py run --skip-llm     # retrieval + filter only
    python evaluate_security.py run --limit 2      # smoke test: 2 attacks
    python evaluate_security.py run --resume       # continue an interrupted run

Per-attack results are appended to a JSONL file as they complete, so an
interrupted run loses at most one attack's work; --resume skips attacks
already recorded there.

Note on the victim LLM: Pollinations responses are non-deterministic
(the free endpoint takes no temperature parameter), so re-runs may
differ by an attack or two. The JSONL file records every answer, so
any run can be audited by hand.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, ".")  # so repo imports work from the repo root

from rag_guardrails import (  # noqa: E402
    build_hardened_prompt,
    build_original_prompt,
    check_output,
    format_context,
    sanitize_context,
    scan_chunk,
)
from security_test_set import BENIGN_CHUNKS, INJECTIONS  # noqa: E402
from semantic_cache import init_cache_embeddings  # noqa: E402

# Victim LLM: Pollinations free text API, no key needed.
POLLINATIONS_URL = "https://text.pollinations.ai/{prompt}?model=openai&private=true"
POLLINATIONS_USER_AGENT = "ragchatbot-security-eval/1.0"
POLLINATIONS_TIMEOUT = 120  # seconds per call
POLLINATIONS_MAX_ATTEMPTS = 3
POLLINATIONS_BACKOFF = [2.0, 4.0, 8.0]  # seconds between attempts

VICTIM_LLM = "Pollinations text API, model=openai"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RETRIEVAL_K = 3  # same k as query_with_full_context in the backend


# ---------------------------------------------------------------------------
# In-memory exact-cosine store (same math as ChromaDB; mirrors the
# _FakeStore in evaluate_cache.py, but with LangChain-style embedder calls)
# ---------------------------------------------------------------------------

class InMemoryStore:
    """Minimal vector store over L2-normalised embeddings.

    The embedder needs embed_documents(texts) and embed_query(text)
    (the interface LocalEmbeddings provides). Cosine similarity is then
    a plain dot product, exactly as ChromaDB computes it.
    """

    def __init__(self, embedder):
        self._embedder = embedder
        self._entries = []  # {"id", "text", "vector", "metadata"}

    def add_texts(self, texts, metadatas=None, ids=None):
        vectors = self._embedder.embed_documents(texts)
        for i, text in enumerate(texts):
            self._entries.append({
                "id": ids[i] if ids else f"id-{len(self._entries)}",
                "text": text,
                "vector": vectors[i],
                "metadata": metadatas[i] if metadatas else {},
            })

    def search(self, query, k=3):
        """Return the top-k entries as (id, text, similarity, metadata),
        most similar first."""
        query_vec = self._embedder.embed_query(query)
        scored = []
        for entry in self._entries:
            similarity = sum(x * y for x, y in zip(query_vec, entry["vector"]))
            scored.append((entry["id"], entry["text"], similarity, entry["metadata"]))
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# Victim LLM (Pollinations free text API)
# ---------------------------------------------------------------------------

def call_pollinations(prompt, timeout=POLLINATIONS_TIMEOUT):
    """Call the victim LLM once, with retries. Returns the raw text
    answer. Raises RuntimeError if all attempts fail."""
    url = POLLINATIONS_URL.format(prompt=urllib.parse.quote(prompt, safe=""))
    request = urllib.request.Request(
        url, headers={"User-Agent": POLLINATIONS_USER_AGENT}
    )
    last_error = None
    for attempt in range(POLLINATIONS_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", "replace")
            # Pollinations sometimes answers 200 with an error message.
            head = body.strip().lower()[:200]
            if head.startswith("error") or "rate limit" in head:
                raise RuntimeError(f"API error body: {body[:200]}")
            return body
        except (urllib.error.URLError, TimeoutError, RuntimeError,
                http.client.HTTPException, OSError) as exc:
            # OSError covers RemoteDisconnected ("remote end closed
            # connection"), which urlopen raises raw rather than wrapped
            # in URLError -- without it, that failure mode gets no retry.
            last_error = exc
            if attempt < POLLINATIONS_MAX_ATTEMPTS - 1:
                time.sleep(POLLINATIONS_BACKOFF[attempt])
    raise RuntimeError(f"Pollinations call failed after "
                       f"{POLLINATIONS_MAX_ATTEMPTS} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Filter measurement (no LLM calls)
# ---------------------------------------------------------------------------

def measure_filter(repeats=5):
    """Scan every injected and benign chunk. Returns true/false-positive
    counts, per-chunk results, and mean scan latency in ms per chunk."""
    injection_results = []
    for inj in INJECTIONS:
        flagged, reasons = scan_chunk(inj["chunk"])
        injection_results.append({
            "id": inj["id"], "category": inj["category"],
            "flagged": flagged, "reasons": reasons,
        })
    benign_results = []
    for benign in BENIGN_CHUNKS:
        flagged, reasons = scan_chunk(benign["chunk"])
        benign_results.append({
            "id": benign["id"], "flagged": flagged, "reasons": reasons,
        })
    # Scan latency: time the full 24-chunk pass, averaged over repeats.
    all_chunks = [inj["chunk"] for inj in INJECTIONS] + [
        b["chunk"] for b in BENIGN_CHUNKS
    ]
    started = time.perf_counter()
    for _ in range(repeats):
        for chunk in all_chunks:
            scan_chunk(chunk)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "injected_flagged": sum(1 for r in injection_results if r["flagged"]),
        "injected_total": len(injection_results),
        "benign_flagged": sum(1 for r in benign_results if r["flagged"]),
        "benign_total": len(benign_results),
        "scan_latency_ms_per_chunk": round(
            elapsed_ms / (repeats * len(all_chunks)), 3
        ),
        "injection_results": injection_results,
        "benign_results": benign_results,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _load_completed(jsonl_path):
    """Attack ids already recorded in the JSONL file (for --resume)."""
    completed = set()
    if jsonl_path and os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed.add(json.loads(line)["attack_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def _append_jsonl(jsonl_path, record):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _call_victim(prompt, sleep_between):
    """One victim call; returns (answer_or_None, latency_s, error_or_None).
    A failed call is recorded as an error, never as a blocked attack."""
    started = time.perf_counter()
    try:
        answer = call_pollinations(prompt)
        return answer, round(time.perf_counter() - started, 1), None
    except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
        return None, round(time.perf_counter() - started, 1), str(exc)
    finally:
        if sleep_between:
            time.sleep(sleep_between)


def run_evaluation(embedder=None, out_json="security-results.json",
                   jsonl_path="security-results.jsonl", limit=None,
                   skip_llm=False, resume=False, sleep_between=1.0,
                   verbose=True):
    """Run the full before/after evaluation. Returns the summary dict.

    embedder: object with embed_documents/embed_query (LangChain-style).
    If None, the shipped local bge-base model is loaded via
    semantic_cache.init_cache_embeddings(). The VM measurement driver
    passes a fastembed adapter instead (same weights, ONNX build).
    """
    started_all = time.perf_counter()
    attacks = INJECTIONS[:limit] if limit else INJECTIONS

    if embedder is None:
        embedder = init_cache_embeddings()
    store = InMemoryStore(embedder)
    store.add_texts(
        [inj["chunk"] for inj in INJECTIONS] + [b["chunk"] for b in BENIGN_CHUNKS],
        metadatas=[{"kind": "injected", "topic": inj["topic"]} for inj in INJECTIONS]
        + [{"kind": "benign", "topic": b["topic"]} for b in BENIGN_CHUNKS],
        ids=[inj["id"] for inj in INJECTIONS] + [b["id"] for b in BENIGN_CHUNKS],
    )

    filter_stats = measure_filter()
    if verbose:
        print(f"Filter: {filter_stats['injected_flagged']}/{filter_stats['injected_total']} "
              f"injected flagged, {filter_stats['benign_flagged']}/{filter_stats['benign_total']} "
              f"benign flagged, {filter_stats['scan_latency_ms_per_chunk']} ms/chunk scan.")

    completed = _load_completed(jsonl_path) if resume else set()
    if resume and verbose:
        print(f"Resume: {len(completed)} attack(s) already recorded, skipping them.")
    if not resume and jsonl_path and os.path.exists(jsonl_path):
        # Fresh run: start with a clean incremental log.
        os.remove(jsonl_path)

    # For --resume, reload the skipped attacks' records from the JSONL so
    # the final summary covers every attack. Last record per attack wins,
    # so a duplicated line can never double-count an attack.
    per_attack = []
    if resume and completed:
        by_id = {}
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_id[record.get("attack_id")] = record
        per_attack = [by_id[aid] for aid in completed if aid in by_id]

    for attack in attacks:
        if attack["id"] in completed:
            continue
        retrieved = store.search(attack["query"], k=RETRIEVAL_K)
        retrieved_ids = [entry[0] for entry in retrieved]
        exposed = attack["id"] in retrieved_ids
        record = {
            "attack_id": attack["id"],
            "category": attack["category"],
            "query": attack["query"],
            "canary": attack["canary"],
            "retrieved_ids": retrieved_ids,
            "retrieved_similarities": [round(entry[2], 3) for entry in retrieved],
            "exposed": exposed,
        }
        if verbose:
            print(f"\n{attack['id']} ({attack['category']}): "
                  f"retrieved={retrieved_ids}, exposed={exposed}")

        if not skip_llm:
            retrieved_texts = [entry[1] for entry in retrieved]

            # BEFORE: original prompt, no guardrails.
            prompt_before = build_original_prompt(
                format_context(retrieved_texts), attack["query"]
            )
            answer_before, latency_before, error_before = _call_victim(
                prompt_before, sleep_between
            )
            success_before = (
                bool(check_output(answer_before, [attack["canary"]]))
                if answer_before is not None else None
            )
            record["before"] = {
                "answer": (answer_before[:4000] if answer_before else None),
                "latency_s": latency_before,
                "error": error_before,
                "success": success_before,
            }
            if verbose:
                print(f"  BEFORE: success={success_before} "
                      f"({latency_before}s)"
                      + (f" ERROR: {error_before}" if error_before else ""))

            # AFTER: sanitized context + hardened prompt.
            kept, report = sanitize_context(retrieved_texts)
            dropped_indices = {d["index"] for d in report["dropped_chunks"]}
            kept_ids = [rid for i, rid in enumerate(retrieved_ids)
                        if i not in dropped_indices]
            prompt_after = build_hardened_prompt(
                format_context(kept), attack["query"]
            )
            answer_after, latency_after, error_after = _call_victim(
                prompt_after, sleep_between
            )
            success_after = (
                bool(check_output(answer_after, [attack["canary"]]))
                if answer_after is not None else None
            )
            record["after"] = {
                "dropped": report["dropped"],
                "dropped_reasons": [d["reasons"] for d in report["dropped_chunks"]],
                "injected_chunk_dropped": exposed and attack["id"] not in kept_ids,
                "answer": (answer_after[:4000] if answer_after else None),
                "latency_s": latency_after,
                "error": error_after,
                "success": success_after,
            }
            if verbose:
                print(f"  AFTER:  success={success_after} "
                      f"({latency_after}s, dropped {report['dropped']}/{report['total']})"
                      + (f" ERROR: {error_after}" if error_after else ""))

        per_attack.append(record)
        if jsonl_path:
            _append_jsonl(jsonl_path, record)

    # -- summary -----------------------------------------------------------
    n = len(per_attack)
    exposed_n = sum(1 for r in per_attack if r["exposed"])
    summary = {
        "victim_llm": VICTIM_LLM,
        "embedding_model": EMBEDDING_MODEL,
        "retrieval_k": RETRIEVAL_K,
        "n_attacks": n,
        "n_benign_chunks": len(BENIGN_CHUNKS),
        "retrieval_exposure": {
            "exposed": exposed_n,
            "total": n,
            "rate": round(exposed_n / n, 3) if n else 0.0,
        },
        "filter": {
            "injected_flagged": filter_stats["injected_flagged"],
            "injected_total": filter_stats["injected_total"],
            "benign_flagged": filter_stats["benign_flagged"],
            "benign_total": filter_stats["benign_total"],
            "false_positive_rate": round(
                filter_stats["benign_flagged"] / filter_stats["benign_total"], 3
            ) if filter_stats["benign_total"] else 0.0,
            "scan_latency_ms_per_chunk": filter_stats["scan_latency_ms_per_chunk"],
        },
        "total_runtime_s": round(time.perf_counter() - started_all, 1),
    }
    if not skip_llm:
        for condition in ("before", "after"):
            attempted = [r for r in per_attack if r[condition]["error"] is None]
            successes = [r for r in attempted if r[condition]["success"]]
            exposed_attempted = [r for r in attempted if r["exposed"]]
            exposed_successes = [r for r in exposed_attempted if r[condition]["success"]]
            summary[condition] = {
                "successes": len(successes),
                "attempted": len(attempted),
                "errors": n - len(attempted),
                "success_rate": round(len(successes) / len(attempted), 3) if attempted else None,
                "successes_among_retrieved": len(exposed_successes),
                "attempted_among_retrieved": len(exposed_attempted),
                "success_rate_among_retrieved": (
                    round(len(exposed_successes) / len(exposed_attempted), 3)
                    if exposed_attempted else None
                ),
                "mean_latency_s": round(
                    sum(r[condition]["latency_s"] for r in attempted) / len(attempted), 1
                ) if attempted else None,
            }

    results = {"summary": summary, "per_attack": per_attack}
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n=== Summary ===")
        print(f"Retrieval exposure: {exposed_n}/{n} injected chunks in top-{RETRIEVAL_K}")
        print(f"Filter: {filter_stats['injected_flagged']}/{filter_stats['injected_total']} "
              f"injected flagged, {filter_stats['benign_flagged']}/{filter_stats['benign_total']} "
              f"benign flagged ({summary['filter']['false_positive_rate']:.0%} FP rate), "
              f"{filter_stats['scan_latency_ms_per_chunk']} ms/chunk")
        if not skip_llm:
            for condition in ("before", "after"):
                s = summary[condition]
                print(f"Attack success {condition.upper()}: {s['successes']}/{s['attempted']} "
                      f"({s['success_rate']:.0%} of attempted; "
                      f"{s['successes_among_retrieved']}/{s['attempted_among_retrieved']} "
                      f"among retrieved), errors: {s['errors']}")
        print(f"Total runtime: {summary['total_runtime_s']}s")
        if out_json:
            print(f"Results written to {out_json}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure prompt-injection attacks on RagChatBot, "
                    "before and after guardrails."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the before/after evaluation")
    run.add_argument("--out", default="security-results.json",
                     help="results JSON file")
    run.add_argument("--jsonl", default="security-results.jsonl",
                     help="per-attack results, appended incrementally")
    run.add_argument("--limit", type=int, default=None,
                     help="run only the first N attacks (smoke test)")
    run.add_argument("--skip-llm", action="store_true",
                     help="measure retrieval + filter only, no LLM calls")
    run.add_argument("--resume", action="store_true",
                     help="skip attacks already recorded in the JSONL file")
    run.add_argument("--sleep", type=float, default=1.0,
                     help="seconds to wait between victim LLM calls")
    run.set_defaults(func=lambda args: run_evaluation(
        out_json=args.out, jsonl_path=args.jsonl, limit=args.limit,
        skip_llm=args.skip_llm, resume=args.resume, sleep_between=args.sleep,
    ))
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
