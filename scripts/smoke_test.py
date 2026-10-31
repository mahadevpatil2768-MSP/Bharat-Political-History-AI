"""End-to-end smoke test against a running API (uvicorn app.api.main:app).

    python -m scripts.smoke_test           # no Gemini calls: health, search, no-key/allowlist path, decline
    python -m scripts.smoke_test --live    # + exactly ONE Gemini generation request, after a y/N prompt

Uses API_BASE_URL or http://127.0.0.1:8000. Exits non-zero on any failed check.
The live request uses one request from the Gemini free-tier quota; it never enables billing.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LIVE_QUESTION = "What does Article 17 of the Constitution say about untouchability?"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def post(path: str, payload: dict) -> dict:
    r = requests.post(f"{API}{path}", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="make exactly one real Gemini request (asks first)")
    args = ap.parse_args(argv)

    try:
        health = requests.get(f"{API}/health", timeout=30).json()
    except requests.ConnectionError:
        print(f"Cannot reach {API}. Start: uvicorn app.api.main:app --host 127.0.0.1 --port 8000")
        return 2
    check("health ok", health.get("status") == "ok", str(health))
    check("index not empty", health.get("indexed_chunks", 0) > 0, f"{health.get('indexed_chunks')} chunks")
    check("provider is gemini", health.get("llm_provider") == "gemini", str(health.get("llm_model")))

    hits = post("/search", {"query": "Proclamation of Emergency Article 352", "top_k": 5})["hits"]
    check("search returns constitution passages", bool(hits) and hits[0]["citation"]["document_id"]
          == "constitution_of_india_2024", hits[0]["citation"]["title"] if hits else "no hits")

    # Declined before retrieval or any LLM call.
    res = post("/query", {"question": "Which party will win the next Lok Sabha election?"})
    check("declines election prediction (no LLM call)", res["status"] == "declined")

    if not health.get("llm_configured"):
        res = post("/query", {"question": LIVE_QUESTION})  # no key → no request is sent
        check("no-key path returns evidence with clear message",
              res["status"] == "error" and "GEMINI_API_KEY" in res["answer"] and bool(res["retrieved"]))
        print("\nGEMINI_API_KEY not configured — add it to .env and restart the API to test generation.")
    elif not health.get("llm_model_free_tier_allowlisted"):
        check("model on free-tier allowlist", False, str(health.get("llm_model")))
    elif args.live:
        print(f"\nAbout to send ONE request to the Gemini API ({health.get('llm_model')}, free tier).")
        print(f"Question: {LIVE_QUESTION}")
        print("It sends the question and ~8 retrieved passages from public-domain/government texts to Google.")
        if input("Proceed? [y/N] ").strip().lower() != "y":
            print("Live request skipped.")
        else:
            res = post("/query", {"question": LIVE_QUESTION})
            retrieved = {c["chunk_id"] for c in res["retrieved"]}
            check("grounded answer", res["status"] == "answered", res["answer"][:300])
            check("citations present", bool(res["citations"]), ", ".join(c["source_id"] for c in res["citations"]))
            check("every citation maps to a retrieved chunk",
                  all(c["chunk_id"] in retrieved for c in res["citations"]))
            check("citations carry page references", all(c.get("page_start") for c in res["citations"]),
                  "; ".join(f"{c['source_id']} p.{c.get('page_start')}" for c in res["citations"]))
            for n in res.get("notices", []):
                print(f"  notice: {n}")
            print("\nAnswer:\n" + res["answer"] + "\n")
    else:
        print("\nKey configured. Skipping the live Gemini request; run with --live to send exactly one.")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
