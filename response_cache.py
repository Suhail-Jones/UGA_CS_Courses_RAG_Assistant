import json
import hashlib
import os
from pathlib import Path

CACHE_PATH = Path("cache.json")


def _load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _key_for(question: str):
    # Hash the question so cache.json keys are short and filesystem-safe
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def cached_query_rag(question: str, real_query_rag_fn, force_refresh: bool = False):
    cache = _load_cache()
    key = _key_for(question)

    if not force_refresh and key in cache:
        print(f"[cache hit] {question[:60]}...")
        return cache[key]["response"]

    print(f"[cache MISS - calling agent] {question[:60]}...")
    response = real_query_rag_fn(question)

    cache[key] = {
        "question": question,
        "response": response,
    }
    _save_cache(cache)
    return response
