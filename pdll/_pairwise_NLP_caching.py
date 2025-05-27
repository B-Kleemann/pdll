import os
from datetime import datetime

from anyio import Path
import pandas as pd

CACHE_PATH_BASELINE = "baseline_cache.parquet"
CACHE_PATH_PAIRWISE = "pairwise_cache.parquet"

cache_baseline = pd.DataFrame()
cache_pairwise = pd.DataFrame()

cache_stats_baseline = {"hits": 0, "misses": 0}
cache_stats_pairwise = {"hits": 0, "misses": 0}


def pairwise_or_baseline(is_pairwise: bool):
    if is_pairwise:
        cache_path = CACHE_PATH_PAIRWISE
        cache = cache_pairwise
        cache_stats = cache_stats_pairwise
    else:
        cache_path = CACHE_PATH_BASELINE
        cache = cache_baseline
        cache_stats = cache_stats_baseline
    return cache_path, cache, cache_stats


def load_cache(is_pairwise: bool):
    cache_path, cache, _ = pairwise_or_baseline(is_pairwise)
    if os.path.exists(cache_path):
        read_cache_file(cache_path, is_pairwise)
    else:
        initialize_new_cache(is_pairwise)


def initialize_new_cache(is_pairwise: bool):
    cache_df = pd.DataFrame(columns=["prompt", "score", "timestamp"])
    global cache_pairwise, cache_baseline
    if is_pairwise:
        cache_pairwise = cache_df
    else:
        cache_baseline = cache_df


def read_cache_file(cache_path, is_pairwise: bool):
    cache_df = pd.read_parquet(cache_path)
    global cache_pairwise, cache_baseline
    if is_pairwise:
        cache_pairwise = cache_df
    else:
        cache_baseline = cache_df


def save_cache(cache: pd.DataFrame, is_pairwise: bool):
    cache_path, _, _ = pairwise_or_baseline(is_pairwise)
    cache.to_parquet(cache_path, index=False)


def lookup_in_cache(prompt: str, is_pairwise: bool):
    _, cache, cache_stats = pairwise_or_baseline(is_pairwise)

    cached_row = cache[cache["prompt"] == prompt]
    if not cached_row.empty:
        cache_stats["hits"] += 1
        return cached_row.iloc[0]["score"]

    cache_stats["misses"] += 1


def new_cache_entry(prompt: str, score, is_pairwise: bool):
    _, cache, _ = pairwise_or_baseline(is_pairwise)

    timestamp = datetime.utcnow().isoformat()
    new_entry = pd.DataFrame(
        [[prompt, score, timestamp]], columns=["prompt", "score", "timestamp"]
    )
    cache = pd.concat([cache, new_entry], ignore_index=True)

    save_cache(cache, is_pairwise)


def print_cache_stats(is_pairwise: bool):
    _, _, cache_stats = pairwise_or_baseline(is_pairwise)
    if is_pairwise:
        print("Pairwise:")
    else:
        print("Baseline:")
    print(f"Cache hits: {cache_stats['hits']}")
    print(f"Cache misses: {cache_stats['misses']}")
