import os
from datetime import datetime

from anyio import Path
import pandas as pd

CACHE_PATH_BASELINE = "pdll\\baseline_cache.parquet"
CACHE_PATH_PAIRWISE = "pdll\\pairwise_cache.parquet"

cache_baseline = pd.DataFrame()
cache_pairwise = pd.DataFrame()

cache_stats_baseline = {"hits": 0, "misses": 0}
cache_stats_pairwise = {"hits": 0, "misses": 0}


def load_cache(is_pairwise: bool):
    if is_pairwise:
        cache_path = CACHE_PATH_PAIRWISE
    else:
        cache_path = CACHE_PATH_BASELINE

    if os.path.exists(cache_path):
        read_cache_file(cache_path, is_pairwise)
    else:
        initialize_new_cache(is_pairwise)


def read_cache_file(cache_path, is_pairwise: bool):
    cache_df = pd.read_parquet(cache_path)
    global cache_pairwise, cache_baseline
    if is_pairwise:
        cache_pairwise = cache_df
    else:
        cache_baseline = cache_df


def initialize_new_cache(is_pairwise: bool):
    cache_df = pd.DataFrame(columns=["prompt", "score", "timestamp"])
    global cache_pairwise, cache_baseline
    if is_pairwise:
        cache_pairwise = cache_df
    else:
        cache_baseline = cache_df


def save_cache(cache: pd.DataFrame, is_pairwise: bool):
    if is_pairwise:
        cache_path = CACHE_PATH_PAIRWISE
    else:
        cache_path = CACHE_PATH_BASELINE
    cache.to_parquet(cache_path, index=False)


def lookup_in_cache(prompt: str, is_pairwise: bool):
    if is_pairwise:
        global cache_pairwise, cache_stats_pairwise
        cached_row = cache_pairwise[cache_pairwise["prompt"] == prompt]
        if not cached_row.empty:
            cache_stats_pairwise["hits"] += 1
            return cached_row.iloc[0]["score"]
        cache_stats_pairwise["misses"] += 1

    else:
        global cache_baseline, cache_stats_baseline
        cached_row = cache_baseline[cache_baseline["prompt"] == prompt]
        if not cached_row.empty:
            cache_stats_baseline["hits"] += 1
            return cached_row.iloc[0]["score"]
        cache_stats_baseline["misses"] += 1


def new_cache_entry(prompt: str, score, is_pairwise: bool):
    timestamp = datetime.now().isoformat()
    if is_pairwise:
        global cache_pairwise
        new_entry = pd.DataFrame(
            [[prompt, score, timestamp]], columns=["prompt", "score", "timestamp"]
        )
        cache_pairwise = pd.concat([cache_pairwise, new_entry], ignore_index=True)
        save_cache(cache_pairwise, is_pairwise)
    else:
        global cache_baseline
        new_entry = pd.DataFrame(
            [[prompt, score, timestamp]], columns=["prompt", "score", "timestamp"]
        )
        cache_baseline = pd.concat([cache_baseline, new_entry], ignore_index=True)
        save_cache(cache_baseline, is_pairwise)


def print_cache_stats(is_pairwise: bool):
    if is_pairwise:
        cache_stats = cache_stats_pairwise
        print("Pairwise:")
    else:
        cache_stats = cache_stats_baseline
        print("Baseline:")

    print(f"Cache hits: {cache_stats['hits']}")
    print(f"Cache misses: {cache_stats['misses']}")
