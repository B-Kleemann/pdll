import logging
import logging.config
import os
from datetime import datetime

import pandas as pd

logging.config.fileConfig("pdll\\\\log\\_logging.conf")
logger = logging.getLogger("result")

CACHE_PATH_BL = "pdll\\caching\\baseline_cache.parquet"
CACHE_PATH_PW = "pdll\\caching\\pairwise_cache.parquet"

cache_BL = pd.DataFrame()
cache_PW = pd.DataFrame()

cache_stats_BL = {"hits": 0, "misses": 0}
cache_stats_PW = {"hits": 0, "misses": 0}


def load_cache(is_pairwise: bool):
    if is_pairwise:
        cache_path = CACHE_PATH_PW
    else:
        cache_path = CACHE_PATH_BL

    if os.path.exists(cache_path):
        read_cache_file(cache_path, is_pairwise)
    else:
        initialize_new_cache(is_pairwise)
    logger.debug("loaded cache")


def read_cache_file(cache_path, is_pairwise: bool):
    cache_df = pd.read_parquet(cache_path)
    global cache_PW, cache_BL
    if is_pairwise:
        cache_PW = cache_df
    else:
        cache_BL = cache_df
    logger.debug("read cache file")


def initialize_new_cache(is_pairwise: bool):
    cache_df = pd.DataFrame(columns=["model", "prompt", "score", "timestamp"])
    global cache_PW, cache_BL
    if is_pairwise:
        cache_PW = cache_df
    else:
        cache_BL = cache_df
    logger.debug("initialized new cache")


def save_cache(cache: pd.DataFrame, is_pairwise: bool):
    if is_pairwise:
        cache_path = CACHE_PATH_PW
    else:
        cache_path = CACHE_PATH_BL
    cache.to_parquet(cache_path, index=False)
    logger.debug("saved cache file")


def lookup_in_cache(model: str, prompt: str, is_pairwise: bool):
    logger.debug("looked-up prompt in cache")
    if is_pairwise:
        global cache_PW, cache_stats_PW
        cached_row = cache_PW[
            (cache_PW["model"] == model) & (cache_PW["prompt"] == prompt)
        ]
        if not cached_row.empty:
            cache_stats_PW["hits"] += 1
            return cached_row.iloc[0]["score"]
        cache_stats_PW["misses"] += 1

    else:
        global cache_BL, cache_stats_BL
        cached_row = cache_BL[
            (cache_BL["model"] == model) & (cache_BL["prompt"] == prompt)
        ]
        if not cached_row.empty:
            cache_stats_BL["hits"] += 1
            return cached_row.iloc[0]["score"]
        cache_stats_BL["misses"] += 1


def new_cache_entry(model: str, prompt: str, score, is_pairwise: bool):
    timestamp = datetime.now().isoformat()
    if is_pairwise:
        global cache_PW
        new_entry = pd.DataFrame(
            [[model, prompt, score, timestamp]],
            columns=["model", "prompt", "score", "timestamp"],
        )
        cache_PW = pd.concat([cache_PW, new_entry], ignore_index=True)
        save_cache(cache_PW, is_pairwise)
    else:
        global cache_BL
        new_entry = pd.DataFrame(
            [[model, prompt, score, timestamp]],
            columns=["model", "prompt", "score", "timestamp"],
        )
        cache_BL = pd.concat([cache_BL, new_entry], ignore_index=True)
        save_cache(cache_BL, is_pairwise)
    logger.debug("created new cache entry")


def print_cache_stats(is_pairwise: bool):
    if is_pairwise:
        cache_stats = cache_stats_PW
        logger.info("Pairwise Cache-Stats:")
    else:
        cache_stats = cache_stats_BL
        logger.info("Baseline Cache-Stats:")
    logger.info(f"Cache hits: {cache_stats['hits']}")
    logger.info(f"Cache misses: {cache_stats['misses']}")
    logger.debug("printed cache stats\n")
