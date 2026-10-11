"""Where the process's memory is, for measuring under load. Off unless
RECOMMENDER_MEMORY_REPORT=1: it stops the process for a moment to count."""

import contextlib
import ctypes
import gc
import pickle
from pathlib import Path

from . import between_requests, labels
from .embeddings import default_embedder
from .kept import Kept


def _resident() -> dict[str, int]:
    """What the system counts for the process, in megabytes."""
    wanted = {"VmRSS": "resident", "RssAnon": "own", "RssFile": "files"}
    found: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in wanted:
                found[wanted[name]] = int(value.split()[0]) // 1024
    except OSError:
        pass
    return found


def _megabytes(thing: object) -> float:
    try:
        return round(len(pickle.dumps(thing, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6, 1)
    except Exception:  # noqa: BLE001 - a report must not fail on what it cannot weigh
        return -1.0


def _between_requests() -> dict:
    """The histories and plans held for pages in progress: how many, and the watches and
    likes they hold (what their limits count)."""
    with between_requests._histories_lock:
        histories = [len(h) + len(f) for _, h, f in between_requests._histories.values()]
    with between_requests._plans_lock:
        kept_plans = list(between_requests._plans.values())
    plans = [between_requests._items(snapshot) for _, snapshot in kept_plans]
    return {
        "histories": len(histories),
        "history_items": sum(histories),
        "plans": len(plans),
        "plan_items": sum(plans),
        "parts": len(between_requests._parts),
    }


def report() -> dict:
    before = _resident()
    vectors = getattr(default_embedder(), "_cache", {})
    kept = [
        {"results": len(store), "megabytes": [_megabytes(result) for result in store.results()]}
        for store in Kept.all()
    ]
    found = {
        "before": before,
        "title_vectors": {
            "entries": len(vectors),
            "megabytes": round(sum(v.nbytes for v in list(vectors.values())) / 1e6, 1),
        },
        "title_texts": labels.topic_text.cache_info().currsize,
        "title_informative": labels.informative.cache_info().currsize,
        "kept": kept,
        "between_requests": _between_requests(),
        "objects": len(gc.get_objects()),
    }
    # What the process holds but no longer uses: collected, then handed back to the system.
    found["collected"] = gc.collect()
    found["after_collect"] = _resident()
    with contextlib.suppress(OSError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    found["after_trim"] = _resident()
    return found
