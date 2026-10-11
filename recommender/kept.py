"""What was worked out for a person, kept while their history has not changed.

A page's plan and rank already share one snapshot. Beyond that, every page of the same person
(another refresh, the "what the algorithm kept of me" page) used to read the whole history
again. When the history is named by a key, what was concluded from it is kept under that key
together with everything else it was read with (settings, subscriptions, corrections, time
zone), for RECOMMENDER_PROFILE_TTL_SECONDS at most: recent watches fade with time, so a
profile is never kept longer than that. Nothing is kept while titles are still being
embedded in the background, since the next page will read more of them.
"""

import hashlib
import threading
import weakref
from collections import OrderedDict

from . import settings
from .contract import Snapshot

PROFILE_SECONDS = settings.whole("RECOMMENDER_PROFILE_TTL_SECONDS", 600)
KEPT = 16


def shared_key(snapshot: Snapshot, *extra: object) -> tuple | None:
    """Names everything a profile was read from, or None when the history has no key."""
    if not snapshot.history_key or PROFILE_SECONDS <= 0:
        return None
    rest = snapshot.model_dump_json(
        exclude={"history", "favorites", "context", "algorithm", "recently_shown"}
    )
    context = snapshot.context
    return (
        snapshot.history_key,
        # Read with or without each family of data (an algorithm reads only what it lists).
        bool(snapshot.history),
        bool(snapshot.favorites),
        context.now // (PROFILE_SECONDS * 1000),
        context.timezone,
        context.locale,
        hashlib.sha1(rest.encode()).hexdigest(),
        *extra,
    )


class Kept:
    """The last few results, by shared key."""

    _all: "weakref.WeakSet[Kept]" = weakref.WeakSet()

    def __init__(self):
        self._lock = threading.Lock()
        self._results: OrderedDict[tuple, object] = OrderedDict()
        Kept._all.add(self)

    @classmethod
    def all(cls) -> list["Kept"]:
        """Every store of the process, for the memory report."""
        return list(cls._all)

    def __len__(self) -> int:
        return len(self._results)

    def results(self) -> list[object]:
        with self._lock:
            return list(self._results.values())

    def get(self, key: tuple | None) -> object | None:
        if key is None:
            return None
        with self._lock:
            found = self._results.get(key)
            if found is not None:
                self._results.move_to_end(key)
            return found

    def put(self, key: tuple | None, result: object, embedder: object) -> None:
        busy = getattr(embedder, "busy", None)
        if key is None or (busy is not None and busy()):
            return
        with self._lock:
            self._results[key] = result
            while len(self._results) > KEPT:
                self._results.popitem(last=False)
