"""Plan snapshots kept between recommendation requests."""

import threading
import time
import uuid
from collections import OrderedDict

from . import settings
from .contract import Snapshot
from .history_cache import UnknownHistory, with_history
from .registry import UnknownAlgorithm, algorithm_of, restricted
from .state import default_state


class UnknownPlan(LookupError):
    """A rank asked for a plan this process no longer keeps: send the snapshot instead."""


# A plan lives from a page's plan call to its rank call (the sources in between: ~10 s).
# Each one holds a whole parsed history (~185 MB for 87,064 watches and 34,577 likes), so
# they are bounded by the watches and likes they hold, not by count: 64 of them took the
# service from 1.7 to 5.3 GB in 20 pages.
# The newest plan is always kept; a rank whose plan is gone sends its snapshot (409).
PLAN_SECONDS = settings.whole("RECOMMENDER_PLAN_TTL_SECONDS", 120)
PLAN_ITEMS_KEPT = 150_000
_plans: OrderedDict[str, tuple[float, Snapshot]] = OrderedDict()
_plans_lock = threading.Lock()


def _items(snapshot: Snapshot) -> int:
    # A snapshot naming a kept history shares its lists: it holds almost nothing of its own.
    return 0 if snapshot.history_key else len(snapshot.history) + len(snapshot.favorites)


def _plan_name(plan_id: str) -> str:
    return f"recommender:plan:v1:{plan_id}"


def _keep(snapshot: Snapshot) -> str:
    plan_id = uuid.uuid4().hex
    now = time.monotonic()
    with _plans_lock:
        _plans[plan_id] = (now, snapshot)
        held = sum(_items(s) for _, s in _plans.values())
        while len(_plans) > 1:
            oldest_at, oldest = next(iter(_plans.values()))
            if now - oldest_at <= PLAN_SECONDS and held <= PLAN_ITEMS_KEPT:
                break
            _plans.popitem(last=False)
            held -= _items(oldest)
    state = default_state()
    if state is not None and snapshot.history_key:
        # The plan without the history it names: small, and enough for another instance to
        # rank the page. A snapshot that carries its whole history stays with this process.
        light = snapshot.model_copy(update={"history": [], "favorites": []})
        state.set(_plan_name(plan_id), light.model_dump_json().encode(), PLAN_SECONDS)
    return plan_id


def _kept(plan_id: str) -> Snapshot:
    with _plans_lock:
        kept = _plans.get(plan_id)
    if kept is None or time.monotonic() - kept[0] > PLAN_SECONDS:
        return _shared_plan(plan_id)
    return kept[1]


def _shared_plan(plan_id: str) -> Snapshot:
    """A plan made by another instance, or before a restart: read back with its history."""
    state = default_state()
    stored = state.get(_plan_name(plan_id)) if state is not None else None
    if stored is None:
        raise UnknownPlan(plan_id)
    try:
        light = Snapshot.model_validate_json(stored)
        snapshot = restricted(with_history(light), algorithm_of(light))
    except (ValueError, UnknownHistory, UnknownAlgorithm) as error:
        raise UnknownPlan(plan_id) from error
    with _plans_lock:
        _plans[plan_id] = (time.monotonic(), snapshot)
    return snapshot
