"""What is kept between the requests of one page: the history sent once under a key, and the
plan a rank names instead of sending the snapshot again."""

import threading
import time
import uuid
import zlib
from collections import OrderedDict

import msgspec

from . import settings
from .contract import (
    HistoryItem,
    HistoryUpload,
    Snapshot,
    VideoRef,
    read_history_upload,
    write_history_upload,
)
from .registry import ALGORITHMS, UnknownAlgorithm, algorithm_of, restricted
from .state import default_state


class UnknownHistory(LookupError):
    """A snapshot named a history this process no longer keeps: send the history again."""


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


# Histories sent to /v1/history, read once and shared by every page that names them. Bounded
# like plans: by age since last used and by the watches and likes held; the newest is kept.
# A history that is no longer held is read again from the shared store, so the bound costs
# time, never a page. 1,000 watches and likes held take about 0.5 MB (measured 2026-10-10
# on 121,641 of them: 61 MB; they took 2 MB per 1,000 as pydantic models).
HISTORY_SECONDS = settings.whole("RECOMMENDER_HISTORY_TTL_SECONDS", 30 * 60)
HISTORY_ITEMS_KEPT = settings.whole("RECOMMENDER_HISTORY_ITEMS_KEPT", 400_000)
_histories: OrderedDict[str, tuple[float, list[HistoryItem], list[VideoRef]]] = OrderedDict()
_histories_lock = threading.Lock()


def _history_name(key: str) -> str:
    return f"recommender:history:v1:{key}"


def _plan_name(plan_id: str) -> str:
    return f"recommender:plan:v1:{plan_id}"


def _share_history(upload: HistoryUpload) -> None:
    """Leaves the history in the shared store, for a restart and for the other instances."""
    state = default_state()
    if state is not None:
        packed = zlib.compress(write_history_upload(upload), 1)
        state.set(_history_name(upload.key), packed, HISTORY_SECONDS)


def _shared_history(key: str) -> HistoryUpload | None:
    state = default_state()
    packed = state.get(_history_name(key)) if state is not None else None
    if packed is None:
        return None
    try:
        return read_history_upload(zlib.decompress(packed))
    except (zlib.error, ValueError, msgspec.MsgspecError):
        return None


# The parts of a history still arriving, kept until the last one comes (or for a few minutes).
PART_SECONDS = settings.whole("RECOMMENDER_HISTORY_PART_TTL_SECONDS", 300)
_parts: dict[tuple[str, int], tuple[float, bytes]] = {}
_parts_lock = threading.Lock()


def _part_name(key: str, part: int) -> str:
    return f"recommender:history-part:v1:{key}:{part}"


def _keep_part(upload: HistoryUpload) -> None:
    packed = zlib.compress(write_history_upload(upload), 1)
    state = default_state()
    if state is not None:  # another instance may receive the next part
        state.set(_part_name(upload.key, upload.part), packed, PART_SECONDS)
        return
    now = time.monotonic()
    with _parts_lock:
        for name in [n for n, (kept_at, _) in _parts.items() if now - kept_at > PART_SECONDS]:
            del _parts[name]
        _parts[upload.key, upload.part] = (now, packed)


def _kept_part(key: str, part: int) -> HistoryUpload:
    state = default_state()
    if state is not None:
        packed = state.get(_part_name(key, part))
    else:
        with _parts_lock:
            packed = _parts.pop((key, part), (0.0, None))[1]
    if packed is None:
        raise UnknownHistory(f"{key}, part {part}")  # 409: Server sends the history again
    return read_history_upload(zlib.decompress(packed))


def receive_history(upload: HistoryUpload) -> None:
    """A history, whole or in parts. The parts are put back together when the last arrives;
    a part that is missing by then (it expired, the service restarted) answers 409."""
    if upload.part == 0 and upload.last:
        remember_history(upload)
        return
    if not upload.last:
        _keep_part(upload)
        return
    earlier = [_kept_part(upload.key, part) for part in range(upload.part)]
    remember_history(
        HistoryUpload(
            key=upload.key,
            history=[item for part in [*earlier, upload] for item in part.history],
            favorites=[item for part in [*earlier, upload] for item in part.favorites],
        )
    )


def remember_history(upload: HistoryUpload, share: bool = True) -> None:
    _hold_history(upload)
    if share and default_state() is not None:
        # Packing a long history takes longer than answering: done beside the request.
        threading.Thread(target=_share_history, args=(upload,), daemon=True).start()
    for algorithm in ALGORITHMS.values():
        warm = getattr(algorithm, "warm", None)
        if warm:
            warm(upload.history, upload.favorites)


def _hold_history(upload: HistoryUpload) -> None:
    now = time.monotonic()
    with _histories_lock:
        _histories.pop(upload.key, None)
        _histories[upload.key] = (now, upload.history, upload.favorites)
        held = sum(len(h) + len(f) for _, h, f in _histories.values())
        while len(_histories) > 1:
            used_at, history, favorites = next(iter(_histories.values()))
            if now - used_at <= HISTORY_SECONDS and held <= HISTORY_ITEMS_KEPT:
                break
            _histories.popitem(last=False)
            held -= len(history) + len(favorites)


def with_history(snapshot: Snapshot) -> Snapshot:
    """The snapshot with the history its key names (a snapshot that brings its own history
    under a key also leaves it for the next pages)."""
    key = snapshot.history_key
    if not key:
        return snapshot
    if snapshot.history or snapshot.favorites:
        remember_history(
            HistoryUpload(key=key, history=snapshot.history, favorites=snapshot.favorites)
        )
        return snapshot
    with _histories_lock:
        kept = _histories.get(key)
        if kept is not None and time.monotonic() - kept[0] <= HISTORY_SECONDS:
            _histories[key] = (time.monotonic(), kept[1], kept[2])
            _histories.move_to_end(key)
    if kept is None or time.monotonic() - kept[0] > HISTORY_SECONDS:
        # Not held by this process (it restarted, or another instance received it).
        shared = _shared_history(key)
        if shared is None:
            raise UnknownHistory(key)
        remember_history(shared, share=False)
        return snapshot.model_copy(
            update={"history": shared.history, "favorites": shared.favorites}
        )
    return snapshot.model_copy(update={"history": kept[1], "favorites": kept[2]})


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
