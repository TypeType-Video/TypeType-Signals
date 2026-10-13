"""History snapshots retained between recommendation requests."""

import threading
import time
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
from .registry import ALGORITHMS
from .state import default_state


class UnknownHistory(LookupError):
    """A snapshot named a history this process no longer keeps: send the history again."""


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


def _expire_histories_locked(now: float) -> None:
    while _histories:
        key, (used_at, _, _) = next(iter(_histories.items()))
        if now - used_at <= HISTORY_SECONDS:
            break
        del _histories[key]


def _hold_history(upload: HistoryUpload) -> None:
    now = time.monotonic()
    with _histories_lock:
        _expire_histories_locked(now)
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
        with _histories_lock:
            _expire_histories_locked(time.monotonic())
        return snapshot
    if snapshot.history or snapshot.favorites:
        remember_history(
            HistoryUpload(key=key, history=snapshot.history, favorites=snapshot.favorites)
        )
        return snapshot
    with _histories_lock:
        now = time.monotonic()
        _expire_histories_locked(now)
        kept = _histories.get(key)
        if kept is not None:
            _histories[key] = (now, kept[1], kept[2])
            _histories.move_to_end(key)
    if kept is None:
        # Not held by this process (it restarted, or another instance received it).
        shared = _shared_history(key)
        if shared is None:
            raise UnknownHistory(key)
        remember_history(shared, share=False)
        return snapshot.model_copy(
            update={"history": shared.history, "favorites": shared.favorites}
        )
    return snapshot.model_copy(update={"history": kept[1], "favorites": kept[2]})


def memory_stats() -> tuple[int, int, int]:
    with _histories_lock:
        sizes = [len(history) + len(favorites) for _, history, favorites in _histories.values()]
    with _parts_lock:
        part_count = len(_parts)
    return len(sizes), sum(sizes), part_count
