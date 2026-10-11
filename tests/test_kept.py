"""A person's profile is worked out once while their history has not changed."""

from collections import OrderedDict

from recommender import engine, history_cache
from recommender.algorithms import signals as signals_module
from recommender.contract import HistoryUpload, Snapshot
from recommender.kept import PROFILE_SECONDS, Kept, shared_key

from .factories import NOW, snapshot, subscription, watched


def _named(key: str, **fields) -> Snapshot:
    return Snapshot.model_validate(
        {**snapshot("signals", **fields).model_dump(), "history_key": key}
    )


def test_a_second_page_of_the_same_history_does_not_read_it_again(monkeypatch):
    history = [watched(f"v{i}", f"c{i % 4}", title=f"Cooking pasta at home {i}") for i in range(20)]
    history_cache.remember_history(HistoryUpload(key="kept-1", history=history), share=False)
    calls = []
    real = signals_module.conclude
    monkeypatch.setattr(signals_module, "conclude", lambda *args: calls.append(1) or real(*args))
    first = engine.plan(_named("kept-1"))
    again = engine.plan(_named("kept-1"))
    engine.profile(_named("kept-1"))
    assert len(calls) == 1
    assert first.sources == again.sources


def test_what_changes_a_profile_changes_its_key():
    base = _named("kept-2")
    assert shared_key(base) == shared_key(_named("kept-2"))
    assert shared_key(base) != shared_key(_named("kept-3"))
    assert shared_key(base) != shared_key(_named("kept-2", subscriptions=[subscription("c")]))
    muted = base.model_copy(
        update={"overrides": base.overrides.model_copy(update={"muted_groups": ["g"]})}
    )
    assert shared_key(base) != shared_key(muted)
    later = base.model_copy(
        update={"context": base.context.model_copy(update={"now": NOW + PROFILE_SECONDS * 1000})}
    )
    assert shared_key(base) != shared_key(later)
    other_seed = base.model_copy(update={"context": base.context.model_copy(update={"seed": 9})})
    assert shared_key(base) == shared_key(other_seed)


def test_a_snapshot_that_carries_its_history_is_never_kept():
    assert shared_key(snapshot("signals", history=[watched("v", "c")])) is None


def test_nothing_is_kept_while_titles_are_still_being_embedded():
    class Busy:
        def busy(self):
            return True

    kept = Kept()
    kept.put(("key",), "profile", Busy())
    assert kept.get(("key",)) is None

    kept.put(("key",), "profile", object())
    assert kept.get(("key",)) == "profile"


def test_expired_history_is_removed_on_next_request(monkeypatch):
    now = [10.0]
    histories = OrderedDict()
    monkeypatch.setattr(history_cache, "_histories", histories)
    monkeypatch.setattr(history_cache, "HISTORY_SECONDS", 30)
    monkeypatch.setattr(history_cache.time, "monotonic", lambda: now[0])
    history_cache._hold_history(HistoryUpload(key="idle", history=[watched("v", "c")]))
    now[0] += 31

    history_cache.with_history(snapshot("subscriptions"))

    assert not histories
