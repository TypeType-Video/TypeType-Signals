"""An imported watch of unknown length is a Short when it was swiped through, or when its
channel only makes Shorts."""

from recommender import engine
from recommender.contract import HistoryItem
from recommender.signals_history import inferred_shorts

from .factories import HOUR, NOW, snapshot, watched


def imported(id: str, channel_name: str, at: int, title: str = "") -> HistoryItem:
    item = watched(id, channel_name, title=title or f"Video {id}")
    return HistoryItem(**{**item.model_dump(), "duration_s": 0, "progress_s": 0, "watched_at": at})


def test_a_swipe_run_is_shorts_and_a_long_gap_is_not():
    start = int(NOW - 5 * HOUR)
    run = [imported(f"s{i}", "swipes", start + i * 20_000) for i in range(5)]
    alone = imported("long", "docs", start + 3 * HOUR)
    found = inferred_shorts([*run, alone])
    assert {f"history:s{i}" for i in range(4)} <= found
    assert "history:long" not in found


def test_a_channel_of_shorts_makes_its_unknown_videos_shorts():
    history = [watched(f"k{i}", "shortsonly", duration_s=40, ago_h=20 + i) for i in range(3)]
    history = [h.model_copy(update={"is_short": True}) for h in history]
    unknown = imported("u", "shortsonly", int(NOW - 2 * HOUR))
    assert "history:u" in inferred_shorts([*history, unknown])


def test_inferred_shorts_count_for_no_channel():
    start = int(NOW - 5 * HOUR)
    run = [imported(f"s{i}", "swipes", start + i * 20_000) for i in range(6)]
    kept = imported("long", "docs", start + 3 * HOUR, title="Camping alone in the forest")
    profile = engine.profile(snapshot("signals", history=[*run, kept]))
    evidence = {c.id: c.from_ for c in profile.channels}
    assert evidence["channel:docs"] == ["history:long"]
    # The last of the run is followed three hours later: it alone still counts.
    assert evidence["channel:swipes"] == ["history:s5"]
