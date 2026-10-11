"""Removed videos (no channel, their URL as title) and watches without a channel teach nothing."""

from recommender import engine
from recommender.contract import HistoryItem
from recommender.labels import informative

from .factories import snapshot, watched


def removed(id: str, ago_h: float) -> HistoryItem:
    item = watched(id, "x", ago_h=ago_h)
    url = f"https://www.youtube.com/watch?v={id}"
    return HistoryItem(
        **{**item.model_dump(), "channel_url": "", "channel_name": "Unknown channel", "title": url}
    )


def test_a_link_is_not_a_topic():
    assert not informative("https://www.youtube.com/watch?v=abcdefghijk")
    assert not informative("https://www.youtube.com/watch?v=VcpyMeiwz0o&feature=youtu.be")
    assert informative("Camping alone in the snowy forest")


def test_removed_videos_form_no_channel_and_no_group():
    history = [removed(f"r{i}", i + 1) for i in range(20)]
    history += [watched("a", "wild", title="Camping alone in the snowy forest", ago_h=30)]
    history += [watched("b", "wild", title="Winter camping shelter in the forest", ago_h=31)]
    profile = engine.profile(snapshot("signals", history=history))
    assert [c.id for c in profile.channels] == ["channel:wild"]
    assert all(not ref.startswith("history:r") for g in profile.groups for ref in g.from_)
