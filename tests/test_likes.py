"""Likes carry their date: they fade like watches, and a long list of them stays bounded."""

from recommender import engine, signals_groups
from recommender.contract import VideoRef

from .factories import HOUR, NOW, channel, snapshot

DAY = 24 * HOUR


def like(id: str, channel_name: str, title: str, days_ago: float | None) -> VideoRef:
    return VideoRef(
        id=f"favorite:{id}",
        video_url=f"https://www.youtube.com/watch?v={id}",
        channel_url=channel(channel_name),
        title=title,
        added_at=None if days_ago is None else int(NOW - days_ago * DAY),
    )


def weights(favorites: list[VideoRef]) -> dict[str, float]:
    profile = engine.profile(snapshot("signals", favorites=favorites))
    return {c.id: c.weight for c in profile.channels}


def test_an_old_like_weighs_less_than_a_recent_one():
    w = weights(
        [
            like("a", "recent", "Camping in the snow", 2),
            like("b", "old", "Camping in the rain", 900),
        ]
    )
    assert w["channel:recent"] > 50 * w["channel:old"]


def test_undated_likes_count_in_full_as_before():
    w = weights(
        [
            like("a", "one", "Camping in the snow", None),
            like("b", "two", "Camping in the rain", None),
        ]
    )
    assert w["channel:one"] == w["channel:two"] == 0.5


def test_the_clustering_keeps_the_most_recent_titles(monkeypatch):
    monkeypatch.setattr(signals_groups, "MAX_TITLES", 4)
    recent = [
        like(f"r{i}", "recent", f"Winter camping shelter in the forest {i}", i) for i in range(4)
    ]
    old = [like(f"o{i}", "old", f"Street food noodles night market {i}", 800 + i) for i in range(4)]
    profile = engine.profile(snapshot("signals", favorites=old + recent))
    members = {ref for g in profile.groups for ref in g.from_}
    assert members and members <= {f.id for f in recent}


def test_a_liked_short_counts_for_nothing():
    short = like("s", "shorts", "Funny cat jumps #shorts", 1).model_copy(update={"is_short": True})
    w = weights([short, like("a", "long", "Camping in the snow", 1)])
    assert "channel:shorts" not in w and w["channel:long"] == 1.0
