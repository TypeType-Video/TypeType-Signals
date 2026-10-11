"""Shorts Server did not mark are told apart by length and title (goal: no Short on a page)."""

from recommender.base import looks_short

from .factories import snapshot, subscription, video
from .test_subscriptions import rank


def test_unmarked_shorts_are_dropped_and_short_videos_kept():
    snap = snapshot("subscriptions", subscriptions=[subscription("a")])
    result = rank(
        snap,
        [
            video("meme", "a", duration_s=35, title="Première animation"),
            video("tagged", "a", duration_s=88, title="À la recherche d'amis #animation"),
            video("song", "a", duration_s=150, title="Werenoi - Malus (Visualizer)"),
            video("trailer", "a", duration_s=90, title="Official Trailer"),
            video("long", "a", duration_s=1200),
            video("live", "a", duration_s=None),
        ],
    )
    kept = {v.url.rsplit("=", 1)[1] for row in result.rows for v in row.videos}
    assert kept == {"song", "trailer", "long", "live"}
    assert sorted(d.why for d in result.trace.dropped) == ["short", "short"]


def test_the_rule():
    assert looks_short(video("x", "a", duration_s=60))
    assert not looks_short(video("x", "a", duration_s=61))
    assert looks_short(video("x", "a", duration_s=180, title="clip #fyp"))
    assert not looks_short(video("x", "a", duration_s=181, title="clip #fyp"))
    assert looks_short(video("x", "a", duration_s=600, is_short=True))


def test_an_obscure_tagged_video_from_an_unknown_channel_is_dropped():
    from recommender.algorithms.signals_scoring import obscure

    junk = video("j", "nobody", views=2, title="Funny Moments #anime #fyp")
    assert obscure(junk, known_channels=set())
    assert not obscure(junk, known_channels={junk.uploader_url})  # a channel they watch
    assert not obscure(video("n", "nobody", views=2, title="Funny Moments"), set())
    assert not obscure(video("p", "nobody", views=5000, title="Moments #fyp"), set())
