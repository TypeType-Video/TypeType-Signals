"""Measure 5 for Subscriptions and Discovery: reasons are true, controls do what they say."""

from recommender import engine, registry
from recommender.contract import Overrides, Source, VideoRef
from recommender.contract_rank import RankRequest

from .factories import channel, group, snapshot, subscription, video, watched

FEED = Source(kind="subscription_feed", limit=120)
RELATED_FEED = Source(kind="related", of="subscription_feed", count=6, limit=10)


def feed_videos():
    return [video(f"s{i}", f"sub{i % 4}", age_h=i * 5) for i in range(40)]


def discovery_candidates():
    return [
        group(FEED, feed_videos()),
        group(
            RELATED_FEED,
            [video(f"r{i}", f"rel{i % 7}", age_h=i) for i in range(20)]
            + [video(f"rs{i}", "sub0", age_h=i) for i in range(5)],
        ),
        group(Source(kind="trending", limit=40), [video(f"t{i}", f"trend{i}") for i in range(20)]),
    ]


def subs():
    return [subscription(f"sub{i}") for i in range(4)]


def urls(result):
    return [v.url for row in result.rows for v in row.videos]


def by_url(groups):
    return {v.url: v for g in groups for v in g.videos}


def test_subscriptions_every_video_is_from_a_followed_channel_and_cites_it():
    snap = snapshot("subscriptions", subscriptions=subs())
    result = engine.rank(RankRequest(snapshot=snap, candidates=[group(FEED, feed_videos())]))
    followed = {s.channel_url: s.id for s in subs()}
    videos = by_url([group(FEED, feed_videos())])
    for row in result.rows:
        for v in row.videos:
            assert videos[v.url].uploader_url in followed
            assert v.because[0].ref == followed[videos[v.url].uploader_url]


def test_subscriptions_never_reads_history():
    def run(snap):
        return urls(
            engine.rank(RankRequest(snapshot=snap, candidates=[group(FEED, feed_videos())]))
        )

    base = snapshot("subscriptions", subscriptions=subs())
    heavy = snapshot("subscriptions", subscriptions=subs(), history=[watched("s0", "sub0")])
    assert run(base) == run(heavy)


def test_the_engine_hides_what_an_algorithm_does_not_list():
    snap = snapshot("discovery", subscriptions=subs(), history=[watched("s0", "sub0")])
    seen = registry.restricted(snap, registry.ALGORITHMS["discovery"])
    assert seen.history == [] and seen.subscriptions


def test_subscriptions_shorts_only_when_asked():
    videos = feed_videos() + [video("short", "sub1", is_short=True)]
    for include, expected in ((False, False), (True, True)):
        snap = snapshot("subscriptions", {"include_shorts": include}, subscriptions=subs())
        result = engine.rank(RankRequest(snapshot=snap, candidates=[group(FEED, videos)]))
        assert ("https://www.youtube.com/watch?v=short" in urls(result)) is expected


def test_discovery_never_reads_history():
    def run(history):
        snap = snapshot("discovery", subscriptions=subs(), history=history)
        return urls(engine.rank(RankRequest(snapshot=snap, candidates=discovery_candidates())))

    assert run([]) == run([watched("r0", "rel0"), watched("x", "other", share=0.5)])


def test_discovery_reasons_are_true():
    snap = snapshot("discovery", {"random_enabled": False}, subscriptions=subs())
    result = engine.rank(RankRequest(snapshot=snap, candidates=discovery_candidates()))
    videos = by_url(discovery_candidates())
    followed = {s.channel_url for s in subs()}
    trending = {v.url for v in discovery_candidates()[2].videos}
    related = {v.url for v in discovery_candidates()[1].videos}
    for row in result.rows:
        for v in row.videos:
            if row.reason.kind == "new_from_subscriptions":
                assert videos[v.url].uploader_url in followed
            if row.reason.kind == "discovery":
                assert v.url in related and videos[v.url].uploader_url not in followed
            if row.reason.kind == "trending":
                assert v.url in trending


def test_discovery_without_related_videos_asks_for_none():
    snap = snapshot("discovery", {"related_per_seed": 0}, subscriptions=subs())
    assert all(s.kind != "related" for s in engine.plan(snap).sources)


def test_discovery_respects_mutes_and_the_channel_cap():
    snap = snapshot(
        "discovery",
        {"max_per_channel": 3},
        subscriptions=subs(),
        overrides=Overrides(muted_channels=[channel("sub2")]),
        not_interested=[VideoRef(id="ni:1", video_url="https://www.youtube.com/watch?v=t0")],
    )
    result = engine.rank(RankRequest(snapshot=snap, candidates=discovery_candidates()))
    videos = by_url(discovery_candidates())
    per_channel: dict[str, int] = {}
    for u in urls(result):
        per_channel[videos[u].uploader_url] = per_channel.get(videos[u].uploader_url, 0) + 1
    assert channel("sub2") not in per_channel
    assert max(per_channel.values()) <= 3
    assert "https://www.youtube.com/watch?v=t0" not in urls(result)


def test_discovery_plan_rotates_feed_starts_only_with_randomness():
    snap = snapshot("discovery", subscriptions=subs())
    related = next(s for s in engine.plan(snap).sources if s.of == "subscription_feed")
    assert related.among and related.count < related.among
    fixed = snapshot("discovery", {"random_enabled": False}, subscriptions=subs())
    related = next(s for s in engine.plan(fixed).sources if s.of == "subscription_feed")
    assert related.among is None
