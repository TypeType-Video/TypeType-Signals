from recommender import engine
from recommender.contract import Source, VideoRef
from recommender.contract_rank import RankRequest

from .factories import group, snapshot, subscription, video, watched

FEED = Source(kind="subscription_feed", limit=120)
TRENDING = Source(kind="trending", limit=40)
RELATED_FEED = Source(kind="related", of="subscription_feed", count=3, limit=10)


def favorite(id: str) -> VideoRef:
    return VideoRef(id=f"fav:{id}", video_url=f"https://www.youtube.com/watch?v={id}", title=id)


def page(result):
    return [v.url for row in result.rows for v in row.videos]


def candidates():
    return [
        group(FEED, [video(f"s{i}", f"sub{i % 8}", age_h=i) for i in range(80)]),
        group(RELATED_FEED, [video(f"r{i}", f"rel{i}", age_h=i) for i in range(30)]),
        group(TRENDING, [video(f"t{i}", f"trend{i}", age_h=i) for i in range(40)]),
    ]


def test_plan_reads_subscriptions_favorites_and_trending():
    snap = snapshot(
        "discovery",
        subscriptions=[subscription("a")],
        favorites=[favorite("f1"), favorite("f2"), favorite("f3"), favorite("f4")],
    )
    kinds = [(s.kind, s.of, s.ref) for s in engine.plan(snap).sources]
    assert kinds[:2] == [("subscription_feed", None, None), ("related", "subscription_feed", None)]
    assert len([k for k in kinds if k[2] and k[2].startswith("fav:")]) == 3
    assert kinds[-1][0] == "trending"


def test_no_trending_source_when_its_share_is_zero():
    snap = snapshot("discovery", {"trending_share": 0}, subscriptions=[subscription("a")])
    assert "trending" not in [s.kind for s in engine.plan(snap).sources]


def test_page_is_split_by_share_and_capped_per_channel():
    snap = snapshot("discovery", {"random_enabled": False}, subscriptions=[subscription("a")])
    result = engine.rank(RankRequest(snapshot=snap, candidates=candidates(), trace=True))
    rows = {row.id.rsplit("-", 1)[0]: row for row in result.rows}
    assert len(rows["discovery-trending"].videos) == 16  # 20 % of 80
    # 40 asked from subscriptions, but 8 channels at 6 videos max
    assert len(rows["discovery-subscriptions"].videos) == 40
    assert len(page(result)) == len(set(page(result)))


def test_same_seed_same_page_new_seed_new_page():
    def run(seed):
        snap = snapshot("discovery", subscriptions=[subscription("a")])
        snap.context.seed = seed
        return page(engine.rank(RankRequest(snapshot=snap, candidates=candidates())))

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_without_randomness_the_seed_does_not_matter():
    def run(seed):
        snap = snapshot("discovery", {"random_enabled": False}, subscriptions=[subscription("a")])
        snap.context.seed = seed
        return page(engine.rank(RankRequest(snapshot=snap, candidates=candidates())))

    assert run(1) == run(2)


def test_finished_and_recently_shown_videos_are_left_out():
    snap = snapshot(
        "discovery",
        subscriptions=[subscription("a")],
        history=[watched("s0", "sub0", share=1.0)],
        recently_shown=["https://www.youtube.com/watch?v=t0"],
    )
    result = engine.rank(RankRequest(snapshot=snap, candidates=candidates(), trace=True))
    assert "https://www.youtube.com/watch?v=s0" not in page(result)
    assert "https://www.youtube.com/watch?v=t0" not in page(result)
    assert {d.why for d in result.trace.dropped} >= {"seen"}


def test_favorite_rows_cite_their_favorite():
    fav = favorite("f1")
    source = Source(kind="related", video_url=fav.video_url, limit=10, ref=fav.id)
    snap = snapshot("discovery", {"trending_share": 0}, favorites=[fav], locale="fr")
    result = engine.rank(
        RankRequest(
            snapshot=snap,
            candidates=[group(source, [video(f"x{i}", f"c{i}") for i in range(5)])],
            trace=True,
        )
    )
    row = result.rows[0]
    assert row.title == "Proche de f1"
    assert row.reason.ref == "fav:f1"
    assert row.videos[0].because[0].ref == "fav:f1"
    assert "favori f1" in result.trace.rows[0].why


def test_subscription_videos_cite_their_subscription():
    snap = snapshot("discovery", {"random_enabled": False}, subscriptions=[subscription("sub0")])
    result = engine.rank(RankRequest(snapshot=snap, candidates=candidates()))
    row = next(r for r in result.rows if r.reason.kind == "new_from_subscriptions")
    refs = {v.because[0].ref for v in row.videos if v.url.endswith(("=s0", "=s8"))}
    assert refs == {"subscription:sub0"}
