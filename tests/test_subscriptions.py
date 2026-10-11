from recommender import engine
from recommender.contract import Overrides, Source, VideoRef
from recommender.contract_rank import RankRequest

from .factories import DAY, HOUR, group, snapshot, subscription, video

FEED = Source(kind="subscription_feed", limit=150)


def rank(snap, videos, trace=True):
    return engine.rank(RankRequest(snapshot=snap, candidates=[group(FEED, videos)], trace=trace))


def test_plan_asks_for_the_feed_only_when_there_are_subscriptions():
    assert engine.plan(snapshot("subscriptions")).sources == []
    sources = engine.plan(snapshot("subscriptions", subscriptions=[subscription("a")])).sources
    assert [s.kind for s in sources] == ["subscription_feed"]


def test_rows_split_by_day_newest_first():
    snap = snapshot("subscriptions", subscriptions=[subscription("a")])
    result = rank(
        snap,
        [
            video("old", "a", age_h=10 * 24),
            video("new", "a", age_h=1),
            video("mid", "a", age_h=3 * 24),
            video("newer", "a", age_h=0.5),
        ],
    )
    assert [row.id for row in result.rows] == [
        "subscriptions-today",
        "subscriptions-week",
        "subscriptions-earlier",
    ]
    assert [v.url[-5:] for v in result.rows[0].videos] == ["newer", "v=new"]
    assert result.rows[0].videos[0].because[0].ref == "subscription:a"


def test_muted_not_interested_and_shorts_are_dropped_and_traced():
    snap = snapshot(
        "subscriptions",
        subscriptions=[subscription("a"), subscription("b")],
        overrides=Overrides(muted_channels=["https://www.youtube.com/channel/b"]),
        not_interested=[VideoRef(id="ni:1", video_url="https://www.youtube.com/watch?v=meh")],
    )
    result = rank(
        snap,
        [
            video("ok", "a"),
            video("muted", "b"),
            video("meh", "a"),
            video("short", "a", is_short=True),
        ],
    )
    urls = [v.url for row in result.rows for v in row.videos]
    assert urls == ["https://www.youtube.com/watch?v=ok"]
    assert sorted(d.why for d in result.trace.dropped) == ["muted", "not_interested", "short"]


def test_channel_cap():
    snap = snapshot("subscriptions", {"max_per_channel": 2}, subscriptions=[subscription("a")])
    result = rank(snap, [video(str(i), "a", age_h=i) for i in range(5)])
    assert sum(len(row.videos) for row in result.rows) == 2
    assert [d.why for d in result.trace.dropped] == ["channel_cap"] * 3


def test_row_titles_follow_the_locale():
    snap = snapshot("subscriptions", subscriptions=[subscription("a")], locale="fr")
    result = rank(snap, [video("x", "a")])
    assert result.rows[0].title == "Aujourd'hui"
    assert "depuis minuit" in result.trace.rows[0].why


def test_values_are_clamped_to_bounds():
    snap = snapshot("subscriptions", {"max_per_channel": 999}, subscriptions=[subscription("a")])
    result = rank(snap, [video(str(i), "a", age_h=i) for i in range(120)])
    assert sum(len(row.videos) for row in result.rows) == 100


def test_by_default_a_prolific_channel_loses_nothing():
    snap = snapshot("subscriptions", subscriptions=[subscription("a")])
    result = rank(snap, [video(str(i), "a", age_h=i) for i in range(60)])
    assert sum(len(row.videos) for row in result.rows) == 60


def test_unknown_candidates_never_appear_and_days_ignore_missing_dates():
    snap = snapshot("subscriptions", subscriptions=[subscription("a")])
    undated = video("undated", "a").model_copy(update={"uploaded_at": None})
    result = rank(snap, [undated])
    assert result.rows[0].id == "subscriptions-earlier"
    assert DAY > HOUR


def test_a_watched_video_of_unknown_progress_is_not_offered_again():
    from recommender.contract import HistoryItem

    seen = HistoryItem(
        id="history:old",
        video_url="https://www.youtube.com/watch?v=old",
        channel_url="https://www.youtube.com/channel/a",
        watched_at=1,
    )  # imported: no duration, no progress
    snap = snapshot("signals", subscriptions=[subscription("a")], history=[seen])
    result = engine.rank(
        RankRequest(snapshot=snap, candidates=[group(FEED, [video("old", "a"), video("new", "a")])])
    )
    urls = [v.url for row in result.rows for v in row.videos]
    assert urls == ["https://www.youtube.com/watch?v=new"]
