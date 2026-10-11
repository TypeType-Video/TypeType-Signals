"""Measure 5 of the goal: every reason is true, every control does what it says."""

import pytest

from recommender import engine
from recommender.contract import Overrides, Source, VideoRef
from recommender.contract_rank import RankRequest

from .factories import channel, group, snapshot, subscription, video
from .test_signals import history

FEED = Source(kind="subscription_feed", limit=60)
NOW_RELATED = Source(
    kind="related", video_url="https://www.youtube.com/watch?v=now", limit=20, ref="history:now"
)


def candidates():
    return [
        group(
            Source(
                kind="videos",
                urls=["https://www.youtube.com/watch?v=half"],
                limit=1,
                ref="continue",
            ),
            [video("half", "chef", title="Cooking a whole fish")],
        ),
        group(
            NOW_RELATED,
            [
                video(f"r{i}", f"new{i % 5}", title="Winter camping in the snowy forest")
                for i in range(15)
            ]
            + [video(f"rw{i}", "wild", title="Forest camping cabin") for i in range(3)],
        ),
        group(
            Source(
                kind="channel_videos",
                channel_url=channel("chef"),
                sort="popular",
                limit=10,
                ref="channel:chef",
            ),
            [video(f"k{i}x", "chef", title="Bangkok street food night market") for i in range(10)],
        ),
        group(FEED, [video(f"s{i}", f"sub{i % 3}", age_h=i) for i in range(20)]),
        group(
            Source(kind="trending", limit=30),
            [video(f"t{i}", f"trend{i}", views=10**8) for i in range(15)],
        ),
    ]


def page(parameters=None, **fields):
    fields.setdefault("history", history())
    fields.setdefault("subscriptions", [subscription(f"sub{i}") for i in range(3)])
    snap = snapshot("signals", {"random_enabled": False, **(parameters or {})}, **fields)
    return snap, engine.rank(RankRequest(snapshot=snap, candidates=candidates(), trace=True))


def urls(result):
    return [v.url for row in result.rows for v in row.videos]


def channel_of(url):
    for g in candidates():
        for v in g.videos:
            if v.url == url:
                return v.uploader_url
    raise KeyError(url)


def test_every_reason_is_true():
    snap, result = page()
    profile = engine.profile(snap)
    known_refs = (
        {i.id for i in snap.history}
        | {s.id for s in snap.subscriptions}
        | {g.id for g in profile.groups}
        | {"channel:" + i.channel_url.rsplit("/", 1)[-1] for i in snap.history}
    )
    history_channels = {i.channel_url for i in snap.history}
    subscribed = {s.channel_url for s in snap.subscriptions}
    related_to = {v.url for g in candidates() if g.source.ref == "history:now" for v in g.videos}
    for row in result.rows:
        for v in row.videos:
            assert v.because, (row.id, v.url)
            for reason in v.because:
                assert reason.ref is None or reason.ref in known_refs, (row.id, reason)
            if row.reason.kind == "discovery":
                assert channel_of(v.url) not in history_channels | subscribed, v.url
            if row.reason.kind == "new_from_subscriptions":
                assert channel_of(v.url) in subscribed, v.url
            if row.reason.kind == "watched":
                assert v.url in related_to or channel_of(v.url) == channel("wild"), v.url
            if row.reason.kind == "continue":
                assert v.url == "https://www.youtube.com/watch?v=half"
    for row in result.trace.rows:
        assert row.why


def test_muted_channel_never_appears():
    _, result = page(overrides=Overrides(muted_channels=[channel("chef")]))
    assert all(channel_of(u) != channel("chef") for u in urls(result))


def test_not_interested_video_never_appears():
    bad = "https://www.youtube.com/watch?v=r0"
    _, result = page(not_interested=[VideoRef(id="ni:1", video_url=bad)])
    assert bad not in urls(result)


def test_ignored_channel_does_not_count_as_an_interest():
    snap, _ = page()
    before = engine.profile(snap)
    assert any(c.id == "channel:chef" and c.weight > 0 for c in before.channels)
    snap, result = page(overrides=Overrides(ignored_signals=["channel:chef"]))
    for s in result.trace.scores:
        if channel_of(s.url) == channel("chef"):
            assert s.factors["same_channel"] == 0


def test_ignored_group_gives_no_topic_score():
    snap, _ = page()
    groups = engine.profile(snap).groups
    assert groups
    ignored = [g.id for g in groups]
    _, result = page(overrides=Overrides(ignored_signals=ignored))
    assert all(s.factors["topic"] == 0 for s in result.trace.scores)
    assert all(row.reason.kind != "topic" for row in result.rows)


def test_pinned_group_always_gets_a_row():
    snap, _ = page()
    last = engine.profile(snap).groups[-1]
    _, result = page(overrides=Overrides(pinned_groups=[last.id]))
    assert any(row.reason.ref == last.id for row in result.rows)


def test_a_subject_the_person_added_is_a_pinned_interest_searched_in_the_plan():
    twice = ["  astronomy  telescopes ", "Astronomy telescopes"]
    snap, _ = page(overrides=Overrides(custom_interests=twice))
    added = [g for g in engine.profile(snap).groups if g.id.startswith("custom:")]
    assert [g.label for g in added] == ["astronomy telescopes"]
    assert added[0].pinned and added[0].weight > 0
    searches = [s for s in engine.plan(snap).sources if s.kind == "search" and s.ref == added[0].id]
    assert [s.query for s in searches] == ["astronomy telescopes"]


def test_without_an_added_subject_nothing_changes():
    snap, plain = page()
    _, same = page(overrides=Overrides(custom_interests=[]))
    assert plain.rows == same.rows
    assert all(not g.id.startswith("custom:") for g in engine.profile(snap).groups)


@pytest.mark.parametrize(
    ("parameter", "kind"),
    [
        ("discovery_share", "discovery"),
        ("subscriptions_share", "new_from_subscriptions"),
        ("trending_share", "trending"),
    ],
)
def test_a_share_at_zero_means_no_such_row(parameter, kind):
    _, result = page({parameter: 0, "page_size": 30})
    assert all(row.reason.kind != kind for row in result.rows)


@pytest.mark.parametrize(
    ("parameter", "kind"),
    [("discovery_share", "discovery"), ("subscriptions_share", "new_from_subscriptions")],
)
def test_a_share_is_never_exceeded(parameter, kind):
    _, result = page({parameter: 0.1, "page_size": 40})
    shown = sum(len(row.videos) for row in result.rows if row.reason.kind == kind)
    assert shown <= round(40 * 0.1)


def test_max_per_channel_is_respected():
    _, result = page({"max_per_channel": 2})
    per_channel = {}
    for u in urls(result):
        per_channel[channel_of(u)] = per_channel.get(channel_of(u), 0) + 1
    assert max(per_channel.values()) <= 2


def test_page_size_is_respected():
    _, result = page({"page_size": 20})
    assert len(urls(result)) <= 20


def test_without_randomness_same_data_same_page_with_it_the_seed_changes_it():
    snap, first = page()
    snap.context.seed = 99
    again = engine.rank(RankRequest(snapshot=snap, candidates=candidates()))
    assert urls(first) == urls(again)
    random_snap = snapshot(
        "signals",
        {"random_enabled": True},
        history=history(),
        subscriptions=[subscription(f"sub{i}") for i in range(3)],
    )
    pages = []
    for seed in (1, 2):
        random_snap.context.seed = seed
        pages.append(urls(engine.rank(RankRequest(snapshot=random_snap, candidates=candidates()))))
    assert pages[0] != pages[1]


def test_session_boost_off_removes_the_watched_rows():
    _, result = page({"session_boost": False})
    assert all(row.reason.kind != "watched" for row in result.rows)


def test_the_trace_explains_every_row_and_scores_every_ranked_video():
    _, result = page()
    assert {r.id for r in result.trace.rows} == {row.id for row in result.rows}
    scored = {s.url for s in result.trace.scores}
    for row in result.rows:
        if row.reason.kind != "continue":
            assert {v.url for v in row.videos} <= scored


def test_the_room_for_recent_watches_is_never_exceeded():
    for share in (0.05, 0.1):
        _, result = page({"recent_share": share, "page_size": 40})
        recent = sum(len(row.videos) for row in result.rows if row.reason.kind == "watched")
        assert recent <= round(40 * share)


def test_no_interest_takes_more_than_its_cap_outside_the_new_creators_row():
    # The new creators row has its own limit per interest (DISCOVERY_BUCKET_SHARE).
    snap, result = page({"max_per_topic": 2, "session_boost": False})
    topics = engine.topics(
        __import__("recommender.contract_profile", fromlist=["TopicRequest"]).TopicRequest(
            snapshot=snap,
            titles=[
                next(
                    v.title + " — " + v.uploader_name
                    for g in candidates()
                    for v in g.videos
                    if v.url == u
                )
                for row in result.rows
                for u in [x.url for x in row.videos]
                if row.reason.kind not in ("continue", "discovery")
            ],
        )
    ).matches
    counts = {}
    for m in topics:
        if m.group:
            counts[m.group] = counts.get(m.group, 0) + 1
    assert counts and max(counts.values()) <= 2


def test_a_channel_row_holds_only_that_channel_and_says_so():
    _, result = page({"max_per_topic": 80, "session_boost": False})
    channel_rows = [row for row in result.rows if row.id.startswith("channel-")]
    for row in channel_rows:
        assert row.reason.kind == "same_channel" and row.reason.ref
        assert len({channel_of(v.url) for v in row.videos}) == 1
