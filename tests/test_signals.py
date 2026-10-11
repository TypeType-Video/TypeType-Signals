from recommender import engine, signals_history
from recommender.contract import Overrides, Source
from recommender.contract_rank import RankRequest

from .factories import DAY, HOUR, NOW, channel, group, snapshot, subscription, video, watched

CAMPING = [
    ("c1", "wild", "Camping alone in the snowy forest"),
    ("c2", "wild", "Winter camping shelter in the forest"),
    ("c3", "north", "Solo camping forest cabin build"),
]
COOKING = [
    ("k1", "chef", "Street food noodles in Bangkok night market"),
    ("k2", "chef", "Night market street food tour Bangkok"),
]


def history(recent_share=1.0, recent_ago_h=0.5):
    items = [watched(id, ch, title=t, ago_h=24 * (i + 2)) for i, (id, ch, t) in enumerate(CAMPING)]
    items += [watched(id, ch, title=t, ago_h=24 * (i + 6)) for i, (id, ch, t) in enumerate(COOKING)]
    items.append(
        watched("now", "wild", title="Camping in a storm", share=recent_share, ago_h=recent_ago_h)
    )
    items.append(watched("half", "chef", title="Cooking a whole fish", share=0.4, ago_h=48))
    return items


def test_profile_finds_groups_channels_and_the_recent_watch():
    profile = engine.profile(snapshot("signals", history=history()))
    labels = [g.label.lower() for g in profile.groups]
    assert any("camping" in label for label in labels)
    assert profile.channels[0].name == "wild"
    assert profile.session_boosts[0].ref == "history:now"
    # 30 min after watching, with a 45 min half-life: 0.5 ** (30 / 45)
    assert abs(profile.session_boosts[0].strength - 0.63) < 0.01


def test_the_effect_of_a_watch_fades_and_stops():
    fresh = engine.profile(snapshot("signals", history=history(recent_ago_h=0.1)))
    older = engine.profile(snapshot("signals", history=history(recent_ago_h=1.5)))
    gone = engine.profile(snapshot("signals", history=history(recent_ago_h=4)))
    strength = lambda p: next(b.strength for b in p.session_boosts if b.ref == "history:now")  # noqa: E731
    assert strength(fresh) > strength(older)
    assert all(b.ref != "history:now" for b in gone.session_boosts)


def test_session_boost_off_and_reset_remove_recent_effect():
    off = engine.profile(snapshot("signals", {"session_boost": False}, history=history()))
    assert off.session_boosts == []
    reset = engine.profile(
        snapshot("signals", history=history(), overrides=Overrides(session_reset_at=NOW - 60_000))
    )
    assert reset.session_boosts == []


def test_shorts_are_left_out_unless_allowed():
    items = history() + [watched("s", "tiktoker", title="Funny cat", ago_h=0.2, is_short=True)]
    without = engine.profile(snapshot("signals", history=items))
    assert all(b.ref != "history:s" for b in without.session_boosts)
    allowed = engine.profile(snapshot("signals", {"use_shorts_history": True}, history=items))
    assert any(b.ref == "history:s" for b in allowed.session_boosts)


def test_plan_asks_for_no_trending_by_default():
    snap = snapshot("signals", history=history(), subscriptions=[subscription("sub")])
    assert "trending" not in [s.kind for s in engine.plan(snap).sources]


def test_plan_asks_for_started_related_catalogue_feed_and_trending():
    snap = snapshot(
        "signals",
        {"trending_share": 0.1},
        history=history(),
        subscriptions=[subscription("sub")],
    )
    sources = engine.plan(snap).sources
    kinds = [s.kind for s in sources]
    assert sources[0].kind == "videos" and sources[0].urls == [
        "https://www.youtube.com/watch?v=half"
    ]
    assert any(s.kind == "related" and s.ref == "history:now" for s in sources)
    assert any(s.kind == "related" and (s.ref or "").startswith("group:") for s in sources)
    assert sources[kinds.index("channel_videos")].channel_url == channel("wild")
    assert kinds[-2:] == ["subscription_feed", "trending"]


def rank(snap, trace=True):
    return engine.rank(RankRequest(snapshot=snap, candidates=rank_candidates(), trace=trace))


def rank_candidates():
    related_now = Source(
        kind="related", video_url="https://www.youtube.com/watch?v=now", limit=20, ref="history:now"
    )
    groups = [
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
            related_now,
            [
                video(f"r{i}", f"new{i}", title="Winter camping in the snowy forest")
                for i in range(12)
            ],
        ),
        group(
            Source(
                kind="channel_videos",
                channel_url=channel("chef"),
                sort="popular",
                limit=10,
                ref="channel:chef",
            ),
            [video(f"k{i}x", "chef", title="Bangkok street food market") for i in range(10)],
        ),
        group(
            Source(kind="subscription_feed", limit=60),
            [video(f"s{i}", "sub", age_h=i) for i in range(20)],
        ),
        group(
            Source(kind="trending", limit=30),
            [video(f"t{i}", f"trend{i}", views=10**8) for i in range(20)],
        ),
    ]
    return groups


def test_rows_follow_the_plan_of_the_page():
    snap = snapshot(
        "signals",
        {"random_enabled": False, "trending_share": 0.1},
        history=history(),
        subscriptions=[subscription("sub")],
    )
    result = rank(snap)
    ids = [row.id for row in result.rows]
    assert ids[0] == "continue"
    assert ids[1] == "watched-now"
    assert "subscriptions" in ids and "trending" in ids
    watched_row = result.rows[1]
    assert watched_row.title == "Because you watched Camping in a storm"
    assert watched_row.videos[0].because[0].ref == "history:now"
    urls = [v.url for row in result.rows for v in row.videos]
    assert len(urls) == len(set(urls))
    assert {s.url for s in result.trace.scores} >= set(urls) - {
        "https://www.youtube.com/watch?v=half"
    }


def test_every_video_is_explained_and_scores_add_up():
    snap = snapshot("signals", history=history(), subscriptions=[subscription("sub")])
    result = rank(snap)
    for row in result.rows:
        for v in row.videos:
            assert v.because, (row.id, v.url)
    for s in result.trace.scores:
        assert abs(sum(s.factors.values()) - s.total) < 1e-3


def test_muted_group_drops_close_candidates():
    base = engine.profile(snapshot("signals", history=history()))
    camping = next(g for g in base.groups if "camping" in g.label.lower())
    snap = snapshot(
        "signals",
        {"random_enabled": False, "session_boost": False},
        history=history(),
        overrides=Overrides(muted_groups=[camping.id]),
    )
    result = rank(snap)
    assert "muted" in {d.why for d in result.trace.dropped}
    shown = [v.url for row in result.rows for v in row.videos]
    assert not any("=r" in url for url in shown)


def test_french_titles():
    snap = snapshot("signals", history=history(), locale="fr")
    result = rank(snap)
    assert result.rows[0].title == "Reprendre"
    assert any(row.title.startswith("Parce que vous avez regardé") for row in result.rows)
    assert HOUR > 0


def test_the_page_fills_up_to_its_size_when_candidates_allow():
    snap = snapshot(
        "signals",
        {"random_enabled": False, "page_size": 40, "max_per_channel": 12, "max_per_topic": 80},
        history=history(),
        subscriptions=[subscription("sub")],
    )
    result = rank(snap)
    assert sum(len(row.videos) for row in result.rows) == 40


def test_no_topic_row_is_shorter_than_three_videos():
    snap = snapshot("signals", {"random_enabled": False}, history=history())
    for row in rank(snap).rows:
        if row.reason.kind == "topic":
            assert len(row.videos) >= 3, row.id


def test_a_group_without_a_readable_name_gets_no_row():
    from recommender.labels import group_label

    label, named = group_label(["hello there", "general kenobi"], ["x"], ["Chan A", "Chan B"])
    assert (label, named) == ("Chan A · Chan B", False)
    label, named = group_label(
        ["camping forest", "forest camping trip"],
        ["camping forest", "forest camping trip", "cooking"],
        ["x", "y"],
    )
    assert named and "camping" in label.lower()


def test_new_creators_row_keeps_only_creators_close_to_an_interest():
    snap = snapshot("signals", {"random_enabled": False}, history=history())
    related = Source(kind="search", query="camping forest", limit=20)
    close = [
        video(f"n{i}", f"newcamp{i}", views=50_000, title="Camping alone in a winter forest cabin")
        for i in range(6)
    ]
    far = [
        video(
            f"f{i}", f"studio{i}", views=50_000, title="Official movie trailer summer blockbuster"
        )
        for i in range(6)
    ]
    result = engine.rank(RankRequest(snapshot=snap, candidates=[group(related, close + far)]))
    discovery = [r for r in result.rows if r.reason.kind == "discovery"]
    assert discovery
    urls = {v.url for r in discovery for v in r.videos}
    assert urls and all("=n" in u for u in urls)


def test_new_creators_row_is_not_held_back_by_the_cap_per_interest():
    snap = snapshot("signals", {"random_enabled": False, "max_per_topic": 1}, history=history())
    related = Source(kind="search", query="camping forest", limit=20)
    close = [
        video(
            f"n{i}",
            f"newcamp{i}",
            views=50_000,
            title=f"Camping alone in a winter forest cabin {i}",
        )
        for i in range(8)
    ]
    result = engine.rank(RankRequest(snapshot=snap, candidates=[group(related, close)]))
    discovery = [v for r in result.rows if r.reason.kind == "discovery" for v in r.videos]
    assert len(discovery) > 1  # one interest, cap 1: the row still takes several


def test_new_creators_row_leaves_out_videos_almost_nobody_watched():
    snap = snapshot("signals", {"random_enabled": False}, history=history())
    related = Source(kind="search", query="camping forest", limit=20)
    title = "Camping alone in a winter forest cabin"
    seen = [video(f"n{i}", f"newcamp{i}", views=50_000, title=title) for i in range(4)]
    unseen = [video(f"u{i}", f"tinycamp{i}", views=900, title=title) for i in range(4)]
    result = engine.rank(RankRequest(snapshot=snap, candidates=[group(related, unseen + seen)]))
    urls = {v.url for r in result.rows if r.reason.kind == "discovery" for v in r.videos}
    assert urls and all("=n" in u for u in urls)


def test_new_creators_that_come_back_among_the_candidates_go_first():
    snap = snapshot("signals", {"random_enabled": False}, history=history())
    related = Source(kind="search", query="camping forest", limit=40)
    title = "Camping alone in a winter forest cabin"
    # More candidates than the row has room for: the creator seen three times gets in.
    once = [video(f"o{i}", f"once{i}", views=50_000, title=f"{title} {i}") for i in range(30)]
    back = [video(f"b{i}", "comesback", views=50_000, title=f"{title} {i}") for i in range(3)]
    result = engine.rank(RankRequest(snapshot=snap, candidates=[group(related, once + back)]))
    discovery = {v.url for r in result.rows if r.reason.kind == "discovery" for v in r.videos}
    assert sum("=b" in url for url in discovery) == 3


def test_videos_watched_on_several_days_get_their_own_row(monkeypatch):
    from recommender.contract import HistoryItem

    monkeypatch.setattr(signals_history, "WATCH_AGAIN", 8)
    song = "https://www.youtube.com/watch?v=song"
    again = [
        HistoryItem(
            id=f"history:song:{d}",
            video_url=song,
            channel_url=channel("music"),
            channel_name="music",
            title="A song",
            watched_at=int(NOW - d * DAY),
        )
        for d in (3, 5, 9)
    ]
    snap = snapshot("signals", {"random_enabled": False}, history=again + history())
    plan = engine.plan(snap)
    source = next(s for s in plan.sources if s.ref == "watch_again")
    assert source.urls == [song]
    result = engine.rank(
        RankRequest(snapshot=snap, candidates=[group(source, [video("song", "music")])])
    )
    rows = [r for r in result.rows if r.reason.kind == "watch_again"]
    assert [v.url for r in rows for v in r.videos] == [song]
    others = [v.url for r in result.rows if r.reason.kind != "watch_again" for v in r.videos]
    assert song not in others


def test_a_short_or_blocked_video_never_comes_back_through_watch_again(monkeypatch):
    from recommender.contract import HistoryItem, VideoRef

    monkeypatch.setattr(signals_history, "WATCH_AGAIN", 8)

    def again(vid, hours):  # watched on two days, hours apart from the other video
        url = f"https://www.youtube.com/watch?v={vid}"
        return [
            HistoryItem(
                id=f"history:{vid}:{d}",
                video_url=url,
                channel_url=channel("music"),
                channel_name="music",
                title="A song",
                watched_at=int(NOW - d * DAY - hours * HOUR),
            )
            for d in (3, 5)
        ]

    blocked = VideoRef(id="ni:1", video_url="https://www.youtube.com/watch?v=blocked")
    snap = snapshot(
        "signals",
        {"random_enabled": False},
        history=again("tiny", 0) + again("blocked", 2) + history(),
        not_interested=[blocked],
    )
    source = next(s for s in engine.plan(snap).sources if s.ref == "watch_again")
    candidates = [video("tiny", "music", duration_s=40), video("blocked", "music")]
    result = engine.rank(RankRequest(snapshot=snap, candidates=[group(source, candidates)]))
    assert not [v for r in result.rows for v in r.videos]


def test_an_explorer_gets_more_room_for_new_creators():
    from recommender.algorithms.signals_rows import discovery_share
    from recommender.contract import HistoryItem
    from recommender.signals_history import exploration_rate

    def watches(channels):
        return [
            HistoryItem(
                id=f"history:{i}",
                video_url=f"https://www.youtube.com/watch?v=v{i}",
                channel_url=channel(c),
                title="t",
                watched_at=int(NOW - i * HOUR),
            )
            for i, c in enumerate(channels)
        ] + [  # a long past, so the recent watches can be told new or not
            HistoryItem(
                id="history:old",
                video_url="https://www.youtube.com/watch?v=old",
                channel_url=channel("a"),
                title="t",
                watched_at=int(NOW - 400 * DAY),
            )
        ]

    loyal = watches(["a", "b"] * 60)  # always the same two channels
    explorer = watches([f"c{i}" if i % 2 else "a" for i in range(120)])  # every other one new
    assert exploration_rate(loyal, NOW) < 0.05
    assert 0.45 < exploration_rate(explorer, NOW) < 0.55
    assert discovery_share(0.3, exploration_rate(loyal, NOW)) == 0.3
    assert discovery_share(0.3, 0.5) == 0.6
    assert discovery_share(0.3, None) == 0.3
    assert exploration_rate(watches(["a"] * 10), NOW) is None  # too few to tell
    recent_only = watches([f"c{i}" for i in range(120)])[:100]  # five days of history
    assert exploration_rate(recent_only, NOW) is None  # too short to tell
