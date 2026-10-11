"""Signals by channels, by videos, and the mix of the two. All data here is made up."""

from fastapi.testclient import TestClient

from recommender import engine, registry
from recommender.algorithms.signals_mix import SignalsMix
from recommender.algorithms.signals_videos import SignalsVideos
from recommender.catalogue import Catalogue, Facts
from recommender.contract import Source
from recommender.contract_rank import RankRequest
from recommender.embeddings import HashingEmbedder
from recommender.main import app

from .factories import channel, group, snapshot, subscription, video, watched

CAMPING = "Camping alone in a winter forest cabin"
COOKING = "Street food noodles in a Bangkok night market"


def history():
    items = [
        watched(f"camp{i}", "wild", title=f"{CAMPING} {i}", ago_h=24 * (i + 1)) for i in range(4)
    ]
    items += [
        watched(f"cook{i}", "chef", title=f"{COOKING} {i}", ago_h=24 * (i + 6)) for i in range(3)
    ]
    return items


def rank(algorithm, groups, parameters=None, **fields):
    snap = snapshot(algorithm, parameters, history=fields.pop("history", history()), **fields)
    return engine.rank(RankRequest(snapshot=snap, candidates=groups, trace=True))


def urls(result, kind=None):
    return [v.url for r in result.rows if kind is None or r.reason.kind == kind for v in r.videos]


# --- channels ---------------------------------------------------------------------------


def test_channels_asks_for_the_watched_channels_and_the_subscriptions():
    snap = snapshot("signals-channels", history=history(), subscriptions=[subscription("news")])
    sources = engine.plan(snap).sources
    assert [s.kind for s in sources] == ["channel_videos", "channel_videos", "subscription_feed"]
    assert sources[0].channel_url == channel("wild")  # the most watched first
    assert all(s.kind not in ("search", "related") for s in sources)


def test_channels_never_shows_a_creator_the_person_does_not_know():
    feed = group(
        Source(kind="subscription_feed", limit=60), [video("s1", "news"), video("x1", "stranger")]
    )
    own = group(
        Source(kind="channel_videos", channel_url=channel("wild"), limit=8),
        [video(f"w{i}", "wild") for i in range(3)] + [video("x2", "stranger")],
    )
    result = rank("signals-channels", [own, feed], subscriptions=[subscription("news")])
    shown = urls(result)
    assert shown and not any("=x" in url for url in shown)
    assert urls(result, "same_channel")[0].endswith(("w0", "w1", "w2"))
    assert urls(result, "new_from_subscriptions") == ["https://www.youtube.com/watch?v=s1"]


def test_channels_puts_the_most_watched_channel_first_and_caps_each_one():
    wild = group(
        Source(kind="channel_videos", channel_url=channel("wild"), limit=20),
        [video(f"w{i}", "wild") for i in range(9)],
    )
    chef = group(
        Source(kind="channel_videos", channel_url=channel("chef"), limit=20),
        [video(f"k{i}", "chef") for i in range(2)],
    )
    shown = urls(rank("signals-channels", [chef, wild], {"max_per_channel": 3}))
    assert "=w" in shown[0]
    assert sum("=w" in url for url in shown) == 3 and sum("=k" in url for url in shown) == 2


# --- videos -----------------------------------------------------------------------------


def test_videos_searches_each_subject_and_asks_what_relates_to_it():
    sources = engine.plan(snapshot("signals-videos", history=history())).sources
    assert {s.kind for s in sources} == {"search", "related"}
    assert any("camping" in (s.query or "").lower() for s in sources)
    assert all(s.ref and s.ref.startswith("subject:") for s in sources)


def test_videos_keeps_what_is_about_a_subject_whoever_made_it():
    candidates = group(
        Source(kind="search", query="camping", limit=20),
        [
            *[
                video(f"n{i}", f"stranger{i}", views=50_000, title=f"{CAMPING} part {i}")
                for i in range(5)
            ],
            *[
                video(
                    f"t{i}",
                    f"studio{i}",
                    views=50_000,
                    title="Official movie trailer summer blockbuster",
                )
                for i in range(5)
            ],
        ],
    )
    result = rank("signals-videos", [candidates])
    shown = urls(result)
    assert shown and all("=n" in url for url in shown)  # creators never watched, on the subject
    assert any(r.reason.kind == "topic" for r in result.rows)


def test_videos_gives_a_watched_channel_no_advantage_and_caps_every_channel():
    candidates = group(
        Source(kind="search", query="camping", limit=20),
        [
            *[video(f"w{i}", "wild", views=50_000, title=f"{CAMPING} {i}") for i in range(6)],
            *[
                video(f"n{i}", f"stranger{i}", views=50_000, title=f"{CAMPING} {i}")
                for i in range(6)
            ],
        ],
    )
    shown = urls(rank("signals-videos", [candidates]))
    assert sum("=w" in url for url in shown) == 2  # max_per_channel, known channel or not
    assert sum("=n" in url for url in shown) == 6


def test_videos_leaves_out_videos_almost_nobody_watched_and_prefers_a_creator_seen_twice():
    title = "Camping alone in a winter forest cabin"
    candidates = group(
        Source(kind="search", query="camping", limit=40),
        [
            video("tiny", "nobody", views=900, title=f"{title} tiny"),
            *[video(f"o{i}", f"once{i}", views=50_000, title=f"{title} {i}") for i in range(30)],
            *[video(f"b{i}", "comesback", views=50_000, title=f"{title} {i}") for i in range(2)],
        ],
    )
    shown = urls(rank("signals-videos", [candidates]))
    assert not any("=tiny" in url for url in shown)
    assert sum("=b" in url for url in shown) == 2  # among 32 alike, the creator seen twice gets in


def test_videos_reads_a_watched_video_by_its_tags_when_the_catalogue_has_them():
    catalogue = Catalogue()
    items = [
        watched(f"vid0000000{i}", "wild", title=f"Episode {i} of the series", ago_h=24 * (i + 1))
        for i in range(3)
    ]
    plain = SignalsVideos(HashingEmbedder(), catalogue)
    without = plain.subjects(snapshot("signals-videos", history=items))
    catalogue.put(
        [
            Facts(
                f"vid0000000{i}",
                f"Episode {i} of the series",
                channel("wild"),
                "Travel",
                ("bushcraft", "winter camping"),
            )
            for i in range(3)
        ]
    )
    with_tags = plain.subjects(snapshot("signals-videos", history=items))
    assert all("bushcraft" not in subject.query for subject in without)
    assert with_tags[0].query in ("bushcraft winter camping", "winter camping bushcraft")
    assert "Bushcraft" in with_tags[0].label


def test_videos_with_no_history_gives_an_empty_page_and_no_error():
    snap = snapshot("signals-videos", history=[])
    assert engine.plan(snap).sources == []
    assert engine.rank(RankRequest(snapshot=snap, candidates=[])).rows == []


# --- mix --------------------------------------------------------------------------------


def mix_groups():
    own = group(
        Source(kind="channel_videos", channel_url=channel("wild"), limit=20),
        [video(f"w{i}", "wild") for i in range(20)] + [video(f"k{i}", "chef") for i in range(20)],
    )
    found = group(
        Source(kind="search", query="camping", limit=40),
        [
            video(f"n{i}", f"stranger{i}", views=50_000, title=f"{CAMPING} part {i}")
            for i in range(30)
        ],
    )
    return [own, found]


def test_mix_gives_each_part_its_share_of_the_page():
    result = rank("signals-mix", mix_groups(), {"videos_share": 0.5, "page_size": 20})
    shown = urls(result)
    assert len(shown) == 20
    assert sum("=n" in url for url in shown) == 10
    assert sum("=w" in url or "=k" in url for url in shown) == 10


def test_mix_with_no_share_for_subjects_is_channels_alone():
    shown = urls(rank("signals-mix", mix_groups(), {"videos_share": 0, "page_size": 20}))
    assert shown and not any("=n" in url for url in shown)
    sources = engine.plan(snapshot("signals-mix", {"videos_share": 0}, history=history())).sources
    assert {s.kind for s in sources} == {"channel_videos"}


def test_mix_lets_one_part_take_the_page_when_the_other_has_nothing():
    only_channels = [mix_groups()[0]]
    shown = urls(rank("signals-mix", only_channels, {"videos_share": 0.5, "page_size": 20}))
    assert len(shown) == 10  # five per channel, the cap: all from the channels
    assert (
        SignalsMix().plan(
            snapshot("signals-mix", history=[]),
            registry.ALGORITHMS["signals-mix"].values(snapshot("signals-mix", history=[])),
        )
        == []
    )


# --- catalogue --------------------------------------------------------------------------


def test_catalogue_keeps_a_video_the_same_in_memory_as_stored():
    catalogue = Catalogue()
    tags = tuple(f"tag{i}" for i in range(30))
    catalogue.put([Facts("abcdefghijk", "A title", "", "Music", tags, "d" * 900)])
    kept = catalogue.get(["abcdefghijk"])["abcdefghijk"]
    assert len(kept.tags) == 12 and len(kept.description) == 300


def test_catalogue_ignores_a_video_that_says_nothing_about_itself():
    catalogue = Catalogue()
    assert catalogue.put([Facts("abcdefghijk", "Only a title")]) == 0
    assert catalogue.get(["abcdefghijk"]) == {}


def test_http_lists_the_six_algorithms_and_takes_catalogue_uploads():
    client = TestClient(app)
    ids = [a["id"] for a in client.get("/v1/algorithms").json()["algorithms"]]
    assert ids == [
        "subscriptions",
        "discovery",
        "signals",
        "signals-channels",
        "signals-videos",
        "signals-mix",
    ]
    body = client.post(
        "/v1/catalogue",
        json={
            "videos": [
                {
                    "url": "https://www.youtube.com/watch?v=abcdefghijk",
                    "title": "A",
                    "category": "Music",
                    "tags": ["jazz"],
                },
                {"url": "https://www.youtube.com/watch?v=lmnopqrstuv", "title": "No facts"},
            ]
        },
    ).json()
    assert body["kept"] == 1 and body["size"] >= 1


# --- a page that changes on refresh ------------------------------------------------------


def _page(algorithm, seed, vary):
    own = group(
        Source(kind="channel_videos", channel_url=channel("wild"), limit=40),
        [video(f"w{i}", "wild", age_h=i + 1) for i in range(40)],
    )
    about = group(
        Source(kind="search", query="camping", limit=40, ref="subject:camp0"),
        [
            video(f"s{i}", f"maker{i}", title=f"{CAMPING} guide {i}", views=50_000)
            for i in range(40)
        ],
    )
    snap = snapshot(
        algorithm,
        {"random_enabled": vary, "page_size": 20},
        history=history(),
    )
    snap.context.seed = seed
    return urls(engine.rank(RankRequest(snapshot=snap, candidates=[own, about])))


def test_the_three_algorithms_show_another_page_on_refresh_and_the_same_for_one_seed():
    for algorithm in ("signals-channels", "signals-videos", "signals-mix"):
        first, again, other = (_page(algorithm, seed, True) for seed in (7, 7, 8))
        assert first and first == again, algorithm
        assert first != other, algorithm


def test_with_variation_off_a_refresh_changes_nothing():
    for algorithm in ("signals-channels", "signals-videos", "signals-mix"):
        assert _page(algorithm, 7, False) == _page(algorithm, 8, False), algorithm


def test_videos_starts_a_varying_page_from_one_of_a_subject_s_heaviest_videos():
    def starts(seed, vary):
        snap = snapshot("signals-videos", {"random_enabled": vary}, history=history())
        snap.context.seed = seed
        return [s.video_url for s in engine.plan(snap).sources if s.kind == "related"]

    fixed = starts(1, False)
    assert fixed == starts(2, False)
    seen = {tuple(starts(seed, True)) for seed in range(12)}
    assert len(seen) > 1
    watched_urls = {item.video_url for item in history()}
    assert all(url in watched_urls for page in seen for url in page)


# --- a subject the person hid, and what each algorithm keeps -------------------------------


def test_a_hidden_subject_is_not_asked_for_and_what_is_close_to_it_is_left_out():
    from recommender.contract import Overrides

    plain = snapshot("signals-videos", history=history())
    subjects = engine.profile(plain).groups
    camping = next(g for g in subjects if g.from_ and g.from_[0].startswith("history:camp"))
    assert not camping.muted
    hidden = snapshot(
        "signals-videos", history=history(), overrides=Overrides(muted_groups=[camping.id])
    )
    assert next(g for g in engine.profile(hidden).groups if g.id == camping.id).muted
    assert all(source.ref != camping.id for source in engine.plan(hidden).sources)
    candidates = [
        group(
            Source(kind="search", query="x", limit=20, ref="other"),
            [video("c1", "new1", title=f"{CAMPING} again", views=50_000)]
            + [
                video(f"k{i}", f"new{i}", title=f"{COOKING} more {i}", views=50_000)
                for i in range(5)
            ],
        )
    ]
    result = engine.rank(RankRequest(snapshot=hidden, candidates=candidates, trace=True))
    assert "https://www.youtube.com/watch?v=c1" not in urls(result)
    assert any(d.why == "muted" for d in result.trace.dropped)


def test_mix_keeps_the_creators_of_one_part_and_the_subjects_of_the_other():
    profile = engine.profile(snapshot("signals-mix", history=history()))
    assert {c.name for c in profile.channels} == {"wild", "chef"}
    assert profile.groups and all(g.from_ for g in profile.groups)


def test_a_setting_an_account_still_holds_from_a_removed_option_is_ignored():
    # Signals v2 was a setting until 2026-10-09: an account that had it on gets the same page.
    for algorithm in ("signals", "signals-videos", "signals-mix"):
        plain = engine.plan(snapshot(algorithm, history=history())).sources
        stored = engine.plan(snapshot(algorithm, {"v2": True}, history=history())).sources
        assert stored == plain
