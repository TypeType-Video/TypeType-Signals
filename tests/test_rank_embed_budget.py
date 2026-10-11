"""A page embeds at most RANK_EMBED_BUDGET new candidate titles; the rest in the background."""

from recommender.algorithms import signals_scoring as signals
from recommender.algorithms.signals_scoring import candidate_budget

from .test_embed_budget import ColdEmbedder

TEXTS = [f"video {i} — channel" for i in range(50)]


def test_only_the_budget_of_new_titles_is_embedded_now(monkeypatch):
    monkeypatch.setattr(signals, "RANK_EMBED_BUDGET", 40)
    now, later = candidate_budget(ColdEmbedder(warm=set()), TEXTS, muted=False)
    assert now == list(range(40))
    assert later == TEXTS[40:]


def test_titles_already_embedded_are_always_used(monkeypatch):
    monkeypatch.setattr(signals, "RANK_EMBED_BUDGET", 5)
    embedder = ColdEmbedder(warm={t.lower() for t in TEXTS[30:]})
    now, later = candidate_budget(embedder, TEXTS, muted=False)
    assert now == list(range(5)) + list(range(30, 50))
    assert later == TEXTS[5:30]


def test_a_title_waiting_for_the_background_is_not_embedded_again(monkeypatch):
    monkeypatch.setattr(signals, "RANK_EMBED_BUDGET", 5)
    embedder = ColdEmbedder(warm=set())
    embedder.prefetch(TEXTS[:10])
    now, later = candidate_budget(embedder, TEXTS, muted=False)
    assert now == list(range(10, 15))
    assert not set(later) & set(TEXTS[:10])


def test_with_a_muted_interest_every_candidate_is_checked(monkeypatch):
    monkeypatch.setattr(signals, "RANK_EMBED_BUDGET", 5)
    now, later = candidate_budget(ColdEmbedder(warm=set()), TEXTS, muted=True)
    assert now == list(range(50))
    assert later == []


def test_two_interests_about_the_same_topic_share_one_cap():
    import numpy as np

    from recommender.algorithms.signals_scoring import same_topic
    from recommender.signals_profile import Group

    def group(gid, vector):
        v = np.array(vector, dtype=float)
        return Group(id=gid, label=gid, members=[], centroid=v / np.linalg.norm(v), score=1.0)

    groups = [group("a", [1, 0, 0]), group("b", [0.95, 0.31, 0]), group("c", [0, 0, 1])]
    family = same_topic(groups)
    assert family["a"] == family["b"] == "a"
    assert family["c"] == "c"
