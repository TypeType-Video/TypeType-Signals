"""A first page after an import embeds at most EMBED_BUDGET new titles, the rest later."""

from recommender import signals_conclude, signals_groups
from recommender.algorithms.signals import Signals
from recommender.contract import VideoRef
from recommender.embeddings import HashingEmbedder
from recommender.parameters import Values

from .factories import HOUR, NOW, channel, snapshot


class ColdEmbedder(HashingEmbedder):
    """Knows the titles listed as warm; records what it is asked to embed later."""

    def __init__(self, warm: set[str]):
        super().__init__()
        self.warm = warm
        self.prefetched: list[str] = []

    def known(self, texts):
        return [t.lower() in self.warm for t in texts]

    def queued(self, texts):
        return [t in self.prefetched for t in texts]

    def prefetch(self, texts):
        self.prefetched += texts


def likes(count: int) -> list[VideoRef]:
    return [
        VideoRef(
            id=f"favorite:{i}",
            video_url=f"https://www.youtube.com/watch?v=v{i}",
            channel_url=channel(f"c{i % 3}"),
            title=f"Winter camping shelter forest trip {i}",
            added_at=int(NOW - i * HOUR),
        )
        for i in range(count)
    ]


def clustered(embedder, favorites) -> set[str]:
    request = snapshot("signals", favorites=favorites)
    c = signals_conclude.conclude(request, Values(Signals.specs, {}), embedder)
    return {ref for g in c.groups for ref in g.members}


def test_only_the_newest_new_titles_are_embedded_now(monkeypatch):
    monkeypatch.setattr(signals_groups, "EMBED_BUDGET", 4)
    embedder = ColdEmbedder(warm=set())
    members = clustered(embedder, likes(10))
    assert members <= {f"favorite:{i}" for i in range(4)}
    assert len(embedder.prefetched) == 6


def test_titles_already_embedded_are_always_used(monkeypatch):
    monkeypatch.setattr(signals_groups, "EMBED_BUDGET", 2)
    favorites = likes(10)
    warm = {f.title.lower() for f in favorites[6:]}
    embedder = ColdEmbedder(warm=warm)
    members = clustered(embedder, favorites)
    assert {f"favorite:{i}" for i in range(6, 10)} <= members
    assert len(embedder.prefetched) == 4


def test_nothing_changes_once_every_title_is_ready(monkeypatch):
    monkeypatch.setattr(signals_groups, "EMBED_BUDGET", 2)
    favorites = likes(10)
    everything = ColdEmbedder(warm={f.title.lower() for f in favorites})
    assert clustered(everything, favorites) == clustered(HashingEmbedder(), favorites)
    assert everything.prefetched == []


def test_a_title_waiting_for_the_background_is_not_embedded_again(monkeypatch):
    monkeypatch.setattr(signals_groups, "EMBED_BUDGET", 4)
    embedder = ColdEmbedder(warm=set())
    favorites = likes(10)
    clustered(embedder, favorites)
    first = list(embedder.prefetched)
    members = clustered(embedder, favorites)
    assert embedder.prefetched == first
    assert members <= {f"favorite:{i}" for i in range(4)}
