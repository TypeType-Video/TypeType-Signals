"""Signals: a home page that follows what you actually watch.

From the history (and favorites) it concludes three things, all shown on
the "what it remembers" screen (see signals_profile.py): the channels you
watch, interest groups made of videos whose titles mean the same thing,
and the videos you just watched, whose effect fades by a half-life.

It then asks Server for: the videos you started and did not finish,
videos related to what you just watched and to one video of each interest
group, popular videos of your main channels, your subscription feed and
trending videos. Every candidate gets a score, factor by factor:

    watch_seed    how strongly the video it came from still counts
    same_channel  how much you watch its channel
    topic         how close its title is to one of your interest groups
    freshness     how recent it is (halves every 14 days)
    popularity    a small penalty for huge view counts, to leave room for
                  smaller creators
    draw          a small random part, only when randomness is on

The page is then shared out: videos you started, one row per video just
watched (bigger while its effect is strong), `subscriptions_share` of the
page for subscriptions, `discovery_share` for channels you never watched
nor followed, `trending_share` for trends, and the rest for your interest
groups, pinned ones first. Within each part the best scores win, with at
most `max_per_channel` videos per channel on the page.
"""

import random
import threading
import weakref

import numpy as np

from ..base import (
    Algorithm,
    Run,
)
from ..contract import HistoryItem, Snapshot, Source, VideoRef
from ..contract_algorithms import Traits
from ..contract_profile import Profile, ProfileChannel, ProfileGroup, SessionBoost, TopicMatch
from ..contract_rank import Row
from ..embeddings import Embedder, default_embedder
from ..kept import Kept, shared_key
from ..labels import informative, topic_text
from ..parameters import Values
from ..signals_clusters import CUSTOM_PREFIX
from ..signals_conclude import conclude
from ..signals_profile import MAX_CHANNELS, MAX_TITLES, Conclusions
from .signals_constants import (
    BOOST_SEEDS,
    CATALOGUE_CHANNELS,
    CONCLUSIONS_KEPT,
    CORE_CHANNELS,
    CORE_GROUPS,
    CORE_TOPICS,
    CUSTOM_SEARCH,
    FEED_LIMIT,
    GROUP_SEEDS,
    MAX_CONTINUE,
    ROTATION_POOL,
    ROW_FLOOR,
    SEARCH_TOPICS,
    TRENDING_LIMIT,
    WARM_MIN_WATCHED,
)
from .signals_ranking import rank_page
from .signals_specs import SPECS


class Rotation:
    """Picks `total` items: the first `core` always, the rest drawn from the next ones.

    YouTube shows its core topics at every refresh and rotates occasional ones. With
    randomness off, the next ones are
    taken in order, so the same data always asks for the same sources.
    """

    def __init__(self, seed: int | None, randomized: bool):
        self._rng = random.Random(seed)
        self._randomized = randomized

    def __call__(self, items: list, core: int, total: int) -> list:
        head, rest = items[:core], items[core : core + total * ROTATION_POOL]
        extra = total - len(head)
        if extra <= 0:
            return items[:total]
        if not self._randomized or len(rest) <= extra:
            return head + rest[:extra]
        picked = set(self._rng.sample(range(len(rest)), extra))
        return head + [item for i, item in enumerate(rest) if i in picked]


class Signals(Algorithm):
    id = "signals"
    name = {"en": "Signals", "fr": "Signals"}
    description = {
        "en": "Reacts to what you watch: videos you started, more from creators you watched, rows "
        "by topic and one row of creators you never watched.",
        "fr": "Suit l'historique : vidéos commencées, autres vidéos des créateurs regardés, lignes "
        "par sujet et une ligne de créateurs jamais vus.",
    }
    uses = ["history", "subscriptions", "favorites", "not_interested", "shown"]
    sources = ["videos", "related", "search", "channel_videos", "subscription_feed", "trending"]
    random = True
    traits = Traits(personal=3, discovery=2, variety=2)
    guide = {
        "for_you_if": {
            "en": "You want a home page that follows what you actually watch, including what you "
            "just watched.",
            "fr": "Vous voulez un accueil qui suit ce que vous regardez vraiment, y compris ce que "
            "vous venez de regarder.",
        },
        "you_will_see": {
            "en": "Videos you started, a row for what you just watched while its effect lasts, "
            "more from the creators you watch, rows per topic with popular videos of that "
            "topic, and a share of creators you never watched. Some topics rotate from one "
            "refresh to the next.",
            "fr": "Vos vidéos commencées, une ligne pour ce que vous venez de regarder tant que "
            "son effet dure, d'autres vidéos des créateurs regardés, des lignes par sujet avec "
            "des vidéos populaires du sujet, et une part de créateurs jamais vus. Certains "
            "sujets changent d'un rafraîchissement à l'autre.",
        },
        "good_to_know": {
            "en": "It reads your history, so it is the most personal and the one with the most "
            "settings. Everything it deduces is shown on this page and can be corrected.",
            "fr": "Il lit votre historique : c'est le plus personnel et le plus réglable. Tout ce "
            "qu'il en déduit est visible sur cette page et corrigeable.",
        },
    }
    specs = SPECS

    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder
        # The conclusions of the last snapshots, by object: a page's plan and rank share one
        # snapshot object (engine plan_id), so its profile is built once. The snapshot is held
        # weakly: once the engine lets its plan go, its whole history is freed.
        self._concluded: list[tuple[weakref.ref, tuple, Conclusions]] = []
        self._concluded_lock = threading.Lock()
        # Beyond one page: kept under the history's key while the history has not changed.
        self._kept = Kept()

    @property
    def embedder(self) -> Embedder:
        return self._embedder or default_embedder()

    def conclude(self, snapshot: Snapshot, values: Values) -> Conclusions:
        key = values.key()
        with self._concluded_lock:
            for kept, kept_key, conclusions in self._concluded:
                if kept() is snapshot and kept_key == key:
                    return conclusions
        shared = shared_key(snapshot, key)
        conclusions = self._kept.get(shared)
        if conclusions is None:
            conclusions = conclude(snapshot, values, self.embedder)
            self._kept.put(shared, conclusions, self.embedder)
        with self._concluded_lock:
            alive = [entry for entry in self._concluded if entry[0]() is not None]
            self._concluded = [
                *alive[-(CONCLUSIONS_KEPT - 1) :],
                (weakref.ref(snapshot), key, conclusions),
            ]
        return conclusions

    def warm(self, history: list[HistoryItem], favorites: list[VideoRef]) -> None:
        """A history just sent (/v1/history): the titles a profile clusters are embedded in
        the background, newest first, so the first page finds them ready instead of spending
        its budget on them. Best effort: watches seen through and likes, Shorts left out."""
        known = getattr(self.embedder, "known", None)
        if known is None:
            return
        dated = [
            (h.watched_at, h.title)
            for h in history
            if h.title and not h.is_short and (h.watched_share or 0) >= WARM_MIN_WATCHED
        ]
        dated += [(f.added_at or 0, f.title) for f in favorites if f.title and not f.is_short]
        texts: list[str] = []
        for _, title in sorted(dated, key=lambda d: -d[0]):
            if informative(title):
                texts.append(topic_text(title))
                if len(texts) == MAX_TITLES:
                    break
        ready, waiting = known(texts), self.embedder.queued(texts)
        cold = [t for t, r, w in zip(texts, ready, waiting, strict=True) if not r and not w]
        if cold:
            self.embedder.prefetch(cold)

    # --- Plan ------------------------------------------------------------------------

    def plan(self, snapshot: Snapshot, values: Values) -> list[Source]:
        c = self.conclude(snapshot, values)
        rotate = Rotation(snapshot.context.seed, values.flag("random_enabled"))
        sources: list[Source] = []
        if c.started:
            started = [item.video_url for item in c.started[: MAX_CONTINUE * 2]]
            sources.append(Source(kind="videos", urls=started, limit=len(started), ref="continue"))
        if c.rewatched:
            again = [item.video_url for item in c.rewatched]
            sources.append(Source(kind="videos", urls=again, limit=len(again), ref="watch_again"))
        sources += self._related_sources(c, snapshot, values.whole("related_per_seed"), rotate)
        sources += self._search_sources(c, values.whole("search_per_topic"), rotate)
        sources += self._catalogue_sources(c, snapshot, values, rotate)
        if snapshot.subscriptions and values.number("subscriptions_share") > 0:
            sources.append(Source(kind="subscription_feed", limit=FEED_LIMIT))
        if values.number("trending_share") > 0:
            sources.append(Source(kind="trending", limit=TRENDING_LIMIT))
        return sources

    @staticmethod
    def _related_sources(
        c: Conclusions, snapshot: Snapshot, related: int, rotate: Rotation
    ) -> list[Source]:
        """Videos related to what was just watched, and to one video of each interest."""
        if not related:
            return []
        urls = {item.id: item.video_url for item in snapshot.history} | {
            f.id: f.video_url for f in snapshot.favorites
        }
        sources = []
        asked: set[str] = set()
        recent = [b.item for b in c.boosts[:BOOST_SEEDS]]
        recent += [i for i in c.latest if i.video_url not in {r.video_url for r in recent}]
        for item in recent:
            asked.add(item.video_url)
            sources.append(
                Source(kind="related", video_url=item.video_url, limit=related, ref=item.id)
            )
        groups = sorted(
            (g for g in c.groups if not g.muted and not g.ignored),
            key=lambda g: (not g.pinned, -g.weight_now),
        )
        for group in rotate(groups, CORE_GROUPS, GROUP_SEEDS):
            anchor = next((urls[m] for m in group.members if urls.get(m) not in asked), None)
            if anchor:
                asked.add(anchor)
                sources.append(
                    Source(kind="related", video_url=anchor, limit=related, ref=group.id)
                )
        return sources

    @staticmethod
    def _search_sources(c: Conclusions, per_topic: int, rotate: Rotation) -> list[Source]:
        """A search by the words of the subjects the person added, and of a few interests."""
        # A subject the person added has nothing watched to start from: it is always searched.
        sources = [
            Source(
                kind="search",
                query=" ".join(group.keywords),
                limit=max(per_topic, CUSTOM_SEARCH),
                ref=group.id,
            )
            for group in c.groups
            if group.id.startswith(CUSTOM_PREFIX)
        ]
        if not per_topic:
            return sources
        topics = sorted(
            (
                g
                for g in c.groups
                if not g.muted
                and not g.ignored
                and g.named
                and g.keywords
                and not g.id.startswith(CUSTOM_PREFIX)
            ),
            key=lambda g: (not g.pinned, -g.weight_now),
        )
        for group in rotate(topics, CORE_TOPICS, SEARCH_TOPICS):
            query = " ".join(group.keywords)
            sources.append(Source(kind="search", query=query, limit=per_topic, ref=group.id))
        return sources

    @staticmethod
    def _catalogue_sources(
        c: Conclusions, snapshot: Snapshot, values: Values, rotate: Rotation
    ) -> list[Source]:
        """Popular videos of the main channels, those just watched through first."""
        catalogue = values.whole("catalogue_per_channel")
        if not catalogue:
            return []
        muted = set(snapshot.overrides.muted_channels)
        lasting = values.number("long_term_min_watched")
        boosted = [b.item.channel_url for b in c.boosts if (b.item.watched_share or 0) >= lasting]
        ordered = boosted + [ch.url for ch in c.channels if not ch.ignored]
        candidates = [u for u in dict.fromkeys(ordered) if u not in muted]
        sources = []
        for url in rotate(candidates, len(boosted) + CORE_CHANNELS, CATALOGUE_CHANNELS):
            channel = c.channel(url)
            sources.append(
                Source(
                    kind="channel_videos",
                    channel_url=url,
                    sort="popular",
                    limit=catalogue,
                    ref=channel.id if channel else None,
                )
            )
        return sources

    # --- Rank ------------------------------------------------------------------------

    def rank(self, run: Run) -> list[Row]:
        return rank_page(self, run)

    def topics(self, snapshot: Snapshot, values: Values, titles: list[str]) -> list[TopicMatch]:
        c = self.conclude(snapshot, values)
        groups = [g for g in c.groups if not g.ignored]
        if not groups or not titles:
            return [TopicMatch(group=None, label=None, similarity=0.0) for _ in titles]
        similarity = self.embedder.embed(titles) @ np.array([g.centroid for g in groups]).T
        matches = []
        for row in similarity:
            j = int(np.argmax(row))
            close = float(row[j]) >= ROW_FLOOR
            matches.append(
                TopicMatch(
                    group=groups[j].id if close else None,
                    label=groups[j].label if close else None,
                    similarity=round(float(row[j]), 4),
                )
            )
        return matches

    # --- Profile ---------------------------------------------------------------------

    def profile(self, snapshot: Snapshot, values: Values) -> Profile:
        c = self.conclude(snapshot, values)
        return Profile(
            groups=[
                ProfileGroup(
                    id=g.id,
                    label=g.label,
                    weight=round(g.weight, 4),
                    hour_factor=round(g.hour_factor, 3),
                    from_=g.members,
                    pinned=g.pinned,
                    muted=g.muted,
                )
                for g in c.groups
            ],
            channels=[
                ProfileChannel(id=ch.id, name=ch.name, weight=round(ch.weight, 4), from_=ch.refs)
                for ch in c.channels[:MAX_CHANNELS]
            ],
            session_boosts=[
                SessionBoost(
                    ref=b.item.id,
                    label=b.item.title,
                    strength=round(b.strength, 4),
                    expires_at=b.expires_at,
                )
                for b in c.boosts
            ],
        )
