"""How Signals scores candidates: what is left out, then each score factor by factor."""

import math
from typing import TYPE_CHECKING

import numpy as np

from ..base import (
    Pool,
    Run,
    drop_unwanted,
    looks_short,
    score,
)
from ..contract_rank import Because, Candidate
from ..embeddings import Embedder
from ..signals_profile import MINUTE, Group
from .signals_constants import (
    DAY_MS,
    FRESHNESS_HALF_LIFE_DAYS,
    LATEST_HALF_LIFE_H,
    LATEST_SEED,
    MAX_CONTINUE,
    MUTED_SIMILARITY,
    PROVENANCE_FLOOR,
    RANK_EMBED_BUDGET,
    ROW_FLOOR,
    TOPIC_FLOOR,
)

if TYPE_CHECKING:
    from .signals import Signals

# A video nobody watches, stuffed with hashtags, from a channel the person never watched nor
# follows ("Top New Anime Funny Moments … #anime #fyp", 2 views, on a test page). On a real
# account's YouTube home, 3 of 3,098 regular video tiles had under 1,000 views and a hashtag
# (0.1 %), none under 100. Language-free:
# views, a "#", and whether the channel is known.
OBSCURE_VIEWS = 1_000


def obscure(video: Candidate, known_channels: set[str]) -> bool:
    return (
        video.view_count is not None
        and video.view_count < OBSCURE_VIEWS
        and "#" in video.title
        and video.uploader_url not in known_channels
    )


# Two interests this close are one topic split in two ("France & français", "France &
# paraguay" on a test page, the same matches in two rows): they share the cap per interest.
# Measured on three real profiles:
# at 0.80 and above only pairs about the same thing; at 0.75-0.78 also a mixed group joined to
# unrelated ones.
SAME_TOPIC = 0.80


def same_topic(groups: list[Group]) -> dict[str, str]:
    """Each group's topic: the id of the first of the groups it is this close to, step by step."""
    if len(groups) < 2:
        return {}
    centres = np.array([g.centroid for g in groups])
    close = centres @ centres.T >= SAME_TOPIC
    root = list(range(len(groups)))

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(close, 1)), strict=True):
        root[find(int(j))] = find(int(i))
    return {g.id: groups[find(i)].id for i, g in enumerate(groups)}


def candidate_budget(
    embedder: Embedder, texts: list[str], muted: bool
) -> tuple[list[int], list[str]]:
    """Rows to embed now, and texts to embed in the background. With a muted interest every
    candidate is embedded now: one close to it must be dropped, never let through unseen."""
    known = getattr(embedder, "known", None)
    if known is None or muted:
        return list(range(len(texts))), []
    ready, waiting = known(texts), embedder.queued(texts)
    cold = [i for i in range(len(texts)) if not ready[i] and not waiting[i]]
    now = sorted({i for i in range(len(texts)) if ready[i]} | set(cold[:RANK_EMBED_BUDGET]))
    return now, list(dict.fromkeys(texts[i] for i in cold[RANK_EMBED_BUDGET:]))


def _strongest(reasons: list[Because]) -> list[Because]:
    """Each reason once, at its strongest, strongest first."""
    strongest: dict[tuple[str, str | None], Because] = {}
    for reason in reasons:
        key = (reason.kind, reason.ref)
        if key not in strongest or reason.weight > strongest[key].weight:
            strongest[key] = reason
    return sorted(strongest.values(), key=lambda b: -b.weight)


class Scoring:
    """The candidates of one page, cleaned then scored: what the rows are made from."""

    def __init__(self, algorithm: "Signals", run: Run):
        self.algorithm = algorithm
        self.run = run
        self.snapshot, self.values = self.run.snapshot, self.run.values
        self.conclusions = self.algorithm.conclude(self.snapshot, self.values)
        self.rng = self.run.rng(self.values.flag("random_enabled"))

    def scored(self) -> "Scoring":
        """The same object once every candidate is cleaned and scored."""
        self._clean()
        self._score()
        return self

    def _clean(self) -> None:
        """Leaves out what must not be shown, and sets aside the videos to resume or watch again."""
        self.pool = Pool.gather(self.run.groups)
        again_urls = {item.video_url for item in self.conclusions.rewatched}
        kept_again = {
            u: (self.pool.videos[u], self.pool.sources[u])
            for u in self.pool.urls()
            if u in again_urls and any(src.ref == "watch_again" for src in self.pool.sources[u])
        }
        drop_unwanted(self.run, self.pool, seen=True, shorts=False)
        for url, (video, from_sources) in kept_again.items():  # seen on purpose: their own row
            if (
                url not in self.pool.videos
                and video.uploader_url not in set(self.snapshot.overrides.muted_channels)
                and not looks_short(video)
                and url not in {item.video_url for item in self.snapshot.not_interested}
            ):
                self.pool.videos[url], self.pool.sources[url] = video, from_sources
        for url in self.pool.urls():
            if obscure(self.pool.videos[url], self.conclusions.known_channels):
                self.run.drop(url, "obscure")
                self.pool.remove(url)

        self.continue_urls = [
            item.video_url
            for item in self.conclusions.started
            if item.video_url in self.pool.videos
        ][:MAX_CONTINUE]
        self.watch_again_urls = [
            item.video_url
            for item in self.conclusions.rewatched
            if item.video_url in kept_again
            and item.video_url in self.pool.videos
            and item.video_url not in set(self.continue_urls)
        ]
        self.ranked = [
            u
            for u in self.pool.urls()
            if u not in set(self.continue_urls) | set(self.watch_again_urls)
        ]

    def _score(self) -> None:
        """Scores every candidate factor by factor, and notes what it is close to and why."""
        self.boosts = {b.item.id: b for b in self.conclusions.boosts}
        self.latest = {i.id: i for i in self.conclusions.latest}
        self.groups = {g.id: g for g in self.conclusions.groups}
        self.scorable = [g for g in self.conclusions.groups if not g.ignored]
        self.top_group = max((g.weight_now for g in self.conclusions.groups), default=0) or 1
        top_channel = max((ch.weight for ch in self.conclusions.channels), default=0) or 1
        self.lasting = self.values.number("long_term_min_watched")
        self.now = self.snapshot.context.now
        self.column = {g.id: j for j, g in enumerate(self.scorable)}
        self.anchor_of = {item.video_url: item.id for item in self.snapshot.history}
        self.similarity = self._similarity()

        self.totals: dict[str, float] = {}
        self.seed_of: dict[str, str] = {}
        self.group_of: dict[str, Group] = {}
        # The interest a video is about, named or not, for the cap per interest.
        self.topic_of: dict[str, Group] = {}
        self.family = same_topic(self.scorable)
        self.because: dict[str, list[Because]] = {}
        for index, url in enumerate(self.ranked):
            video = self.pool.videos[url]
            best, closeness = self._closest(index)
            if best and best.muted and closeness >= MUTED_SIMILARITY:
                self.run.drop(url, "muted")
                self.pool.remove(url)
                continue

            seed, seed_ref = self._seed(index, url)
            channel = self.conclusions.channel(video.uploader_url)
            same = channel.weight / top_channel if channel and not channel.ignored else 0.0
            topic = 0.0
            if best and not best.muted and best.weight > 0 and closeness >= TOPIC_FLOOR:
                topic = (closeness - 0.35) / 0.65 * best.weight_now / self.top_group
                if (best.named or best.pinned) and closeness >= ROW_FLOOR:
                    self.group_of.setdefault(url, best)
            if best and closeness >= ROW_FLOOR:
                self.topic_of[url] = best
            factors = self._factors(video, seed, same, topic)
            self.totals[url] = score(self.run, url, factors)
            if seed_ref:
                self.seed_of[url] = seed_ref
            reasons = self._seed_reasons(url, seed_ref, factors)
            if same:
                weight = round(factors["same_channel"], 4)
                reasons.append(Because(kind="same_channel", ref=channel.id, weight=weight))
            if topic:
                reasons.append(
                    Because(kind="topic", ref=best.id, weight=round(factors["topic"], 4))
                )
            self.because[url] = _strongest(reasons)

        self.ranked = [u for u in self.ranked if u in self.pool.videos]

    def _closest(self, index: int) -> tuple[Group | None, float]:
        """The interest a candidate's title is closest to, and how close."""
        if not self.scorable:
            return None, 0.0
        j = int(np.argmax(self.similarity[index]))
        return self.scorable[j], float(self.similarity[index, j])

    def _similarity(self) -> np.ndarray:
        """How close each candidate's title is to each interest: read now within the budget,
        the rest in the background for the next page."""
        similarity = np.zeros((len(self.ranked), len(self.scorable)))
        if not self.ranked or not self.scorable:
            return similarity
        texts = [
            f"{self.pool.videos[u].title} — {self.pool.videos[u].uploader_name}"
            for u in self.ranked
        ]
        now_rows, later = candidate_budget(
            self.algorithm.embedder, texts, any(g.muted for g in self.scorable)
        )
        if now_rows:
            vectors = self.algorithm.embedder.embed([texts[i] for i in now_rows])
            similarity[now_rows] = vectors @ np.array([g.centroid for g in self.scorable]).T
        if later:
            self.algorithm.embedder.prefetch(later)
        return similarity

    def _seed(self, index: int, url: str) -> tuple[float, str | None]:
        """How strongly what brought the candidate still counts, and what that was. A candidate
        brought for an interest it is close to is noted as belonging to that interest."""
        seed, seed_ref = 0.0, None
        for source in self.pool.sources[url]:
            value = 0.0
            if source.kind == "related" and source.ref in self.boosts:
                value = self.boosts[source.ref].strength
            elif source.kind == "related" and source.ref in self.latest:
                age_h = (self.now - self.latest[source.ref].watched_at) / (60 * MINUTE)
                value = LATEST_SEED * 0.5 ** (age_h / LATEST_HALF_LIFE_H)
            elif source.kind in ("related", "search") and source.ref in self.groups:
                group = self.groups[source.ref]
                j = self.column.get(group.id)
                close = j is not None and self.similarity[index, j] >= PROVENANCE_FLOOR
                value = 0.5 * group.weight_now / self.top_group
                if close and not group.muted and (group.named or group.pinned):
                    self.group_of[url] = group
            elif source.kind == "channel_videos":
                # A full watch brings its channel back (on YouTube, 2 to 5 videos of the
                # channel after one full watch): the catalogue of a channel just watched
                # through counts like related videos.
                value = max(
                    (
                        b.strength
                        for b in self.conclusions.boosts
                        if b.item.channel_url == source.channel_url
                        and (b.item.watched_share or 0) >= self.lasting
                    ),
                    default=0.0,
                )
            if value > seed:
                seed, seed_ref = value, source.ref
        return seed, seed_ref

    def _factors(self, video: Candidate, seed: float, same: float, topic: float) -> dict:
        """The parts of a score, each weighted by its setting."""
        fresh = 0.0
        if video.uploaded_at:
            age_days = max(0, self.now - video.uploaded_at) / DAY_MS
            fresh = 0.5 ** (age_days / FRESHNESS_HALF_LIFE_DAYS)
        popular = min(1.0, math.log10((video.view_count or 0) + 1) / 8)
        popularity = self.values.number("popularity_bonus") - self.values.number(
            "popularity_penalty"
        )
        factors = {
            "watch_seed": self.values.number("w_watch_seed") * seed,
            "same_channel": self.values.number("w_same_channel") * same,
            "topic": self.values.number("w_topic") * topic,
            "freshness": self.values.number("w_freshness") * fresh,
            "popularity": popularity * popular,
        }
        if self.values.flag("random_enabled"):
            factors["draw"] = self.rng.uniform(0, self.values.number("draw_weight"))
        return factors

    def _seed_reasons(self, url: str, seed_ref: str | None, factors: dict) -> list[Because]:
        """Why the candidate is here when something watched brought it."""
        if not seed_ref:
            return []
        kind, ref = "watched", seed_ref
        if seed_ref.startswith("channel:"):
            kind = "same_channel"
        elif seed_ref in self.groups:
            # Related to the group's anchor video: name that video, it is what was watched.
            source = next(s for s in self.pool.sources[url] if s.ref == seed_ref)
            ref = self.anchor_of.get(source.video_url or "", seed_ref)
        return [Because(kind=kind, ref=ref, weight=round(factors["watch_seed"], 4))]
