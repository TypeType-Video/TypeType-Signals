"""What a person watches, by subject: the profile of the algorithms that aim at videos.

A subject is a group of watched videos about the same thing. Each video is read through what
it says about itself: its tags and category when the catalogue has them, its title otherwise.
Shorts are left out (they say nothing about what someone watches in long form), as are removed
videos. Any language: nothing is read word by word except to name a subject.

Bounded whatever the history: the newest MAX_VIDEOS long-form videos, at most EMBED_BUDGET
texts embedded in a request (the rest in the background, ready for the next one).
"""

import threading
import weakref
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .catalogue import Catalogue, video_id
from .contract import Snapshot
from .embeddings import Embedder
from .kept import Kept, shared_key
from .labels import group_words, informative, topic_text
from .signals_clusters import average_linkage
from .signals_history import _unavailable, inferred_shorts

DAY_MS = 86_400_000
MAX_VIDEOS = 400
EMBED_BUDGET = 60
# Videos this close (cosine), on average, are about the same subject.
SUBJECT_SIMILARITY = 0.55
MAX_SUBJECTS = 12
# A watch counts less as it gets older; a like counts as half a watch seen through.
HALF_LIFE_DAYS = 60
LIKE_WEIGHT = 0.5
UNKNOWN_PROGRESS = 0.5
QUERY_CHARS = 80
SEED_VIDEOS = 3
MAX_MEMBER_REFS = 50


@dataclass
class Subject:
    id: str
    label: str
    centroid: np.ndarray
    weight: float  # its share of the person's subjects, 0 to 1
    query: str  # what to search for
    seed_url: str  # its heaviest video, to start related videos from
    videos: int
    # Its few heaviest videos, heaviest first: a page that varies starts from one of them.
    seed_urls: tuple[str, ...] = ()
    # The watches and likes it was made from (their references), heaviest first.
    members: tuple[str, ...] = ()
    # The person hid it: nothing is asked for it, and what is close to it is left out.
    muted: bool = False


_kept: list[tuple[weakref.ref, list[Subject]]] = []
# Beyond one page: kept under the history's key while the history has not changed.
_shared = Kept()
_kept_lock = threading.Lock()


def subjects(snapshot: Snapshot, embedder: Embedder, catalogue: Catalogue) -> list[Subject]:
    """The person's subjects, heaviest first. A page's plan and rank share one snapshot
    object, so they are worked out once."""
    with _kept_lock:
        for kept, result in _kept:
            if kept() is snapshot:
                return result
    shared = shared_key(snapshot)
    result = _shared.get(shared)
    if result is None:
        result = _subjects(snapshot, embedder, catalogue)
        _shared.put(shared, result, embedder)
    with _kept_lock:
        alive = [entry for entry in _kept if entry[0]() is not None]
        _kept[:] = [*alive[-3:], (weakref.ref(snapshot), result)]
    return result


class SubjectFinder:
    """The subjects of one person, found step by step from what they watched and liked."""

    def __init__(self, snapshot: Snapshot, embedder: Embedder, catalogue: Catalogue):
        self.snapshot = snapshot
        self.embedder = embedder
        self.catalogue = catalogue

    def subjects(self) -> list[Subject]:
        if (ended := self._watched()) is not None:
            return ended
        if (ended := self._entries()) is not None:
            return ended
        self._cluster()
        return self._build()

    def _watched(self) -> list[Subject] | None:
        """The videos that count, each with its weight: watches that fade, and likes."""
        now = self.snapshot.context.now
        shorts = inferred_shorts(self.snapshot.history)
        self.watched: dict[str, tuple[str, str, float]] = {}  # url -> (title, channel name, weight)
        self.refs: dict[str, str] = {}  # url -> the reference of the watch or like
        for item in sorted(self.snapshot.history, key=lambda i: -i.watched_at):
            if item.is_short or item.id in shorts or _unavailable(item) or not item.title:
                continue
            if item.video_url in self.watched:
                continue
            share = UNKNOWN_PROGRESS if item.watched_share is None else item.watched_share
            age_days = max(0, now - item.watched_at) / DAY_MS
            self.watched[item.video_url] = (
                item.title,
                item.channel_name,
                share * 0.5 ** (age_days / HALF_LIFE_DAYS),
            )
            self.refs[item.video_url] = item.id
            if len(self.watched) >= MAX_VIDEOS:
                break
        for favorite in self.snapshot.favorites:
            if favorite.is_short or not favorite.title or favorite.video_url in self.watched:
                continue
            age_days = max(0, now - (favorite.added_at or now)) / DAY_MS
            self.watched[favorite.video_url] = (
                favorite.title,
                "",
                LIKE_WEIGHT * 0.5 ** (age_days / HALF_LIFE_DAYS),
            )
            self.refs[favorite.video_url] = favorite.id
        if not self.watched:
            return []
        return None

    def _entries(self) -> list[Subject] | None:
        """What each video is read by, heaviest first, within the budget of titles to read now."""
        facts = self.catalogue.get([vid for url in self.watched if (vid := video_id(url))])
        # (url, text to embed, title, channel name, weight, tags): one per video of the whole
        # history, kept a plain tuple for the same measured reason as in signals_conclude.py.
        self.entries: list[tuple] = []
        for url, (title, channel, weight) in self.watched.items():
            known = facts.get(video_id(url) or "")
            if known:
                self.entries.append(
                    (url, known.text() or title, title, channel, weight, known.tags)
                )
            elif informative(title):
                self.entries.append((url, topic_text(title), title, channel, weight, ()))
        self.entries.sort(key=lambda e: -e[4])
        self.entries = self.entries[:MAX_VIDEOS]
        self.texts = [e[1] for e in self.entries]
        known_texts = getattr(self.embedder, "known", None)
        if known_texts:
            # Only what is embedded already, plus the heaviest of the rest: a first page stays fast.
            ready = known_texts(self.texts)
            budget = EMBED_BUDGET
            keep = []
            for is_ready in ready:
                keep.append(is_ready or budget > 0)
                budget -= 0 if is_ready else 1
            later = [t for t, kept in zip(self.texts, keep, strict=True) if not kept]
            if later and hasattr(self.embedder, "prefetch"):
                self.embedder.prefetch(later)
            self.entries = [e for e, kept in zip(self.entries, keep, strict=True) if kept]
            self.texts = [e[1] for e in self.entries]
        if not self.entries:
            return []
        return None

    def _cluster(self) -> None:
        """Videos whose texts mean the same thing, grouped: the heaviest groups are the subjects."""
        self.vectors = self.embedder.embed(self.texts)
        clusters = average_linkage(self.vectors, SUBJECT_SIMILARITY)
        clusters.sort(key=lambda members: -sum(self.entries[i][4] for i in members))
        # A subject is at least two videos; someone with a short history keeps single videos.
        self.grouped = [c for c in clusters if len(c) >= 2] or clusters
        self.grouped = self.grouped[:MAX_SUBJECTS]
        self.total = sum(self.entries[i][4] for members in self.grouped for i in members) or 1.0
        self.all_titles = [e[2] for e in self.entries]

        # A hidden subject is named by one of its videos, so the decision follows it as it grows.
        self.hidden = {
            ref.split(":", 1)[1]
            for ref in self.snapshot.overrides.muted_groups
            if ref.startswith("subject:")
        }

    def _build(self) -> list[Subject]:
        """Each subject with its name, its weight and what a search for it asks."""
        result = []
        for members in self.grouped:
            weights = np.array([self.entries[i][4] for i in members])
            centroid = (self.vectors[members] * weights[:, None]).sum(axis=0)
            norm = float(np.linalg.norm(centroid))
            centroid = centroid / norm if norm else self.vectors[members[0]]
            heaviest = max(members, key=lambda i: self.entries[i][4])
            tags: Counter[str] = Counter()
            for i in members:
                for tag in self.entries[i][5]:
                    tags[tag.strip().lower()] += self.entries[i][4]
            top_tags = [tag for tag, _ in tags.most_common(2) if tag]
            label, _, words = group_words(
                [self.entries[i][2] for i in members],
                self.all_titles,
                [self.entries[i][3] for i in members],
            )
            if top_tags:
                # The tags creators gave their own videos: the words people search this subject by.
                query, label = " ".join(top_tags), " · ".join(t.capitalize() for t in top_tags)
            else:
                query = (
                    " ".join(words) if len(words) >= 2 else topic_text(self.entries[heaviest][2])
                )
            result.append(
                Subject(
                    id=f"subject:{video_id(self.entries[heaviest][0]) or len(result)}",
                    label=label,
                    centroid=centroid,
                    weight=float(weights.sum() / self.total),
                    query=query[:QUERY_CHARS],
                    seed_url=self.entries[heaviest][0],
                    videos=len(members),
                    members=tuple(
                        self.refs[self.entries[i][0]]
                        for i in sorted(members, key=lambda i: -self.entries[i][4])
                        if self.entries[i][0] in self.refs
                    )[:MAX_MEMBER_REFS],
                    muted=bool(self.hidden)
                    and (
                        str(video_id(self.entries[heaviest][0]) or len(result)) in self.hidden
                        or any(video_id(self.entries[i][0]) in self.hidden for i in members)
                    ),
                    seed_urls=tuple(
                        self.entries[i][0]
                        for i in sorted(members, key=lambda i: -self.entries[i][4])[:SEED_VIDEOS]
                    ),
                )
            )
        return result


def _subjects(snapshot: Snapshot, embedder: Embedder, catalogue: Catalogue) -> list[Subject]:
    return SubjectFinder(snapshot, embedder, catalogue).subjects()
