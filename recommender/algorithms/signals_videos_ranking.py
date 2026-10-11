"""How Signals by videos turns candidates into a page, step by step."""

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from ..base import Pool, Run, drop_unwanted, make_row, score
from ..catalogue import video_id
from ..contract_rank import Because, Reason, Row

if TYPE_CHECKING:
    from .signals_videos import SignalsVideos

# New candidate titles embedded at most per page; the others wait for the background and are
# placed by the subject their source was asked for.
EMBED_BUDGET = 40
# A candidate this close (cosine) to a subject belongs to its row.
ROW_FLOOR = 0.45
# What a candidate not embedded yet is worth: just under a real match.
UNREAD = 0.4
# A candidate this close to a subject the person hid is left out (as in Signals).
MUTED = 0.55
SUBJECT_ROW_MIN = 4
SUBJECT_ROW_MAX = 16
POPULARITY_BONUS = 0.05
# What a creator who comes back several times among the candidates adds, and the views a
# video needs. Both held for Signals' new creators row; here, on the page bench (30 cuts):
# known channels then watched +0.67 point [+0.23, +1.17], real discoveries +0.47
# [+0.00, +0.92].
ECHO_BONUS = 0.15
MIN_VIEWS = 10_000
# How much chance weighs against the score when the page varies on refresh: enough to
# change which videos sit at the edge of the page, not enough to bring in a far worse one.
DRAW = 0.1
# What a candidate brought by a video's related videos adds over one brought by a search. On
# the page bench: subjects then watched +0.36 point [+0.05, +0.70], nothing lost.
RELATED_BONUS = 0.1
TITLES = {"other": {"en": "Close to what you watch", "fr": "Proche de ce que vous regardez"}}
WHY = {
    "subject": {
        "en": "Videos about a subject of {videos} videos you watched ({share} of your subjects), "
        "closest first",
        "fr": "Vidéos sur un sujet de {videos} vidéos regardées ({share} de vos sujets), les "
        "plus proches d'abord",
    },
    "other": {
        "en": "Videos close to one of your subjects, not enough of them for a row of their own",
        "fr": "Vidéos proches d'un de vos sujets, pas assez nombreuses pour une ligne à elles",
    },
}


class VideosRanking:
    """One page of Signals by videos being ranked, step by step."""

    def __init__(self, algorithm: "SignalsVideos", run: Run):
        self.algorithm = algorithm
        self.run = run

    def rows(self) -> list[Row]:
        if (ended := self._read()) is not None:
            return ended
        self._compare()
        self._score()
        self._place()
        return self._build()

    def _read(self) -> list[Row] | None:
        """The candidates worth ranking and the text each is read by."""
        self.found = self.algorithm.subjects(self.run.snapshot)
        self.pool = Pool.gather(self.run.groups)
        drop_unwanted(self.run, self.pool, seen=True, shorts=False)
        self.urls = [
            url
            for url in self.pool.urls()
            if self.pool.videos[url].title
            and (self.pool.videos[url].view_count or MIN_VIEWS) >= MIN_VIEWS
        ]
        if not self.found or not self.urls:
            return []
        self.by_id = {subject.id: index for index, subject in enumerate(self.found)}
        facts = self.algorithm.catalogue.get([vid for url in self.urls if (vid := video_id(url))])
        self.texts = [self.algorithm.text_of(self.pool.videos[url], facts) for url in self.urls]

        # Embedded already, plus EMBED_BUDGET new ones; the rest wait for the background.
        self.read = [True] * len(self.urls)
        known_texts = getattr(self.algorithm.embedder, "known", None)
        if known_texts:
            budget = EMBED_BUDGET
            self.read = []
            for ready in known_texts(self.texts):
                self.read.append(ready or budget > 0)
                budget -= 0 if ready else 1
            later = [t for t, ok in zip(self.texts, self.read, strict=True) if not ok]
            if later and hasattr(self.algorithm.embedder, "prefetch"):
                self.algorithm.embedder.prefetch(later)
        return None

    def _compare(self) -> None:
        """How close each candidate is to each subject, and which ones a hidden subject refuses."""
        centroids = np.array([subject.centroid for subject in self.found])
        self.closeness = np.zeros((len(self.urls), len(self.found)))
        rows_read = [i for i, ok in enumerate(self.read) if ok]
        hidden = [s.centroid for s in self.algorithm.all_subjects(self.run.snapshot) if s.muted]
        refused: set[int] = set()
        if rows_read:
            vectors = self.algorithm.embedder.embed([self.texts[i] for i in rows_read])
            self.closeness[rows_read] = vectors @ centroids.T
            if hidden:
                # Close to a subject the person hid: left out, whatever else it is close to.
                far = (vectors @ np.array(hidden).T).max(axis=1)
                refused = {i for i, value in zip(rows_read, far, strict=True) if value >= MUTED}
        if hidden:
            # With a hidden subject nothing passes unread: it could be about that subject.
            refused |= {i for i, ok in enumerate(self.read) if not ok}
            for i in sorted(refused):
                self.run.drop(self.urls[i], "muted")
            self.closeness[sorted(refused)] = 0.0
            self.read = [ok and i not in refused for i, ok in enumerate(self.read)]
        for i, ok in enumerate(self.read):
            if i in refused:
                continue
            if not ok:  # not read yet: its source's subject
                asked = {
                    self.by_id[s.ref]
                    for s in self.pool.sources[self.urls[i]]
                    if s.ref in self.by_id
                }
                for index in asked:
                    self.closeness[i, index] = UNREAD

    def _score(self) -> None:
        """The score of each candidate and the subject it suits best."""
        rng = self.run.rng(self.run.values.flag("random_enabled"))
        varies = self.run.values.flag("random_enabled")
        self.position = {url: i for i, url in enumerate(self.urls)}
        echo = Counter(self.pool.videos[url].uploader_url for url in self.urls)
        most = max((video.view_count or 0 for video in self.pool.videos.values()), default=0)
        self.totals: dict[str, float] = {}
        self.best: dict[str, int] = {}
        for i, url in enumerate(self.urls):
            index = int(self.closeness[i].argmax())
            views = self.pool.videos[url].view_count or 0
            popularity = math.log1p(views) / math.log1p(most) if most else 0.0
            self.best[url] = index
            self.totals[url] = score(
                self.run,
                url,
                {
                    "subject": float(self.closeness[i, index])
                    * (0.7 + 0.3 * self.found[index].weight / self.found[0].weight),
                    "popularity": POPULARITY_BONUS * popularity,
                    "echo": ECHO_BONUS * min(echo[self.pool.videos[url].uploader_url] - 1, 3) / 3,
                    "related": RELATED_BONUS
                    * any(source.kind == "related" for source in self.pool.sources[url]),
                    **({"draw": DRAW * rng.random()} if varies else {}),
                },
            )

    def _place(self) -> None:
        """Shares the page between the subjects, best scores first, under the cap per channel."""
        size = self.run.values.whole("page_size")
        cap = self.run.values.whole("max_per_channel")
        room = [
            min(SUBJECT_ROW_MAX, max(SUBJECT_ROW_MIN, round(size * subject.weight)))
            for subject in self.found
        ]
        self.placed: dict[int, list[str]] = {index: [] for index in range(len(self.found))}
        self.other: list[str] = []
        per_channel: dict[str, int] = {}
        count = 0
        for url in sorted(self.urls, key=lambda u: -self.totals[u]):
            if count >= size:
                break
            channel = self.pool.videos[url].uploader_url
            if per_channel.get(channel, 0) >= cap:
                self.run.drop(url, "channel_cap")
                continue
            index = self.best[url]
            close = float(self.closeness[self.position[url], index])
            if close >= ROW_FLOOR and len(self.placed[index]) < room[index]:
                self.placed[index].append(url)
            elif close >= UNREAD:
                self.other.append(url)
            else:
                continue  # about none of the person's subjects
            per_channel[channel] = per_channel.get(channel, 0) + 1
            count += 1

    def _build(self) -> list[Row]:
        """One row per subject that got videos, then the rest."""
        rows: list[Row] = []
        for index, subject in enumerate(self.found):
            chosen = self.placed[index]
            if len(chosen) < SUBJECT_ROW_MIN:
                self.other += chosen
                continue
            reason = Reason(kind="topic", ref=subject.id)
            rows.append(
                make_row(
                    subject.id,
                    subject.label,
                    reason,
                    [
                        (
                            url,
                            [
                                Because(
                                    kind="topic",
                                    ref=subject.id,
                                    weight=round(
                                        float(self.closeness[self.position[url], index]), 3
                                    ),
                                )
                            ],
                        )
                        for url in chosen
                    ],
                )
            )
            self.run.explain(
                subject.id, WHY["subject"], videos=subject.videos, share=f"{subject.weight:.0%}"
            )
        if self.other:
            self.other.sort(key=lambda u: -self.totals[u])
            rows.append(
                make_row(
                    "videos-other",
                    self.run.text(TITLES["other"]),
                    Reason(kind="recommended"),
                    [
                        (
                            url,
                            [
                                Because(
                                    kind="topic",
                                    ref=self.found[self.best[url]].id,
                                    weight=round(
                                        float(self.closeness[self.position[url], self.best[url]]), 3
                                    ),
                                )
                            ],
                        )
                        for url in self.other
                    ],
                )
            )
            self.run.explain("videos-other", WHY["other"])
        return rows
