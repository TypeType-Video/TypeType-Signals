"""What every algorithm is made of, and the steps they share.

An algorithm answers three questions from one snapshot: which sources to
fetch (plan), how to arrange what came back (rank), and what it concludes
about the user (profile). The steps below are the ones several algorithms
need: gathering candidates, dropping what the user does not want, and
capping a channel. Every drop is written to the trace with its reason.
"""

import random
from collections import Counter
from dataclasses import dataclass, field

from . import settings
from .contract import Snapshot, Source
from .contract_algorithms import AlgorithmInfo, Guide, Traits
from .contract_profile import Profile, TopicMatch
from .contract_rank import (
    Because,
    Candidate,
    CandidateGroup,
    Dropped,
    DropReason,
    Reason,
    Row,
    RowVideo,
    Score,
    TraceRow,
)
from .parameters import Spec, Values
from .text import Text, pick

SOURCE_BASE = settings.text(
    "RECOMMENDER_SOURCE_BASE",
    "https://github.com/TypeType-Video/TypeType-Signals/blob/main",
)

# A video watched this far is finished; it is not offered again.
FINISHED_SHARE = 0.9
# A watch whose progress is unknown (an imported history) counts as seen: 5-8 % of a page's
# videos were videos the person had already watched, on three real imported profiles.
UNKNOWN_PROGRESS_IS_SEEN = True


@dataclass
class Run:
    """One rank call: what came in, the user's values, and the trace being written."""

    snapshot: Snapshot
    values: Values
    groups: list[CandidateGroup]
    dropped: list[Dropped] = field(default_factory=list)
    scores: list[Score] = field(default_factory=list)
    row_reasons: list[TraceRow] = field(default_factory=list)

    @property
    def locale(self) -> str:
        return self.snapshot.context.locale

    def text(self, text: Text, **values: object) -> str:
        return pick(text, self.locale).format(**values)

    def drop(self, url: str, why: DropReason) -> None:
        self.dropped.append(Dropped(url=url, why=why))

    def explain(self, row_id: str, why: Text | str, **values: object) -> None:
        """Why a row exists; `why` is a text to translate, or one already written."""
        written = why if isinstance(why, str) else self.text(why, **values)
        self.row_reasons.append(TraceRow(id=row_id, why=written))

    def rng(self, enabled: bool) -> random.Random:
        # Without randomness the same data always gives the same page.
        return random.Random(self.snapshot.context.seed if enabled else 0)


@dataclass
class Pool:
    """Every candidate once, in arrival order, with all the sources that brought it."""

    videos: dict[str, Candidate]
    sources: dict[str, list[Source]]

    @classmethod
    def gather(cls, groups: list[CandidateGroup]) -> "Pool":
        videos: dict[str, Candidate] = {}
        sources: dict[str, list[Source]] = {}
        for group in groups:
            for video in group.videos:
                videos.setdefault(video.url, video)
                sources.setdefault(video.url, []).append(group.source)
        return cls(videos, sources)

    def urls(self) -> list[str]:
        return list(self.videos)

    def from_kind(self, kind: str) -> list[str]:
        return [url for url in self.videos if any(s.kind == kind for s in self.sources[url])]

    def remove(self, url: str) -> None:
        self.videos.pop(url, None)
        self.sources.pop(url, None)


# Server does not mark every Short (channel pages and related lists often leave it unset), so length
# and title decide too. Measured on a real account's YouTube home: of 2,765 regular video tiles,
# this drops 18 (0.65 %: memes under a minute), keeps music clips and trailers of 1-3 min, and a
# third of the Shorts shelf's titles carry a hashtag.
SHORT_MAX_S = 60
TAGGED_SHORT_MAX_S = 180


def looks_short(video: Candidate) -> bool:
    if video.is_short:
        return True
    length = video.duration_s or 0
    return 0 < length <= SHORT_MAX_S or (0 < length <= TAGGED_SHORT_MAX_S and "#" in video.title)


def drop_unwanted(run: Run, pool: Pool, *, seen: bool, shorts: bool) -> None:
    """Removes what the user said no to, and optionally what they already saw or Shorts."""
    snapshot = run.snapshot
    muted = set(snapshot.overrides.muted_channels)
    not_interested = {item.video_url for item in snapshot.not_interested}
    already = set(snapshot.recently_shown) | {
        item.video_url
        for item in snapshot.history
        if (item.watched_share or 0) >= FINISHED_SHARE
        or (UNKNOWN_PROGRESS_IS_SEEN and item.watched_share is None)
    }
    for url in pool.urls():
        video = pool.videos[url]
        why: DropReason | None = None
        if video.uploader_url in muted:
            why = "muted"
        elif url in not_interested:
            why = "not_interested"
        elif not shorts and looks_short(video):
            why = "short"
        elif seen and url in already:
            why = "seen"
        if why:
            run.drop(url, why)
            pool.remove(url)


def cap_per_channel(run: Run, urls: list[str], pool: Pool, cap: int) -> list[str]:
    """Keeps at most `cap` videos per channel, in order; the rest are dropped."""
    kept: list[str] = []
    counts: Counter[str] = Counter()
    for url in urls:
        channel = pool.videos[url].uploader_url
        if counts[channel] >= cap:
            run.drop(url, "channel_cap")
            continue
        counts[channel] += 1
        kept.append(url)
    return kept


def make_row(
    row_id: str, title: str, reason: Reason, videos: list[tuple[str, list[Because]]]
) -> Row:
    return Row(
        id=row_id,
        title=title,
        reason=reason,
        videos=[RowVideo(url=url, because=because) for url, because in videos],
    )


def score(run: Run, url: str, factors: dict[str, float]) -> float:
    total = round(sum(factors.values()), 4)
    run.scores.append(
        Score(url=url, total=total, factors={k: round(v, 4) for k, v in factors.items()})
    )
    return total


class Algorithm:
    """Subclasses fill in the fields and the three methods."""

    id: str
    name: Text
    description: Text
    uses: list[str]
    sources: list[str]
    random: bool
    guide: dict[str, Text]
    traits: Traits
    specs: list[Spec]

    def info(self, locale: str) -> AlgorithmInfo:
        return AlgorithmInfo(
            id=self.id,
            name=pick(self.name, locale),
            description=pick(self.description, locale),
            uses=self.uses,
            sources=self.sources,
            random=self.random,
            source_url=f"{SOURCE_BASE}/recommender/algorithms/{self.id}.py",
            guide=Guide(
                for_you_if=pick(self.guide["for_you_if"], locale),
                you_will_see=pick(self.guide["you_will_see"], locale),
                good_to_know=pick(self.guide["good_to_know"], locale),
                traits=self.traits,
            ),
            parameters=[spec.describe(locale) for spec in self.specs],
        )

    def values(self, snapshot: Snapshot) -> Values:
        return Values(self.specs, snapshot.algorithm.parameters)

    def plan(self, snapshot: Snapshot, values: Values) -> list[Source]:
        raise NotImplementedError

    def rank(self, run: Run) -> list[Row]:
        raise NotImplementedError

    def profile(self, snapshot: Snapshot, values: Values) -> Profile:
        return Profile()

    def topics(self, snapshot: Snapshot, values: Values, titles: list[str]) -> list[TopicMatch]:
        """The interest group each title is closest to; algorithms without groups have none."""
        return [TopicMatch(group=None, label=None, similarity=0.0) for _ in titles]
