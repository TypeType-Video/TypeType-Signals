"""The interest groups Signals concludes: videos watched through, clustered by their titles."""

from typing import TYPE_CHECKING

import numpy as np

from .embeddings import EMBED_BUDGET
from .labels import generic_words, group_words, informative, title_counts, topic_text
from .signals_clusters import average_linkage, custom_groups
from .signals_history import (
    _recency,
    _zone,
    hour_factor,
)
from .signals_profile import (
    CHANNEL_HALF_LIFE_DAYS,
    DAY,
    GROUP_SIMILARITY,
    IMPORTED_GROUP_WEIGHT,
    MAX_GROUP_REFS,
    MAX_LASTING,
    MAX_TITLES,
    TIME_PRECISION_S,
    Group,
    ref_tail,
)

if TYPE_CHECKING:
    from .signals_conclude import Concluding


# What can make a group: (reference, title, channel name, weight). A plain tuple, read by
# position: there is one per watch and per like of the whole history, and naming the fields
# (a NamedTuple) was measured 4 to 5 % slower on 261,000 watches and 104,000 likes.
Entry = tuple[str, str, str, float]


def _decided(group: Group, decisions: list[str] | set[str]) -> bool:
    """A decision about a group follows it even when its id moves: a group is muted, pinned or
    ignored when the decision names one of its videos."""
    tails = {ref_tail(member) for member in group.members} | {ref_tail(group.id)}
    return any(ref_tail(d) in tails for d in decisions if d.startswith("group:"))


class InterestGroups:
    """The interest groups of the person a profile is being concluded for."""

    def __init__(self, concluding: "Concluding"):
        self.of = concluding

    def groups(self) -> list[Group]:
        """Interest groups: videos watched through, clustered by what their titles mean."""
        entries, background = self._read_now(self._informative(self._entries()))
        self.found: list[Group] = self._clustered(entries)
        for group in self.found:
            group.muted = _decided(group, self.of.overrides.muted_groups)
            group.pinned = _decided(group, self.of.overrides.pinned_groups)
            group.ignored = _decided(group, self.of.ignored)
        self.found += custom_groups(
            self.of.overrides.custom_interests, self.found, self.of.embedder
        )
        counted_groups = [g for g in self.found if not g.muted and not g.ignored]
        total = sum(g.score for g in counted_groups) or 1.0
        for group in counted_groups:
            group.weight = group.score / total
        if self.of.values.flag("time_of_day"):
            self._hour_factors()
        self.found.sort(key=lambda g: (-g.weight, g.id))
        if background:
            self.of.embedder.prefetch(background)
        return self.found

    def _entries(self) -> list[Entry]:
        """What can make a group, as (reference, title, channel name, weight): videos watched
        through, imported watches and likes."""
        wanted = self.of.lasting_share
        lasting = [
            item
            for item in self.of.history
            if (0.5 if item.watched_share is None else item.watched_share) >= wanted
            and item.title
            and not item.is_short
        ]
        lasting = list({item.video_url: item for item in reversed(lasting)}.values())[::-1]
        # A bound for very long histories, so the first page stays fast (measure 6). Measured:
        # cutting at 200 changed the groups themselves and halved the share of new channels,
        # so the bound sits far above an ordinary history (about 350 on the measured account).
        lasting = lasting[:MAX_LASTING]
        self.of.favorites = [f for f in self.of.favorites if f.title]
        entries = [
            (i.id, i.title, i.channel_name, _recency(self.of.now, i.watched_at)) for i in lasting
        ]
        imported_weight = IMPORTED_GROUP_WEIGHT
        if imported_weight > 0:
            # Imported watches (progress unknown, Shorts already left out) are most of an imported
            # history: without them its groups came from likes alone.
            imported = [
                item for item in self.of.history if item.watched_share is None and item.title
            ]
            imported = list({item.video_url: item for item in reversed(imported)}.values())[::-1]
            entries += [
                (
                    i.id,
                    i.title,
                    i.channel_name,
                    imported_weight * _recency(self.of.now, i.watched_at),
                )
                for i in imported[:MAX_LASTING]
            ]
        # A like carries no channel name: the name the channel's watches gave it, if any.
        now, channels = self.of.now, self.of.channels
        for f in self.of.favorites:
            channel = channels.get(f.channel_url)
            at = f.added_at
            liked = 1.0 if at is None else 0.5 ** (max(0, now - at) / DAY / CHANNEL_HALF_LIFE_DAYS)
            entries.append((f.id, f.title, channel.name if channel else "", liked))
        return entries

    @staticmethod
    def _informative(entries: list[Entry]) -> list[Entry]:
        """The entries whose title says something about a topic ("Welcome!", "X is live" cannot
        place a video in a group; it still counts for its channel), MAX_TITLES at most."""
        if len(entries) <= MAX_TITLES:
            return [e for e in entries if informative(e[1])]
        # The most recent informative titles, read newest first only as far as needed.
        picked = []
        for entry in sorted(entries, key=lambda e: -e[3]):
            if informative(entry[1]):
                picked.append(entry)
                if len(picked) == MAX_TITLES:
                    break
        kept = {id(e) for e in picked}
        return picked if len(picked) == MAX_TITLES else [e for e in entries if id(e) in kept]

    def _read_now(self, entries: list[Entry]) -> tuple[list[Entry], list[str]]:
        """The entries whose title can be read for this page, and the texts left for the
        background. A first page after an import would embed thousands of titles never seen:
        the newest EMBED_BUDGET go now, the others in the background (started once this page is
        done), and join the groups when ready. A title already waiting is never embedded twice."""
        known = getattr(self.of.embedder, "known", None)
        if known is None or not entries:
            return entries, []
        texts = [topic_text(e[1]) for e in entries]
        ready, waiting = known(texts), self.of.embedder.queued(texts)
        cold = sorted(
            (i for i in range(len(entries)) if not ready[i] and not waiting[i]),
            key=lambda i: -entries[i][3],
        )
        later = {i for i in range(len(entries)) if waiting[i] and not ready[i]}
        later |= set(cold[EMBED_BUDGET:])
        background = [texts[i] for i in cold[EMBED_BUDGET:]]
        return [e for i, e in enumerate(entries) if i not in later], background

    def _clustered(self, entries: list[Entry]) -> list[Group]:
        """One group per cluster of at least two titles that mean the same thing."""
        if not entries:
            return []
        titles = [e[1] for e in entries]
        vectors = self.of.embedder.embed([topic_text(title) for title in titles])
        counts = title_counts(titles)
        clusters = [m for m in average_linkage(vectors, GROUP_SIMILARITY) if len(m) >= 2]
        generic = generic_words([[titles[i] for i in members] for members in clusters])
        groups = []
        for members in clusters:
            # Ids come from the oldest video, so a group keeps its id as it grows.
            oldest = max(members)
            centroid = vectors[members].mean(axis=0)
            label, named, keywords = group_words(
                [titles[i] for i in members],
                titles,
                [entries[i][2] for i in members],
                counts,
                generic,
            )
            groups.append(
                Group(
                    id="group:" + ref_tail(entries[oldest][0]),
                    label=label,
                    named=named,
                    keywords=keywords,
                    members=[entries[i][0] for i in sorted(members)][:MAX_GROUP_REFS],
                    centroid=centroid / (np.linalg.norm(centroid) or 1),
                    score=sum(entries[i][3] for i in members),
                )
            )
        return groups

    def _hour_factors(self) -> None:
        """How much each group is usually watched at this hour of the day."""
        members = {m for group in self.found for m in group.members}
        timed = {
            item.id: item.watched_at
            for item in self.of.history
            if item.id in members and item.watched_at_precision_s <= TIME_PRECISION_S
        }
        zone = _zone(self.of.snapshot.context.timezone)
        for group in self.found:
            watched_at = [timed[m] for m in group.members if m in timed]
            group.hour_factor = hour_factor(watched_at, self.of.now, zone)
