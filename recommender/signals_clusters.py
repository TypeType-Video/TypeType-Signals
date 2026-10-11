"""Grouping titles that mean the same thing, and the subjects a person adds themselves."""

import hashlib

import numpy as np

from .embeddings import Embedder
from .labels import topic_text
from .signals_profile import Group


def average_linkage(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Merges the two closest groups while their members are, on average, this similar.

    Comparing every member with every member (not a moving centre) keeps a
    group from drifting one step at a time into an unrelated topic.

    Built with the nearest-neighbour chain, in O(n²) time instead of O(n³) (a
    Takeout brings tens of thousands of titles). Average linkage is reducible:
    merging never brings a group closer to a third one than its closest part
    was. So two groups that are each other's closest are merged in the same way
    whatever the order, and when they are closest but under `threshold`, neither
    can ever reach it again: both are set aside. The groups are the same as
    merging the closest pair while it is at least `threshold` similar.
    """
    count = len(vectors)
    sums = (vectors @ vectors.T).astype(np.float64)
    np.fill_diagonal(sums, -np.inf)
    sizes = np.ones(count)
    open_ = np.ones(count, dtype=bool)  # can still merge
    closed = np.zeros(count, dtype=bool)  # its opposite, kept beside it: read at every step
    remaining = count
    owner = np.arange(count)  # the group each vector ended in (its lowest index)
    chain: list[int] = []
    while remaining > 1:
        if not chain:
            chain.append(int(np.argmax(open_)))  # the first one still open
        a = chain[-1]
        row = sums[a] / (sizes[a] * sizes)
        row[closed] = -np.inf
        row[a] = -np.inf
        b = int(np.argmax(row))
        if len(chain) > 1 and row[chain[-2]] >= row[b]:
            b = chain[-2]
        if len(chain) == 1 or b != chain[-2]:
            chain.append(b)
            continue
        chain.pop()
        chain.pop()
        if row[b] < threshold:
            open_[a] = open_[b] = False
            closed[a] = closed[b] = True
            remaining -= 2
            chain.clear()
            continue
        keep, drop = min(a, b), max(a, b)
        sums[keep, :] += sums[drop, :]
        sums[:, keep] += sums[:, drop]
        sums[keep, keep] = -np.inf
        sizes[keep] += sizes[drop]
        open_[drop] = False
        closed[drop] = True
        remaining -= 1
        owner[owner == drop] = keep
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(int(owner[index]), []).append(index)
    return [groups[key] for key in sorted(groups)]


# An interest the person wrote themselves: at most this many, each of at most this many
# characters and words (the words are what its search asks for).
CUSTOM_INTERESTS = 10
CUSTOM_CHARS = 60
CUSTOM_WORDS = 6
CUSTOM_PREFIX = "custom:"


def custom_groups(texts: list[str], deduced: list[Group], embedder: Embedder) -> list[Group]:
    """The subjects the person added, as pinned interests. Nothing was watched about them, so
    each one counts as much as the strongest interest deduced from the history: the person
    asked for it. Its id comes from its text, so adding the same subject twice is one group."""
    wanted: dict[str, str] = {}
    for text in texts:
        cleaned = " ".join(text.split())[:CUSTOM_CHARS].strip()
        if cleaned:
            wanted.setdefault(cleaned.casefold(), cleaned)
    labels = list(wanted.values())[:CUSTOM_INTERESTS]
    if not labels:
        return []
    vectors = embedder.embed([topic_text(label) for label in labels])
    strongest = max((g.score for g in deduced if not g.muted and not g.ignored), default=1.0)
    return [
        Group(
            id=CUSTOM_PREFIX + hashlib.sha1(label.casefold().encode()).hexdigest()[:12],
            label=label,
            members=[],
            centroid=vector / (np.linalg.norm(vector) or 1),
            score=strongest,
            keywords=label.split()[:CUSTOM_WORDS],
            pinned=True,
        )
        for label, vector in zip(labels, vectors, strict=True)
    ]
