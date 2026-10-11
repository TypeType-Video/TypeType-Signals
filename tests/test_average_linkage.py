"""The O(n²) average linkage gives the groups of the original O(n³) one, and stays fast."""

import time

import numpy as np
import pytest

from recommender.signals_clusters import average_linkage


def reference(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """The implementation Signals used up to v7, kept to check the new one against."""
    count = len(vectors)
    clusters: list[list[int]] = [[i] for i in range(count)]
    sums = vectors @ vectors.T
    np.fill_diagonal(sums, -np.inf)
    sizes = np.ones(count)
    alive = np.ones(count, dtype=bool)
    while alive.sum() > 1:
        average = sums / np.outer(sizes, sizes)
        average[~alive, :] = -np.inf
        average[:, ~alive] = -np.inf
        a, b = np.unravel_index(int(np.argmax(average)), average.shape)
        if average[a, b] < threshold:
            break
        a, b = min(a, b), max(a, b)
        clusters[a] += clusters[b]
        clusters[b] = []
        sums[a, :] += sums[b, :]
        sums[:, a] += sums[:, b]
        sums[a, a] = -np.inf
        sizes[a] += sizes[b]
        alive[b] = False
    return [sorted(c) for c in clusters if c]


def topics(seed: int, count: int, spread: float) -> np.ndarray:
    """Unit vectors around a few random topic centres, like titles of a few interests."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(max(2, count // 15), 48))
    vectors = centres[rng.integers(len(centres), size=count)] + spread * rng.normal(
        size=(count, 48)
    )
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("threshold", [0.3, 0.55, 0.7])
def test_same_groups_as_the_original(seed, threshold):
    vectors = topics(seed, 160, spread=0.35 + 0.1 * seed)
    assert average_linkage(vectors, threshold) == reference(vectors, threshold)


def test_edge_cases():
    assert average_linkage(np.zeros((0, 4), dtype=np.float32), 0.55) == []
    one = np.ones((1, 4), dtype=np.float32) / 2
    assert average_linkage(one, 0.55) == [[0]]


def test_thousands_of_titles_stay_fast():
    vectors = topics(7, 3000, spread=0.5)
    started = time.perf_counter()
    average_linkage(vectors, 0.55)
    assert time.perf_counter() - started < 5.0
