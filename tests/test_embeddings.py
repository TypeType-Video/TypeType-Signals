import threading
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np

from recommender.embeddings import ModelEmbedder


class FakeTokenizer:
    def encode_batch(self, texts):
        longest = max(len(t) for t in texts)
        return [
            SimpleNamespace(
                ids=[len(t)] * longest, attention_mask=[1] * len(t) + [0] * (longest - len(t))
            )
            for t in texts
        ]


class FakeSession:
    """Every token's vector points along the axis of its text's length."""

    def run(self, _outputs, feeds):
        ids = feeds["input_ids"]
        hidden = np.zeros((*ids.shape, 64))
        for row, length in enumerate(ids[:, 0]):
            hidden[row, :, length] = 1.0
        return [hidden]


def test_titles_batched_by_length_keep_their_own_vectors():
    embedder = ModelEmbedder.__new__(ModelEmbedder)
    embedder._tokenizer = FakeTokenizer()
    embedder._session = FakeSession()
    embedder._inputs = {"input_ids", "attention_mask"}
    embedder._batches = threading.Semaphore(1)
    embedder._cache = {}
    texts = ["a" * n for n in (9, 2, 30, 5, 17, 1)]
    vectors = embedder._run(texts)
    assert [int(np.argmax(v)) for v in vectors] == [len(t) for t in texts]


class RefusingSession:
    def run(self, _outputs, _feeds):
        raise AssertionError("the model must not run: the vectors are on disk")


def _embedder(session, store):
    embedder = ModelEmbedder.__new__(ModelEmbedder)
    embedder._tokenizer = FakeTokenizer()
    embedder._session = session
    embedder._inputs = {"input_ids", "attention_mask"}
    embedder._cache = OrderedDict()
    embedder._store = store
    embedder._lock = threading.Lock()
    embedder._batches = threading.Semaphore(1)
    return embedder


class MemoryStore:
    """A store shared by several embedders, as the database is by several instances."""

    def __init__(self, kept: dict, model: str):
        self._kept, self._model = kept, model

    def get(self, texts):
        return {t: self._kept[self._model, t] for t in texts if (self._model, t) in self._kept}

    def put(self, vectors):
        self._kept.update({(self._model, t): v for t, v in vectors.items()})


def test_vectors_kept_in_the_store_come_back_identical_after_a_restart():
    kept: dict = {}
    texts = ["Camping in the snow", "Street food tour", "Camping in the snow"]
    first = _embedder(FakeSession(), MemoryStore(kept, "model-a")).embed(texts)
    again = _embedder(RefusingSession(), MemoryStore(kept, "model-a")).embed(texts)
    assert first.dtype == again.dtype == np.float32
    assert np.array_equal(first, again)
