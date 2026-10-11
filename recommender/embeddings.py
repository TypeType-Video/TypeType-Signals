"""Turns video titles into vectors, so titles about the same thing end up close.

Titles on a real account mix many languages, so a multilingual model is
used: paraphrase-multilingual-mpnet-base-v2, in its fp16 ONNX build, run
locally with onnxruntime. Nothing leaves the machine. Titles are
cut at 32 tokens, which none of them needs more than.

The int8 build was used until 2026-09-30. Measured then on 379 real titles: its dynamic
quantization scales activations over the whole batch, so a title's vector
depended on the titles embedded with it (similarities moved by up to 0.137,
310 title pairs crossed the 0.55 group threshold between batched and one by
one), and even alone it disagreed with the full model on 290 of the 1,200
pairs above that threshold, the threshold having been chosen with the full
model. The full model is batch-independent (0 pairs) at 1.7x the time of the
int8 one (3.2 s against 1.9 s for 379 titles on 4 CPUs) and 1.1 GB of
memory against 0.28 GB. fp16 is batch-independent too, agrees with the full
model on all but 1 of those 1,200 pairs, loads in 1.0 GB instead of 1.9 GB,
and is slower on CPU (5.4 s for the 379 titles): with EMBED_BUDGET below, a
first page after an import took 1.23 s on 8 CPUs (whole account, 87,064
watches and 34,577 likes). It is the default; RECOMMENDER_EMBEDDING_FILE
picks another build (onnx/model.onnx for the full one).

Vectors are cached by text: the cache holds public video titles only,
never who watched them, and can be turned off with
RECOMMENDER_EMBEDDING_CACHE=0. With RECOMMENDER_DATABASE_URL they are also
kept in PostgreSQL (pgvector), shared by every instance, so a restart does
not send every user back to embedding thousands of titles (about 8 s for
2,000 on 4 CPUs). Vectors are float32 whichever way they come, so a result does
not depend on whether a title was just embedded or read back.
"""

import hashlib
import os
import queue
import re
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Protocol

import numpy as np

from . import settings
from .database import Database, default_database

REPOSITORY = settings.text(
    "RECOMMENDER_EMBEDDING_MODEL", "xenova/paraphrase-multilingual-mpnet-base-v2"
)
MODEL_FILE = settings.text("RECOMMENDER_EMBEDDING_FILE", "onnx/model_fp16.onnx")
MODEL_CACHE = settings.text("RECOMMENDER_MODEL_CACHE") or None
THREADS = settings.whole("RECOMMENDER_THREADS", min(8, os.cpu_count() or 1))
MAX_TOKENS = 32
BATCH = 64
# Batches going through the model at once. Pages computed together each take the model's
# working memory: with 1,000 different people asking for a page (2026-10-10, arena off) the
# process peaked at 1,492 MB with one batch at a time (134 s in all, the slowest ranks at
# 3.6 s), 1,796 MB with two (87 s, 2.3 s) and 2,310 MB with three (87 s, 1.9 s). Smaller
# batches did not lower the peak. One by default, for the 2 GB of an ordinary instance.
PARALLEL_BATCHES = settings.whole("RECOMMENDER_PARALLEL_BATCHES", 1)
# New titles embedded at most per request, the rest in the background: 100 keeps the first page
# after an import under 1.5 s at three times the measured account (150: up to 1.72 s).
EMBED_BUDGET = settings.whole("RECOMMENDER_EMBED_BUDGET", 100)
CACHE_SIZE = 50_000 if settings.text("RECOMMENDER_EMBEDDING_CACHE", "1") != "0" else 0


# Background embedding gives way to pages: it waits while a request is being answered, and works in
# small batches (~0.2 s each on the reference machine) so a page never waits long behind one.
# Sharing the CPU with it took the first page after a history arrived from 0.94 to 1.88 s at three
# times the measured account.
BACKGROUND_BATCH = 16
_requests = 0
_requests_lock = threading.Lock()
_no_request = threading.Event()
_no_request.set()


@contextmanager
def foreground() -> Iterator[None]:
    """A request being answered: background embedding waits for it."""
    global _requests
    with _requests_lock:
        _requests += 1
        _no_request.clear()
    try:
        yield
    finally:
        with _requests_lock:
            _requests -= 1
            if _requests == 0:
                _no_request.set()


class VectorStore(Protocol):
    """Where title vectors are kept beyond the process's own memory."""

    def get(self, texts: list[str]) -> dict[str, np.ndarray]: ...

    def put(self, vectors: dict[str, np.ndarray]) -> None: ...


def _hash(text: str) -> bytes:
    return hashlib.md5(text.encode()).digest()


def _array(vector: object) -> np.ndarray:
    # The driver hands a vector back as its own type, or as an array, depending on its version.
    values = vector.to_numpy() if hasattr(vector, "to_numpy") else vector
    return np.asarray(values, dtype=np.float32)


class PostgresVectorStore:
    """Title vectors in PostgreSQL (pgvector), keyed by model build and cleaned title.

    Two model builds give different vectors: the model is part of the key, so switching
    builds never mixes them. Vectors are float32 there as in memory: a title reads back
    exactly as it was embedded. Shared by every instance, kept across restarts."""

    def __init__(self, database: Database, model: str):
        self._database = database
        self._model = model

    def get(self, texts: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        try:
            with self._database.connection() as connection:
                for start in range(0, len(texts), 1000):
                    rows = connection.execute(
                        "SELECT text, vector FROM title_vectors "
                        "WHERE model = %s AND text_hash = ANY(%s)",
                        [self._model, [_hash(t) for t in texts[start : start + 1000]]],
                    )
                    found.update((t, _array(v)) for t, v in rows)
        except Exception as error:  # noqa: BLE001 - a database that fails reads as "not stored"
            self._database.failed(error)
        return found

    def put(self, vectors: dict[str, np.ndarray]) -> None:
        try:
            with self._database.connection() as connection, connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO title_vectors (model, text_hash, text, vector) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (model, text_hash) DO NOTHING",
                    [(self._model, _hash(t), t, v.astype(np.float32)) for t, v in vectors.items()],
                )
        except Exception as error:  # noqa: BLE001
            self._database.failed(error)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """One unit-length row per text."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def _clean(text: str) -> str:
    # Shouting in capitals and hashtags change what the model sees, not what the video is about.
    return re.sub(r"#\w+", " ", text).lower().strip()


class ModelEmbedder:
    def __init__(self):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(
            hf_hub_download(REPOSITORY, "tokenizer.json", cache_dir=MODEL_CACHE)
        )
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        # onnxruntime's arena keeps the memory of its largest batches for good. Measured on
        # 1,000 different people asking for a page (2026-10-10): 2,734 MB at the end with it,
        # 1,459 MB without (1,725 MB at the highest), for 10 % more time in all and the slowest
        # ranks at 2.2 s instead of 1.7 s. Off unless RECOMMENDER_ONNX_ARENA=1; an arena that
        # grows only by what is asked ("kSameAsRequested") still ended at 2,549 MB.
        options.enable_cpu_mem_arena = settings.text("RECOMMENDER_ONNX_ARENA", "0") == "1"
        self._session = onnxruntime.InferenceSession(
            hf_hub_download(REPOSITORY, MODEL_FILE, cache_dir=MODEL_CACHE),
            options,
            providers=["CPUExecutionProvider"],
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        database = default_database() if CACHE_SIZE else None
        self._store: VectorStore | None = (
            PostgresVectorStore(database, f"{REPOSITORY}/{MODEL_FILE}") if database else None
        )
        self._lock = threading.Lock()
        self._batches = threading.Semaphore(PARALLEL_BATCHES)
        self._pending: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._queued: set[str] = set()

    def _run(self, texts: list[str]) -> np.ndarray:
        # Batches of titles of similar length pad less, so the model does less work.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        rows = []
        for start in range(0, len(order), BATCH):
            encoded = self._tokenizer.encode_batch([texts[i] for i in order[start : start + BATCH]])
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            with self._batches:
                hidden = self._session.run(None, feeds)[0]
            # Mean of the token vectors, padding left out: how the model was trained.
            rows.append((hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True))
        vectors = np.empty((len(texts), rows[0].shape[1]))
        vectors[order] = np.vstack(rows)
        return _normalize(vectors).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        cleaned = [_clean(text) for text in texts]
        if not cleaned:
            return np.zeros((0, 768), dtype=np.float32)
        with self._lock:
            ready = {t: self._cache[t] for t in dict.fromkeys(cleaned) if t in self._cache}
        missing = [t for t in dict.fromkeys(cleaned) if t not in ready]
        stored = self._store.get(missing) if self._store and missing else {}
        missing = [t for t in missing if t not in stored]
        vectors = self._run(missing) if missing else []
        fresh = dict(zip(missing, vectors, strict=True))
        if self._store and fresh:
            self._store.put(fresh)
        fresh |= stored
        if CACHE_SIZE:
            with self._lock:
                for text, vector in fresh.items():
                    self._cache[text] = vector
                while len(self._cache) > CACHE_SIZE:
                    self._cache.popitem(last=False)
        ready |= fresh
        return np.array([ready[t] for t in cleaned])

    def known(self, texts: list[str]) -> list[bool]:
        """Which texts have a vector already (in memory or on disk): no model run needed."""
        cleaned = [_clean(text) for text in texts]
        with self._lock:
            in_memory = {t for t in cleaned if t in self._cache}
        stored = (
            self._store.get([t for t in dict.fromkeys(cleaned) if t not in in_memory])
            if self._store
            else {}
        )
        if stored and CACHE_SIZE:
            with self._lock:
                self._cache.update(stored)
        return [t in in_memory or t in stored for t in cleaned]

    def queued(self, texts: list[str]) -> list[bool]:
        """Which texts wait for the background: a request must not embed them a second time."""
        cleaned = [_clean(text) for text in texts]
        with self._lock:
            return [t in self._queued for t in cleaned]

    def busy(self) -> bool:
        """True while titles wait for the background: what is read now is not yet everything."""
        with self._lock:
            return bool(self._queued)

    def prefetch(self, texts: list[str]) -> None:
        """Embed these texts in the background, so a later request finds them ready."""
        with self._lock:
            fresh = [
                t for t in dict.fromkeys(_clean(text) for text in texts) if t not in self._queued
            ]
            self._queued.update(fresh)
        for text in fresh:
            self._pending.put(text)
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drain, daemon=True)
                self._worker.start()

    def _drain(self) -> None:
        while True:
            _no_request.wait()
            try:
                batch = [self._pending.get(timeout=1)]
            except queue.Empty:
                return
            while len(batch) < BACKGROUND_BATCH and not self._pending.empty():
                batch.append(self._pending.get_nowait())
            self.embed(batch)
            with self._lock:
                self._queued.difference_update(batch)


class HashingEmbedder:
    """Words hashed into buckets: no model, fully predictable. Used by the tests."""

    def __init__(self, size: int = 256):
        self._size = size

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._size))
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", _clean(text)):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._size
                vectors[row, bucket] += 1
        return _normalize(vectors)


@cache
def default_embedder() -> Embedder:
    if settings.text("RECOMMENDER_EMBEDDER") == "hashing":
        return HashingEmbedder()
    return ModelEmbedder()
