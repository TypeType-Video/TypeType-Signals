"""Vectors from an OpenAI-compatible `/v1/embeddings` endpoint, for a model served elsewhere.

RECOMMENDER_EMBEDDING_PROVIDER=openai turns it on. RECOMMENDER_EMBEDDING_BASE_URL is the API root
(`http://host:8080/v1`), RECOMMENDER_EMBEDDING_MODEL the model name it serves, and
RECOMMENDER_EMBEDDING_API_KEY (or _API_KEY_FILE) the key, if it wants one. Titles go to that
endpoint, so it should run on the operator's own machines. The cache, the store and the background
queue are the local model's: vectors are keyed by `openai/<model>` in the store, so switching
models never mixes them, and their length is whatever the model gives.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit

import numpy as np

from . import settings
from .embeddings import BATCH, ModelEmbedder, _normalize

RETRIES = 1
RETRY_PAUSE_S = 1.0


class EmbeddingError(RuntimeError):
    """The endpoint failed or answered something that is not one vector per title."""


Post = Callable[[dict], dict]


class RemoteEmbedder(ModelEmbedder):
    _dimension = 0

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 30.0,
        batch: int = BATCH,
        post: Post | None = None,
    ):
        self._url = base_url.rstrip("/") + "/embeddings"
        self._host = urlsplit(self._url).netloc or self._url
        self._model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._batch = max(1, batch)
        self._post = post or self._http_post
        self._start(f"openai/{model}")

    @classmethod
    def from_settings(cls) -> "RemoteEmbedder":
        base_url = settings.text("RECOMMENDER_EMBEDDING_BASE_URL")
        model = settings.text("RECOMMENDER_EMBEDDING_MODEL")
        if not base_url or not model:
            raise ValueError(
                "RECOMMENDER_EMBEDDING_PROVIDER=openai needs RECOMMENDER_EMBEDDING_BASE_URL "
                "and RECOMMENDER_EMBEDDING_MODEL"
            )
        return cls(
            base_url,
            model,
            api_key=settings.secret("RECOMMENDER_EMBEDDING_API_KEY"),
            timeout_s=settings.whole("RECOMMENDER_EMBEDDING_TIMEOUT", 30),
            batch=settings.whole("RECOMMENDER_EMBEDDING_BATCH", BATCH),
        )

    def _run(self, texts: list[str]) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), self._batch):
            chunk = texts[start : start + self._batch]
            with self._batches:
                answer = self._post({"model": self._model, "input": chunk})
            rows.append(self._vectors(answer, len(chunk)))
        vectors = np.vstack(rows)
        if self._dimension and vectors.shape[1] != self._dimension:
            raise EmbeddingError(
                f"{self._model} at {self._host} changed vector length from {self._dimension} "
                f"to {vectors.shape[1]}: restart the recommender after changing models"
            )
        self._dimension = vectors.shape[1]
        return _normalize(vectors).astype(np.float32)

    def _vectors(self, answer: dict, count: int) -> np.ndarray:
        data = answer.get("data") if isinstance(answer, dict) else None
        if not isinstance(data, list) or len(data) != count:
            got = len(data) if isinstance(data, list) else "no"
            raise EmbeddingError(f"{self._host} returned {got} vectors for {count} titles")
        ordered = sorted(
            data, key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0
        )
        try:
            vectors = np.array([item["embedding"] for item in ordered], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingError(f"{self._host} returned malformed vectors: {error}") from error
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise EmbeddingError(f"{self._host} returned vectors of different lengths")
        return vectors

    def _http_post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = json.dumps(body).encode()
        for attempt in range(RETRIES + 1):
            request = urllib.request.Request(self._url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                detail = error.read(300).decode(errors="replace").strip()
                failure = EmbeddingError(
                    f"{self._host} answered {error.code} for {self._model}: {detail}"
                )
                if error.code != 429 and error.code < 500:
                    raise failure from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                reason = getattr(error, "reason", error)
                failure = EmbeddingError(
                    f"{self._host} did not answer within {self._timeout_s:g} s: {reason}"
                )
            except json.JSONDecodeError as error:
                raise EmbeddingError(f"{self._host} did not answer with JSON") from error
            if attempt < RETRIES:
                time.sleep(RETRY_PAUSE_S)
        raise failure
