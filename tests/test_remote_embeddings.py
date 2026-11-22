import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from recommender import embeddings
from recommender.remote_embeddings import EmbeddingError, RemoteEmbedder


def _embedder(post, batch=64):
    return RemoteEmbedder("http://models.local/v1/", "tiny", batch=batch, post=post)


def _answer(vectors):
    # Out of order on purpose: the API gives an index for each vector.
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    return {"data": list(reversed(data))}


def test_vectors_come_back_unit_length_in_title_order():
    sent = []

    def post(body):
        sent.append(body)
        return _answer([[float(len(t)), 1.0, 0.0] for t in body["input"]])

    vectors = _embedder(post).embed(["Camping", "Street food tour"])
    assert sent == [{"model": "tiny", "input": ["camping", "street food tour"]}]
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors[0][0] / vectors[0][1] == pytest.approx(7.0)
    assert vectors[1][0] / vectors[1][1] == pytest.approx(16.0)


def test_titles_are_sent_in_batches_and_cached():
    sizes = []

    def post(body):
        sizes.append(len(body["input"]))
        return _answer([[1.0, float(i)] for i, _ in enumerate(body["input"])])

    embedder = _embedder(post, batch=2)
    embedder.embed([f"title {i}" for i in range(5)])
    embedder.embed(["title 1", "title 4"])
    assert sizes == [2, 2, 1]


def test_the_vector_length_follows_the_model():
    embedder = _embedder(lambda body: _answer([[0.5] * 384 for _ in body["input"]]))
    assert embedder.embed(["a"]).shape == (1, 384)
    assert embedder.embed([]).shape == (0, 384)


def test_a_model_that_changes_length_is_refused():
    lengths = iter([3, 4])

    def post(body):
        size = next(lengths)
        return _answer([[1.0] * size for _ in body["input"]])

    embedder = _embedder(post)
    embedder.embed(["first"])
    with pytest.raises(EmbeddingError, match="changed vector length from 3 to 4"):
        embedder.embed(["second"])


def test_a_short_answer_is_an_error():
    with pytest.raises(EmbeddingError, match="returned 1 vectors for 2 titles"):
        _embedder(lambda body: _answer([[1.0, 0.0]])).embed(["a", "b"])


def test_a_failing_background_batch_does_not_stay_queued(monkeypatch):
    def post(_body):
        raise EmbeddingError("down")

    embedder = _embedder(post)
    embedder.prefetch(["a", "b"])
    embedder._worker.join(timeout=5)
    assert not embedder.busy()


class _Handler(BaseHTTPRequestHandler):
    calls: list = []
    status = 200

    def do_POST(self):  # noqa: N802 - http.server's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls.append((self.headers.get("Authorization"), body))
        if type(self).status != 200:
            self.send_response(type(self).status)
            self.end_headers()
            self.wfile.write(b"model not found")
            return
        payload = json.dumps(_answer([[1.0, 2.0] for _ in body["input"]])).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


@pytest.fixture
def endpoint():
    _Handler.calls, _Handler.status = [], 200
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_http_requests_carry_the_key_and_errors_name_the_status(endpoint):
    embedder = RemoteEmbedder(endpoint, "tiny", api_key="secret-key")
    assert embedder.embed(["hello"]).shape == (1, 2)
    assert _Handler.calls[0] == ("Bearer secret-key", {"model": "tiny", "input": ["hello"]})
    _Handler.status = 404
    with pytest.raises(EmbeddingError, match="answered 404 for tiny: model not found") as error:
        RemoteEmbedder(endpoint, "tiny", api_key="secret-key").embed(["other"])
    assert "secret-key" not in str(error.value)


def test_an_unreachable_endpoint_is_an_error():
    embedder = RemoteEmbedder("http://127.0.0.1:9/v1", "tiny", timeout_s=1)
    with pytest.raises(EmbeddingError, match="did not answer"):
        embedder.embed(["hello"])


def test_the_provider_is_chosen_from_the_environment(monkeypatch):
    embeddings.default_embedder.cache_clear()
    monkeypatch.setenv("RECOMMENDER_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("RECOMMENDER_EMBEDDING_BASE_URL", "http://models.local/v1")
    monkeypatch.setenv("RECOMMENDER_EMBEDDING_MODEL", "tiny")
    monkeypatch.delenv("RECOMMENDER_EMBEDDER", raising=False)
    try:
        assert isinstance(embeddings.default_embedder(), RemoteEmbedder)
        embeddings.default_embedder.cache_clear()
        monkeypatch.delenv("RECOMMENDER_EMBEDDING_MODEL")
        with pytest.raises(ValueError, match="needs RECOMMENDER_EMBEDDING_BASE_URL"):
            embeddings.default_embedder()
        embeddings.default_embedder.cache_clear()
        monkeypatch.setenv("RECOMMENDER_EMBEDDING_PROVIDER", "cloud")
        with pytest.raises(ValueError, match="local or openai"):
            embeddings.default_embedder()
    finally:
        embeddings.default_embedder.cache_clear()
