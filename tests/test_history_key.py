"""A history is sent once (/v1/history); pages name its key instead of carrying it."""

from fastapi.testclient import TestClient

from recommender import history_cache
from recommender.algorithms.signals import Signals
from recommender.contract import HistoryUpload
from recommender.main import app

from .test_embed_budget import ColdEmbedder, likes
from .test_plan_id import snapshot_json
from .test_signals import rank_candidates

client = TestClient(app)


def test_a_page_by_key_gets_the_page_the_whole_snapshot_gets():
    snap = snapshot_json()
    assert (
        client.post(
            "/v1/history", json={"key": "k1", "history": snap["history"], "favorites": []}
        ).status_code
        == 204
    )
    light = {k: v for k, v in snap.items() if k != "history"} | {"history_key": "k1"}
    candidates = [g.model_dump() for g in rank_candidates()]
    by_key = client.post("/v1/rank", json={"snapshot": light, "candidates": candidates}).json()
    whole = client.post("/v1/rank", json={"snapshot": snap, "candidates": candidates}).json()
    assert by_key["rows"] == whole["rows"]
    assert (
        client.post("/v1/plan", json=light).json()["sources"]
        == client.post("/v1/plan", json=snap).json()["sources"]
    )


def test_an_unknown_key_asks_for_the_history():
    light = {k: v for k, v in snapshot_json().items() if k != "history"}
    response = client.post("/v1/plan", json=light | {"history_key": "never sent"})
    assert response.status_code == 409
    assert "send the history" in response.json()["detail"]


def test_a_snapshot_bringing_its_history_under_a_key_leaves_it_for_the_next_pages():
    snap = snapshot_json() | {"history_key": "k2"}
    assert client.post("/v1/plan", json=snap).status_code == 200
    light = {k: v for k, v in snap.items() if k != "history"}
    assert client.post("/v1/plan", json=light).status_code == 200


def test_histories_are_bounded_by_size_and_the_newest_is_kept(monkeypatch):
    monkeypatch.setattr(history_cache, "HISTORY_ITEMS_KEPT", 15)
    monkeypatch.setattr(history_cache, "_histories", type(history_cache._histories)())
    for key in ("a", "b", "c"):
        history_cache.remember_history(HistoryUpload(key=key, favorites=likes(10)))
    assert list(history_cache._histories) == ["c"]


def test_a_new_history_is_embedded_ahead_in_the_background():
    embedder = ColdEmbedder(warm=set())
    Signals(embedder).warm([], likes(5))
    assert len(embedder.prefetched) == 5
