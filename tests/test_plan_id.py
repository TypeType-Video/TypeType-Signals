"""A page reads its snapshot once: /v1/rank can name the plan instead of sending it again."""

from fastapi.testclient import TestClient

from recommender import signals_conclude
from recommender.main import app

from .test_signals import history, rank_candidates

client = TestClient(app)


def snapshot_json() -> dict:
    return {
        "algorithm": {"id": "signals", "parameters": {"random_enabled": False}},
        "context": {
            "now": 1_790_676_000_000,
            "timezone": "Europe/Paris",
            "locale": "en",
            "seed": 1,
        },
        "history": [h.model_dump() for h in history()],
    }


def test_rank_by_plan_id_gives_the_page_rank_by_snapshot_gives(monkeypatch):
    calls = []
    real = signals_conclude.conclude
    monkeypatch.setattr(
        "recommender.algorithms.signals.conclude", lambda *a: calls.append(1) or real(*a)
    )
    snap = snapshot_json()
    plan = client.post("/v1/plan", json=snap).json()
    assert plan["plan_id"]
    candidates = [g.model_dump() for g in rank_candidates()]
    by_id = client.post(
        "/v1/rank", json={"plan_id": plan["plan_id"], "candidates": candidates}
    ).json()
    assert len(calls) == 1  # the profile was built once, for the plan
    by_snapshot = client.post("/v1/rank", json={"snapshot": snap, "candidates": candidates}).json()
    assert by_id["rows"] == by_snapshot["rows"]


def test_an_unknown_plan_asks_for_the_snapshot():
    response = client.post("/v1/rank", json={"plan_id": "nope", "candidates": []})
    assert response.status_code == 409
    assert "send the snapshot" in response.json()["detail"]
