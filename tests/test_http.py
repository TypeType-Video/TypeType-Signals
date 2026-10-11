from fastapi.testclient import TestClient

from recommender.main import app

client = TestClient(app)


def test_algorithms_are_listed_in_the_requested_language():
    body = client.get("/v1/algorithms", params={"locale": "fr"}).json()
    names = {a["id"]: a["name"] for a in body["algorithms"]}
    assert names["subscriptions"] == "Abonnements"
    parameters = body["algorithms"][0]["parameters"]
    assert all(p["help"] and p["label"] for p in parameters)


def test_the_list_names_the_preselected_algorithm(monkeypatch):
    assert client.get("/v1/algorithms").json()["default_algorithm"] == "signals"
    monkeypatch.setenv("RECOMMENDER_DEFAULT_ALGORITHM", "discovery")
    assert client.get("/v1/algorithms").json()["default_algorithm"] == "discovery"
    monkeypatch.setenv("RECOMMENDER_DEFAULT_ALGORITHM", "no-such")
    assert client.get("/v1/algorithms").json()["default_algorithm"] == "subscriptions"


def test_a_secret_is_read_from_its_file(tmp_path, monkeypatch):
    from recommender.settings import secret

    path = tmp_path / "token"
    path.write_text("abc\n")
    monkeypatch.delenv("RECOMMENDER_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv("RECOMMENDER_INTERNAL_TOKEN_FILE", str(path))
    assert secret("RECOMMENDER_INTERNAL_TOKEN") == "abc"
    monkeypatch.setenv("RECOMMENDER_INTERNAL_TOKEN", "direct")
    assert secret("RECOMMENDER_INTERNAL_TOKEN") == "direct"


def test_unknown_algorithm_is_a_404():
    response = client.post("/v1/plan", json={"algorithm": {"id": "nope"}, "context": {"now": 0}})
    assert response.status_code == 404


def test_unreadable_snapshot_is_a_client_error():
    assert client.post("/v1/plan", json={"context": {}}).status_code == 422


def test_topics_places_titles_in_the_interest_groups():
    from .test_signals import history

    snapshot = {
        "algorithm": {"id": "signals"},
        "context": {"now": 1_790_676_000_000},
        "history": [item.model_dump() for item in history()],
    }
    body = client.post(
        "/v1/topics",
        json={"snapshot": snapshot, "titles": ["Camping in the snowy forest", "zzz qqq"]},
    ).json()
    first, second = body["matches"]
    assert first["group"] and "camping" in first["label"].lower()
    assert second["group"] is None


def test_the_memory_report_is_off_unless_asked_for_and_counts_what_is_kept(monkeypatch):
    from fastapi.testclient import TestClient

    from recommender.kept import Kept
    from recommender.main import app

    client = TestClient(app)
    monkeypatch.delenv("RECOMMENDER_MEMORY_REPORT", raising=False)
    assert client.get("/v1/memory").status_code == 404
    monkeypatch.setenv("RECOMMENDER_MEMORY_REPORT", "1")
    store = Kept()
    store._results[("a person",)] = ["a profile"] * 1000
    found = client.get("/v1/memory").json()
    assert {"results": 1, "megabytes": [0.0]} in found["kept"]
    assert found["title_vectors"]["entries"] >= 0
    assert set(found["after_trim"]) <= {"resident", "own", "files"}
