import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

import recommender
from recommender.main import app


def test_the_version_is_one_semantic_version_said_the_same_everywhere():
    declared = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    assert recommender.__version__ == declared
    major, minor, patch = (int(part) for part in declared.split("."))
    assert (major, minor, patch) >= (0, 0, 1)
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "version": declared}
    assert client.get("/v1/algorithms").json()["version"] == declared


def test_the_service_says_what_runs_in_the_shape_the_other_services_use(monkeypatch):
    monkeypatch.setenv("RECOMMENDER_REVISION", "0123456789abcdef")
    monkeypatch.setenv("RECOMMENDER_BUILD_TIME", "2026-10-10T08:00:00Z")
    assert TestClient(app).get("/version").json() == {
        "service": "recommender",
        "version": recommender.__version__,
        "revision": "0123456789abcdef",
        "shortRevision": "0123456",
        "buildTime": "2026-10-10T08:00:00Z",
    }
