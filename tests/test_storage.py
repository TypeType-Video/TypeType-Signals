"""The stores that outlive the process. The PostgreSQL and Redis tests run when
RECOMMENDER_TEST_DATABASE_URL and RECOMMENDER_TEST_CACHE_URL name throwaway services."""

import os
import uuid

import numpy as np
import pytest

from recommender import between_requests, engine, state
from recommender.catalogue import Catalogue, Facts
from recommender.contract import HistoryUpload, Snapshot

from .factories import snapshot, watched

DATABASE_URL = os.environ.get("RECOMMENDER_TEST_DATABASE_URL", "")
CACHE_URL = os.environ.get("RECOMMENDER_TEST_CACHE_URL", "")
needs_database = pytest.mark.skipif(not DATABASE_URL, reason="no test database")
needs_cache = pytest.mark.skipif(not CACHE_URL, reason="no test cache")


@pytest.fixture
def database():
    from recommender.database import Database

    opened = Database(DATABASE_URL)
    yield opened
    opened.close()


@needs_database
def test_migrations_run_once_and_create_the_tables(database):
    from recommender.database import Database

    Database(DATABASE_URL).close()  # a second start applies nothing again
    with database.connection() as connection:
        names = connection.execute("SELECT name FROM recommender_schema_migrations").fetchall()
        tables = connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name IN ('title_vectors', 'catalogue_videos')"
        ).fetchone()
    assert [name for (name,) in names] == ["0001_initial.sql"]
    assert tables[0] == 2


@needs_database
def test_vectors_read_back_from_postgres_are_the_ones_stored(database):
    from recommender.embeddings import PostgresVectorStore

    model = f"test-{uuid.uuid4().hex}"
    vectors = {
        "camping in the snow": np.random.default_rng(1).random(768).astype(np.float32),
        "مرحبا بالعالم": np.random.default_rng(2).random(768).astype(np.float32),
    }
    PostgresVectorStore(database, model).put(vectors)
    PostgresVectorStore(database, model).put(vectors)  # stored once
    found = PostgresVectorStore(database, model).get([*vectors, "never embedded"])
    assert set(found) == set(vectors)
    assert all(
        found[t].dtype == np.float32 and np.array_equal(found[t], vectors[t]) for t in vectors
    )
    assert PostgresVectorStore(database, model + "-other").get(list(vectors)) == {}


@needs_database
def test_the_catalogue_is_kept_in_postgres(database):
    key = uuid.uuid4().hex[:11]
    Catalogue(database).put([Facts(key, "A title", "", "Music", ("jazz", "live"))])
    again = Catalogue(database)
    assert again.get([key, "unknown0000"])[key].tags == ("jazz", "live")
    assert again.size() >= 1


def test_a_database_that_fails_never_takes_a_page_down():
    class Broken:
        def connection(self):
            raise OSError("down")

        def failed(self, error):
            self.error = error

    from recommender.embeddings import PostgresVectorStore

    broken = Broken()
    store = PostgresVectorStore(broken, "model")
    assert store.get(["a title"]) == {}
    store.put({"a title": np.zeros(4, dtype=np.float32)})
    catalogue = Catalogue(broken)
    assert catalogue.put([Facts("abcdefghijk", "A title", "", "Music", ("jazz",))]) == 1
    assert catalogue.get(["abcdefghijk"])["abcdefghijk"].category == "Music"
    assert isinstance(broken.error, OSError)


def _forget_locally():
    between_requests._histories.clear()
    between_requests._plans.clear()


def _shared(monkeypatch, store):
    monkeypatch.setattr(between_requests, "default_state", lambda: store)
    _forget_locally()


def _page_survives_a_restart(monkeypatch, store):
    _shared(monkeypatch, store)
    key = uuid.uuid4().hex
    history = [watched(f"v{i}", f"c{i % 3}") for i in range(12)]
    between_requests._share_history(HistoryUpload(key=key, history=history))
    between_requests._hold_history(HistoryUpload(key=key, history=history))
    named = Snapshot.model_validate({**snapshot("signals").model_dump(), "history_key": key})
    planned = engine.plan(named)
    _forget_locally()  # the process restarts, or another instance takes the rank
    kept = between_requests._kept(planned.plan_id)
    assert [item.video_url for item in kept.history] == [item.video_url for item in history]
    assert engine.plan(named).sources == planned.sources


def test_a_history_and_a_plan_survive_a_restart(monkeypatch):
    _page_survives_a_restart(monkeypatch, state.MemoryState())


@needs_cache
def test_a_history_and_a_plan_survive_a_restart_in_the_shared_cache(monkeypatch):
    _page_survives_a_restart(monkeypatch, state.RedisState(CACHE_URL))


def test_without_a_shared_store_an_unknown_key_is_still_refused(monkeypatch):
    _shared(monkeypatch, None)
    named = Snapshot.model_validate({**snapshot("signals").model_dump(), "history_key": "gone"})
    with pytest.raises(between_requests.UnknownHistory):
        engine.plan(named)
    with pytest.raises(between_requests.UnknownPlan):
        between_requests._kept("gone")


def test_a_cache_that_fails_reads_as_not_kept():
    broken = state.RedisState("redis://127.0.0.1:1")
    broken.set("key", b"value", 10)
    assert broken.get("key") is None


def _in_parts(monkeypatch, store):
    _shared(monkeypatch, store)
    key = uuid.uuid4().hex
    history = [watched(f"v{i}", f"c{i % 3}") for i in range(9)]
    for part in range(3):
        between_requests.receive_history(
            HistoryUpload(
                key=key, history=history[part * 3 : part * 3 + 3], part=part, last=part == 2
            )
        )
    named = Snapshot.model_validate({**snapshot("signals").model_dump(), "history_key": key})
    kept = between_requests.with_history(named)
    assert [item.video_url for item in kept.history] == [item.video_url for item in history]


def test_a_history_sent_in_parts_is_put_back_together(monkeypatch):
    _in_parts(monkeypatch, None)
    _in_parts(monkeypatch, state.MemoryState())


@needs_cache
def test_a_history_sent_in_parts_through_the_shared_cache(monkeypatch):
    _in_parts(monkeypatch, state.RedisState(CACHE_URL))


def test_a_last_part_whose_earlier_parts_are_gone_is_refused(monkeypatch):
    _shared(monkeypatch, None)
    with pytest.raises(between_requests.UnknownHistory):
        between_requests.receive_history(HistoryUpload(key="lost", history=[], part=2, last=True))
