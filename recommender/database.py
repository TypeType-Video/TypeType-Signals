"""The recommender's PostgreSQL database: its own tables, migrated by itself at start.

RECOMMENDER_DATABASE_URL names it (a database of its own, with the pgvector extension).
Left empty, nothing is kept on disk: vectors live in memory only, which is what the tests
and a quick local run want. A database that fails must never take a page down: callers
treat every error as "not stored".
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from importlib import resources

from . import settings

log = logging.getLogger("recommender.database")

# One lock for every instance that starts at the same time: migrations run once.
MIGRATION_LOCK = 7_316_944_120


class Database:
    def __init__(self, url: str):
        from pgvector.psycopg import register_vector
        from psycopg_pool import ConnectionPool

        self._migrate(url)
        self._pool = ConnectionPool(
            url,
            min_size=1,
            max_size=settings.whole("RECOMMENDER_DATABASE_POOL", 4),
            configure=register_vector,
            open=True,
        )
        self._warned = threading.Event()

    def _migrate(self, url: str) -> None:
        import psycopg

        with psycopg.connect(url, autocommit=True) as connection:
            connection.execute("SELECT pg_advisory_lock(%s)", [MIGRATION_LOCK])
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS recommender_schema_migrations "
                    "(name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
                )
                done = {
                    row[0]
                    for row in connection.execute("SELECT name FROM recommender_schema_migrations")
                }
                folder = resources.files("recommender").joinpath("migrations")
                for script in sorted(folder.iterdir(), key=lambda entry: entry.name):
                    if not script.name.endswith(".sql") or script.name in done:
                        continue
                    with connection.transaction():
                        connection.execute(script.read_text())
                        connection.execute(
                            "INSERT INTO recommender_schema_migrations (name) VALUES (%s)",
                            [script.name],
                        )
                    log.info("applied migration %s", script.name)
            finally:
                connection.execute("SELECT pg_advisory_unlock(%s)", [MIGRATION_LOCK])

    @contextmanager
    def connection(self) -> Iterator:
        with self._pool.connection() as connection:
            yield connection

    def failed(self, error: Exception) -> None:
        """Said once in the log, then quietly: the service goes on without the database."""
        if not self._warned.is_set():
            self._warned.set()
            log.warning("database unavailable, going on without it: %s", error)

    def close(self) -> None:
        self._pool.close()


@cache
def default_database() -> Database | None:
    url = settings.secret("RECOMMENDER_DATABASE_URL")
    return Database(url) if url else None
