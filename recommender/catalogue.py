"""The catalogue: what videos say about themselves, kept as TypeType comes across them.

YouTube gives a video's tags, category and description only when the video is opened, not in
a list of results. Every time TypeType-Server opens one (someone plays it), it can send those
facts here (`POST /v1/catalogue`); the algorithms that work by subject read them back for the
videos a person watched and for the candidates of a page. Nothing is ever fetched from here:
the catalogue only keeps what it is given, and no video length is stored.

In the recommender's PostgreSQL database when it has one (RECOMMENDER_DATABASE_URL), read
through a bounded memory cache; in memory only otherwise. A database that fails must never
take a page down: every error falls back to memory and to "not known", which the callers
already handle (a video without facts is read by its title).
"""

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import cache

from .database import Database, default_database

# A description is mostly links and credits after its first lines; tags beyond the first few
# repeat each other. Enough to say what a video is about, small enough to embed fast.
DESCRIPTION_CHARS = 300
MAX_TAGS = 12
MEMORY_ENTRIES = 200_000
VIDEO_ID = re.compile(r"(?:v=|shorts/|youtu\.be/|live/)([\w-]{11})")


@dataclass(frozen=True)
class Facts:
    video_id: str
    title: str = ""
    channel_url: str = ""
    category: str = ""
    tags: tuple[str, ...] = ()
    description: str = ""

    def text(self) -> str:
        """What the video is about, in one line, for the embedding model."""
        parts = [self.title, ", ".join(self.tags[:MAX_TAGS]), self.category]
        return ". ".join(p for p in parts if p)


def video_id(url: str) -> str | None:
    match = VIDEO_ID.search(url)
    return match.group(1) if match else None


class Catalogue:
    def __init__(self, database: Database | None = None):
        self._lock = threading.Lock()
        self._memory: OrderedDict[str, Facts | None] = OrderedDict()
        self._database = database

    def _remember(self, key: str, facts: Facts | None) -> None:
        self._memory[key] = facts
        self._memory.move_to_end(key)
        while len(self._memory) > MEMORY_ENTRIES:
            self._memory.popitem(last=False)

    def put(self, entries: list[Facts]) -> int:
        """Keeps these videos' facts (the newest given replaces the older); how many were kept."""
        # Cut to what is stored, so a video reads the same before and after a restart.
        entries = [
            replace(e, tags=e.tags[:MAX_TAGS], description=e.description[:DESCRIPTION_CHARS])
            for e in entries
            if e.video_id and (e.tags or e.category)
        ]
        with self._lock:
            for entry in entries:
                self._remember(entry.video_id, entry)
        if self._database is not None and entries:
            try:
                with self._database.connection() as connection, connection.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO catalogue_videos "
                        "(video_id, title, channel_url, category, tags, description) "
                        "VALUES (%s, %s, %s, %s, %s::jsonb, %s) "
                        "ON CONFLICT (video_id) DO UPDATE SET title = EXCLUDED.title, "
                        "channel_url = EXCLUDED.channel_url, category = EXCLUDED.category, "
                        "tags = EXCLUDED.tags, description = EXCLUDED.description, "
                        "updated_at = now()",
                        [
                            (
                                e.video_id,
                                e.title,
                                e.channel_url,
                                e.category,
                                json.dumps(list(e.tags), ensure_ascii=False),
                                e.description,
                            )
                            for e in entries
                        ],
                    )
            except Exception as error:  # noqa: BLE001 - kept in memory all the same
                self._database.failed(error)
        return len(entries)

    def get(self, ids: list[str]) -> dict[str, Facts]:
        """The facts known for these video ids; an unknown video is simply absent."""
        found: dict[str, Facts] = {}
        with self._lock:
            missing = []
            for key in dict.fromkeys(ids):
                if key in self._memory:
                    if self._memory[key] is not None:
                        found[key] = self._memory[key]
                else:
                    missing.append(key)
        stored = self._stored(missing) if self._database is not None and missing else {}
        with self._lock:
            for key in missing:
                # Asked once: not asked from the database again.
                self._remember(key, stored.get(key))
        return found | stored

    def _stored(self, ids: list[str]) -> dict[str, Facts]:
        try:
            with self._database.connection() as connection:
                rows = connection.execute(
                    "SELECT video_id, title, channel_url, category, tags, description "
                    "FROM catalogue_videos WHERE video_id = ANY(%s)",
                    [ids],
                ).fetchall()
            return {
                key: Facts(key, title, channel, category, tuple(tags or ()), description)
                for key, title, channel, category, tags, description in rows
            }
        except Exception as error:  # noqa: BLE001 - a database that fails reads as "not known"
            self._database.failed(error)
            return {}

    def size(self) -> int:
        if self._database is not None:
            try:
                with self._database.connection() as connection:
                    return connection.execute("SELECT count(*) FROM catalogue_videos").fetchone()[0]
            except Exception as error:  # noqa: BLE001
                self._database.failed(error)
        with self._lock:
            return sum(1 for facts in self._memory.values() if facts is not None)


@cache
def default_catalogue() -> Catalogue:
    return Catalogue(default_database())
