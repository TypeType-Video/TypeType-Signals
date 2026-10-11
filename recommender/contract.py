"""Shapes exchanged with TypeType-Server (the contract, in the TypeType documentation).

Everything here is plain data. Server sends a snapshot of what the user
allowed it to send, the recommender answers with sources to fetch, then
with rows that only reference videos Server fetched and filtered itself.
"""

import re
from typing import Annotated, Any, Literal, Self

import msgspec
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_core import core_schema

CONTRACT_VERSION = 0

_CHANNEL_PATH = re.compile(
    r"^(?:https?://)?(?:www\.|m\.)?youtube\.com/(channel/UC[\w-]{22}|@[^/?#]+|c/[^/?#]+|user/[^/?#]+)",
    re.IGNORECASE,
)


_CANONICAL = "https://www.youtube.com/channel/UC"
_CANONICAL_LENGTH = len(_CANONICAL) + 22


def canonical_channel_url(url: str) -> str:
    """One spelling per YouTube channel, whatever tab, scheme or trailing slash it came with.

    `http://youtube.com/channel/UC…/videos` and `https://www.youtube.com/channel/UC…` are
    the same channel; imports and clients write both. Other URLs are only trimmed."""
    # Almost every URL already comes in the canonical spelling: no regular expression then
    # (a request carries one per watch and like, 125,000 on a measured account).
    if url.startswith(_CANONICAL) and len(url) == _CANONICAL_LENGTH:
        return url
    match = _CHANNEL_PATH.match(url.strip())
    return f"https://www.youtube.com/{match.group(1)}" if match else url.strip()


ChannelUrl = Annotated[str, AfterValidator(canonical_channel_url)]


class Model(BaseModel):
    # Unknown fields are ignored, so Server can send more than a given
    # algorithm reads without breaking it.
    model_config = ConfigDict(extra="ignore")


# --- Snapshot ------------------------------------------------------------------------


class Context(Model):
    surface: Literal["home"] = "home"
    now: int = Field(description="Milliseconds since the epoch.")
    timezone: str = "UTC"
    locale: str = "en"
    # Fixes the draw of an algorithm that uses randomness. Server sends a new
    # one on every refresh; the same seed always gives the same page.
    seed: int = 0


class Record(msgspec.Struct, kw_only=True):
    """A watch or a like. A history holds one per video, up to hundreds of thousands, and is
    read again each time it changes: these are plain records, read seven times faster than
    the models around them (87,000 watches: 81 ms against 563 ms, measured 2026-10-10).
    They still fit inside those models, which read and write them through the two hooks
    below, and unknown fields are ignored here as there."""

    def model_dump(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)

    def model_copy(self, update: dict[str, Any] | None = None) -> Self:
        return msgspec.structs.replace(self, **(update or {}))

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: object, _handler: object) -> Any:
        def read(value: object) -> Self:
            return value if isinstance(value, cls) else msgspec.convert(value, cls, strict=False)

        return core_schema.no_info_plain_validator_function(
            read,
            serialization=core_schema.plain_serializer_function_ser_schema(msgspec.to_builtins),
        )


class HistoryItem(Record, kw_only=True, gc=False):
    id: str
    video_url: str
    channel_url: str
    channel_name: str = ""
    title: str = ""
    duration_s: int | None = None
    progress_s: int | None = None
    watched_at: int
    # How far `watched_at` may be from the real time, in seconds: 0 when exact (Server).
    # Times estimated from an ordered history without dates are not used for time of day.
    watched_at_precision_s: int = 0
    is_short: bool = False

    def __post_init__(self) -> None:
        self.channel_url = canonical_channel_url(self.channel_url)

    @property
    def watched_share(self) -> float | None:
        if not self.duration_s or self.progress_s is None:
            return None
        return max(0.0, min(1.0, self.progress_s / self.duration_s))


class Subscription(Model):
    id: str
    channel_url: ChannelUrl
    name: str = ""


class VideoRef(Record, kw_only=True, gc=False):
    id: str
    video_url: str
    channel_url: str = ""
    title: str = ""
    # When the user put it in this list (liked it, saved it for later...), epoch ms.
    added_at: int | None = None
    # A Short (Server knows it from the video's length or YouTube's own flag).
    is_short: bool = False

    def __post_init__(self) -> None:
        self.channel_url = canonical_channel_url(self.channel_url)


class Overrides(Model):
    muted_channels: list[ChannelUrl] = []
    muted_groups: list[str] = []
    ignored_signals: list[str] = []
    pinned_groups: list[str] = []
    # Subjects the person added themselves, in their own words, in any language: each one
    # counts as an interest they pinned, even with nothing watched about it yet.
    custom_interests: list[str] = []
    session_reset_at: int | None = None


class AlgorithmChoice(Model):
    id: str
    parameters: dict[str, float | bool] = {}


class Snapshot(Model):
    contract_version: int = CONTRACT_VERSION
    algorithm: AlgorithmChoice
    context: Context
    history: list[HistoryItem] = []
    subscriptions: list[Subscription] = []
    favorites: list[VideoRef] = []
    watch_later: list[VideoRef] = []
    not_interested: list[VideoRef] = []
    recently_shown: list[str] = []
    overrides: Overrides = Overrides()
    # The key of a history sent before to /v1/history. Without `history` and `favorites`,
    # they come from there: a page then carries a few kilobytes instead of the whole history
    # (40 MB and ~0.8 s to read for 125,000 watches and likes). Unknown or expired: 409.
    history_key: str | None = None


class HistoryUpload(msgspec.Struct, kw_only=True):
    """A history sent once, when it changes (an import, new watches), under a key Server
    picks; pages then name the key (Snapshot.history_key)."""

    key: str
    history: list[HistoryItem] = []
    favorites: list[VideoRef] = []
    # A long history may come in several parts, numbered from 0: each brings some of the
    # watches and likes, and the last one says so. One part alone is the whole history.
    part: int = 0
    last: bool = True


_read_upload = msgspec.json.Decoder(HistoryUpload, strict=False)


def read_history_upload(raw: bytes) -> HistoryUpload:
    """The upload a body of JSON holds; raises msgspec.MsgspecError when it does not."""
    return _read_upload.decode(raw)


def write_history_upload(upload: HistoryUpload) -> bytes:
    return msgspec.json.encode(upload)


# --- Plan ----------------------------------------------------------------------------

SourceKind = Literal[
    "subscription_feed", "related", "channel_videos", "trending", "videos", "search"
]


class Source(Model):
    kind: SourceKind
    limit: int
    # related: the video to start from, or `of` to start from videos Server got from
    # another source: `count` of its newest, or `count` drawn with the context seed among
    # its `among` newest. `limit` is then per starting video.
    video_url: str | None = None
    of: Literal["subscription_feed"] | None = None
    count: int | None = None
    among: int | None = None
    # channel_videos
    channel_url: ChannelUrl | None = None
    # videos: these exact videos, which Server already knows (e.g. from the history).
    urls: list[str] | None = None
    # search: what to search for.
    query: str | None = None
    sort: Literal["latest", "popular"] | None = None
    # What in the snapshot made the algorithm ask for this source, for the trace.
    ref: str | None = None


class Plan(Model):
    sources: list[Source]
    # Send it back in /v1/rank instead of the snapshot: the recommender keeps the snapshot it
    # has just read (and what it concluded from it) for a while, so a page reads it once.
    plan_id: str | None = None
