"""The contract with TypeType-Server: the candidates Server brings and the page answered."""

from typing import Literal

from pydantic import Field

from .contract import ChannelUrl, Model, Snapshot, Source

# --- Rank ----------------------------------------------------------------------------


class Candidate(Model):
    url: str
    title: str = ""
    uploader_url: ChannelUrl = ""
    uploader_name: str = ""
    duration_s: int | None = None
    view_count: int | None = None
    uploaded_at: int | None = None
    is_short: bool = False


class CatalogueVideo(Model):
    """What a video says about itself, as Server reads it when it opens the video."""

    url: str
    title: str = ""
    uploader_url: ChannelUrl = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    description: str = ""


class CatalogueUpload(Model):
    videos: list[CatalogueVideo]


class CatalogueStatus(Model):
    kept: int
    size: int


class CandidateGroup(Model):
    source: Source
    videos: list[Candidate]


class RankRequest(Model):
    # The snapshot, or the `plan_id` /v1/plan returned for it (a large history then travels
    # and is read once per page). With both, the snapshot is used.
    snapshot: Snapshot | None = None
    plan_id: str | None = None
    candidates: list[CandidateGroup]
    trace: bool = False


ReasonKind = Literal[
    "watched",
    "subscriptions",
    "new_from_subscriptions",
    "continue",
    "watch_again",
    "topic",
    "discovery",
    "trending",
    "same_channel",
    "recommended",
]

DropReason = Literal[
    "seen", "muted", "channel_cap", "excluded", "not_interested", "short", "obscure"
]


class Because(Model):
    kind: ReasonKind
    ref: str | None = None
    weight: float


class RowVideo(Model):
    url: str
    because: list[Because]


class Reason(Model):
    kind: ReasonKind
    ref: str | None = None


class Row(Model):
    id: str
    title: str
    reason: Reason
    videos: list[RowVideo]


class Dropped(Model):
    url: str
    why: DropReason


class Score(Model):
    url: str
    total: float
    factors: dict[str, float]


class TraceRow(Model):
    id: str
    why: str


class Trace(Model):
    sources: list[Source]
    candidates_received: int
    dropped: list[Dropped]
    scores: list[Score]
    rows: list[TraceRow]


class RankResponse(Model):
    rows: list[Row]
    trace: Trace | None = None
