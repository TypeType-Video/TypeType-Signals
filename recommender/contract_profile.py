"""The contract with TypeType-Server: what an algorithm concluded about a person."""

from pydantic import ConfigDict, Field

from .contract import Model, Snapshot

# --- Profile -------------------------------------------------------------------------


class ProfileGroup(Model):
    id: str
    label: str
    weight: float
    # How much more (above 1) or less (below 1) this interest counts at this hour, from the
    # hours its videos were watched; 1 when those hours are not known well enough.
    hour_factor: float = 1.0
    from_: list[str] = Field(alias="from", serialization_alias="from")
    pinned: bool = False
    muted: bool = False

    model_config = ConfigDict(populate_by_name=True)


class ProfileChannel(Model):
    id: str
    name: str
    weight: float
    from_: list[str] = Field(alias="from", serialization_alias="from")

    model_config = ConfigDict(populate_by_name=True)


class SessionBoost(Model):
    ref: str
    label: str
    strength: float
    expires_at: int


class TopicRequest(Model):
    snapshot: Snapshot
    titles: list[str]


class TopicMatch(Model):
    group: str | None
    label: str | None
    similarity: float


class TopicResponse(Model):
    matches: list[TopicMatch]


class Profile(Model):
    groups: list[ProfileGroup] = []
    channels: list[ProfileChannel] = []
    session_boosts: list[SessionBoost] = []
