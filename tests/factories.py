"""Small builders for snapshots and candidates. All data here is made up."""

from recommender.contract import (
    AlgorithmChoice,
    Context,
    HistoryItem,
    Snapshot,
    Source,
    Subscription,
)
from recommender.contract_rank import Candidate, CandidateGroup

HOUR = 3_600_000
DAY = 24 * HOUR
# 2026-09-29 12:00 Europe/Paris
NOW = 1_790_676_000_000


def snapshot(algorithm: str, parameters: dict | None = None, **fields) -> Snapshot:
    return Snapshot(
        algorithm=AlgorithmChoice(id=algorithm, parameters=parameters or {}),
        context=Context(now=NOW, timezone="Europe/Paris", locale=fields.pop("locale", "en")),
        **fields,
    )


def channel(name: str) -> str:
    return f"https://www.youtube.com/channel/{name}"


def video(
    id: str, channel_name: str, *, age_h: float = 1, views: int = 1000, **fields
) -> Candidate:
    return Candidate(
        url=f"https://www.youtube.com/watch?v={id}",
        title=fields.pop("title", f"Video {id}"),
        uploader_url=channel(channel_name),
        uploader_name=channel_name,
        uploaded_at=int(NOW - age_h * HOUR),
        view_count=views,
        duration_s=fields.pop("duration_s", 600),
        **fields,
    )


def group(source: Source, videos: list[Candidate]) -> CandidateGroup:
    return CandidateGroup(source=source, videos=videos)


def subscription(name: str) -> Subscription:
    return Subscription(id=f"subscription:{name}", channel_url=channel(name), name=name)


def watched(
    id: str, channel_name: str, *, share: float = 1.0, ago_h: float = 1, **fields
) -> HistoryItem:
    duration = fields.pop("duration_s", 600)
    return HistoryItem(
        id=f"history:{id}",
        video_url=f"https://www.youtube.com/watch?v={id}",
        channel_url=channel(channel_name),
        channel_name=channel_name,
        title=fields.pop("title", f"Video {id}"),
        duration_s=duration,
        progress_s=int(duration * share),
        watched_at=int(NOW - ago_h * HOUR),
        **fields,
    )
