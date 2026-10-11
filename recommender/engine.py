"""Runs an algorithm on a request and guarantees what Server relies on.

Whatever an algorithm returns, the response only contains videos Server
sent as candidates, each video at most once on the page, and no empty row.
"""

from .base import Run
from .between_requests import UnknownPlan, _keep, _kept, with_history
from .catalogue import Facts, default_catalogue, video_id
from .contract import Plan, Snapshot
from .contract_algorithms import AlgorithmList
from .contract_profile import Profile, TopicRequest, TopicResponse
from .contract_rank import CatalogueStatus, CatalogueUpload, RankRequest, RankResponse, Row, Trace
from .embeddings import foreground
from .registry import ALGORITHMS, algorithm_of, default_algorithm, restricted


def remember_videos(upload: CatalogueUpload) -> CatalogueStatus:
    """Keeps what these videos say about themselves (recommender/catalogue.py)."""
    catalogue = default_catalogue()
    kept = catalogue.put(
        [
            Facts(
                video_id=video_id(v.url) or "",
                title=v.title,
                channel_url=v.uploader_url,
                category=v.category,
                tags=tuple(v.tags),
                description=v.description,
            )
            for v in upload.videos
        ]
    )
    return CatalogueStatus(kept=kept, size=catalogue.size())


def list_algorithms(locale: str) -> AlgorithmList:
    return AlgorithmList(
        default_algorithm=default_algorithm(),
        algorithms=[a.info(locale) for a in ALGORITHMS.values()],
    )


def plan(snapshot: Snapshot) -> Plan:
    with foreground():
        return _plan(snapshot)


def _plan(snapshot: Snapshot) -> Plan:
    snapshot = with_history(snapshot)
    algorithm = algorithm_of(snapshot)
    snapshot = restricted(snapshot, algorithm)
    sources = algorithm.plan(snapshot, algorithm.values(snapshot))
    return Plan(sources=sources, plan_id=_keep(snapshot))


def _safe_rows(rows: list[Row], known: set[str]) -> list[Row]:
    placed: set[str] = set()
    safe: list[Row] = []
    for row in rows:
        videos = [v for v in row.videos if v.url in known and v.url not in placed]
        placed.update(v.url for v in videos)
        if videos:
            safe.append(row.model_copy(update={"videos": videos}))
    return safe


def rank(request: RankRequest) -> RankResponse:
    with foreground():
        return _rank(request)


def _rank(request: RankRequest) -> RankResponse:
    if request.snapshot is not None:
        algorithm = algorithm_of(request.snapshot)
        snapshot = restricted(with_history(request.snapshot), algorithm)
    elif request.plan_id:
        snapshot = _kept(request.plan_id)  # already restricted when it was planned
        algorithm = algorithm_of(snapshot)
    else:
        raise UnknownPlan("a rank needs a snapshot or a plan_id")
    run = Run(snapshot, algorithm.values(snapshot), request.candidates)
    rows = _safe_rows(algorithm.rank(run), {v.url for g in request.candidates for v in g.videos})
    if not request.trace:
        return RankResponse(rows=rows)
    kept = {row.id for row in rows}
    return RankResponse(
        rows=rows,
        trace=Trace(
            sources=[group.source for group in request.candidates],
            candidates_received=len({v.url for g in request.candidates for v in g.videos}),
            dropped=run.dropped,
            scores=run.scores,
            rows=[reason for reason in run.row_reasons if reason.id in kept],
        ),
    )


def topics(request: TopicRequest) -> TopicResponse:
    with foreground():
        return _topics(request)


def _topics(request: TopicRequest) -> TopicResponse:
    algorithm = algorithm_of(request.snapshot)
    snapshot = restricted(with_history(request.snapshot), algorithm)
    matches = algorithm.topics(snapshot, algorithm.values(snapshot), request.titles)
    return TopicResponse(matches=matches)


def profile(snapshot: Snapshot) -> Profile:
    with foreground():
        return _profile(snapshot)


def _profile(snapshot: Snapshot) -> Profile:
    snapshot = with_history(snapshot)
    algorithm = algorithm_of(snapshot)
    snapshot = restricted(snapshot, algorithm)
    return algorithm.profile(snapshot, algorithm.values(snapshot))
