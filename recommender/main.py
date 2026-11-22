"""HTTP entry point. Only TypeType-Server calls it, with the shared internal token."""

import hmac
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import msgspec
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from . import __version__, between_requests, engine, history_cache, registry
from .contract import Plan, Snapshot, read_history_upload
from .contract_algorithms import AlgorithmList
from .contract_profile import Profile, TopicRequest, TopicResponse
from .contract_rank import CatalogueStatus, CatalogueUpload, RankRequest, RankResponse
from .embeddings import default_embedder
from .remote_embeddings import EmbeddingError
from .settings import secret, text

TOKEN = secret("RECOMMENDER_INTERNAL_TOKEN")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Load the model before the first request, so no user waits for it. A remote endpoint that is
    # down at start is not fatal: pages answer 503 until it is back.
    try:
        default_embedder().embed(["warm up"])
    except EmbeddingError as error:
        logging.getLogger(__name__).warning("embedding endpoint not ready at start: %s", error)
    yield


app = FastAPI(title="TypeType Recommender", lifespan=lifespan)


def internal(x_internal_token: str = Header(default="")) -> None:
    # An empty token disables the check, for local development only.
    if TOKEN and not hmac.compare_digest(x_internal_token, TOKEN):
        raise HTTPException(status_code=401, detail="invalid internal token")


def known[T](call: Callable[[], T]) -> T:
    try:
        return call()
    except registry.UnknownAlgorithm as error:
        raise HTTPException(status_code=404, detail=f"unknown algorithm {error}") from error
    except history_cache.UnknownHistory as error:
        raise HTTPException(
            status_code=409, detail=f"unknown or expired history {error}: send the history"
        ) from error
    except between_requests.UnknownPlan as error:
        raise HTTPException(
            status_code=409, detail=f"unknown or expired plan {error}: send the snapshot"
        ) from error
    except EmbeddingError as error:
        raise HTTPException(
            status_code=503, detail=f"embedding endpoint failed: {error}"
        ) from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/version", dependencies=[Depends(internal)])
def version() -> dict[str, str]:
    """What runs here, in the shape the other TypeType services give it."""
    revision = text("RECOMMENDER_REVISION", "unknown")
    return {
        "service": "recommender",
        "version": __version__,
        "revision": revision,
        "shortRevision": revision[:7],
        "buildTime": text("RECOMMENDER_BUILD_TIME", "unknown"),
    }


@app.get("/v1/memory", dependencies=[Depends(internal)])
def memory() -> dict:
    """Where the memory is, for measuring under load (off unless asked for)."""
    if text("RECOMMENDER_MEMORY_REPORT") != "1":
        raise HTTPException(status_code=404, detail="memory report is off")
    from .memory_report import report

    return report()


@app.get("/v1/algorithms", dependencies=[Depends(internal)])
def algorithms(locale: str = "en") -> AlgorithmList:
    return engine.list_algorithms(locale)


@app.post("/v1/history", dependencies=[Depends(internal)], status_code=204)
async def history(request: Request) -> None:
    """A history sent once when it changes, whole or in parts; pages then name its key."""
    await run_in_threadpool(_receive_history, await request.body())


def _receive_history(raw: bytes) -> None:
    try:
        upload = read_history_upload(raw)
    except msgspec.MsgspecError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    known(lambda: history_cache.receive_history(upload))


@app.post("/v1/catalogue", dependencies=[Depends(internal)])
def catalogue(upload: CatalogueUpload) -> CatalogueStatus:
    return engine.remember_videos(upload)


@app.post("/v1/plan", dependencies=[Depends(internal)])
def plan(snapshot: Snapshot) -> Plan:
    return known(lambda: engine.plan(snapshot))


@app.post("/v1/rank", dependencies=[Depends(internal)], response_model_by_alias=True)
def rank(request: RankRequest) -> RankResponse:
    return known(lambda: engine.rank(request))


@app.post("/v1/topics", dependencies=[Depends(internal)])
def topics(request: TopicRequest) -> TopicResponse:
    """Which interest group each title belongs to, by the code that ranks: for the bench,
    and for "why this video?"."""
    return known(lambda: engine.topics(request))


@app.post("/v1/profile", dependencies=[Depends(internal)], response_model_by_alias=True)
def profile(snapshot: Snapshot) -> Profile:
    return known(lambda: engine.profile(snapshot))
