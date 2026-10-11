"""The algorithms the service offers, and what each may read of a snapshot."""

from . import settings
from .algorithms.discovery import Discovery
from .algorithms.signals import Signals
from .algorithms.signals_channels import SignalsChannels
from .algorithms.signals_mix import SignalsMix
from .algorithms.signals_videos import SignalsVideos
from .algorithms.subscriptions import Subscriptions
from .base import Algorithm
from .contract import Snapshot

_signals = Signals()
_channels = SignalsChannels(_signals)  # the same channel weights as Signals, worked out once
_videos = SignalsVideos()
ALGORITHMS: dict[str, Algorithm] = {
    algorithm.id: algorithm
    for algorithm in [
        Subscriptions(),
        Discovery(),
        _signals,
        _channels,
        _videos,
        SignalsMix(_channels, _videos),
    ]
}


class UnknownAlgorithm(LookupError):
    pass


# Snapshot fields by the data category that holds them (the `uses` of an algorithm).
FIELDS = {
    "history": "history",
    "subscriptions": "subscriptions",
    "favorites": "favorites",
    "watch_later": "watch_later",
    "not_interested": "not_interested",
    "shown": "recently_shown",
}


def restricted(snapshot: Snapshot, algorithm: Algorithm) -> Snapshot:
    """The snapshot without the categories the algorithm does not list in `uses`.

    What an algorithm says it reads is then all it can read, whatever Server sends.
    """
    empty = {field: [] for category, field in FIELDS.items() if category not in algorithm.uses}
    return snapshot.model_copy(update=empty)


def algorithm_of(snapshot: Snapshot) -> Algorithm:
    try:
        return ALGORITHMS[snapshot.algorithm.id]
    except KeyError as error:
        raise UnknownAlgorithm(snapshot.algorithm.id) from error


def default_algorithm() -> str:
    """The algorithm preselected for a new user: RECOMMENDER_DEFAULT_ALGORITHM, else Signals."""
    wanted = settings.text("RECOMMENDER_DEFAULT_ALGORITHM", "signals")
    return wanted if wanted in ALGORITHMS else next(iter(ALGORITHMS))
