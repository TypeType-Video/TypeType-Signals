from datetime import datetime
from zoneinfo import ZoneInfo

from recommender.signals_history import hour_factor

from .factories import HOUR, NOW, snapshot, watched

PARIS = ZoneInfo("Europe/Paris")
# NOW is 12:00 Paris.


def at_hour(hour: int, days_ago: int = 1) -> int:
    base = datetime.fromtimestamp(NOW / 1000, PARIS).replace(hour=hour, minute=0)
    return int(base.timestamp() * 1000) - days_ago * 24 * HOUR


def test_an_interest_watched_at_this_hour_counts_more_and_at_other_hours_less():
    noon = [at_hour(12, d) for d in range(1, 7)]
    night = [at_hour(2, d) for d in range(1, 7)]
    assert hour_factor(noon, NOW, PARIS) > 2
    assert hour_factor(night, NOW, PARIS) < 0.6


def test_too_few_timed_watches_change_nothing():
    assert hour_factor([at_hour(12)], NOW, PARIS) == 1.0


def test_watches_with_an_estimated_time_are_not_used():
    from recommender import engine

    camping = [
        ("c1", "Camping alone in the snowy forest"),
        ("c2", "Winter camping shelter in the forest"),
        ("c3", "Solo camping forest cabin build"),
        ("c4", "Camping in the forest at night"),
    ]
    exact = [watched(i, "wild", title=t, ago_h=24 * (n + 1)) for n, (i, t) in enumerate(camping)]
    estimated = [item.model_copy(update={"watched_at_precision_s": 10**6}) for item in exact]
    factor = lambda items: engine.profile(snapshot("signals", history=items)).groups[0].hour_factor  # noqa: E731
    assert factor(exact) > 1
    assert factor(estimated) == 1.0
    off = engine.profile(snapshot("signals", {"time_of_day": False}, history=exact)).groups[0]
    assert off.hour_factor == 1.0
