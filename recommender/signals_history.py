"""What Signals reads straight from a history: how a watch fades, the hour it was watched
at, what is watched again, what is a Short."""

import gc
from collections import Counter
from datetime import datetime
from operator import attrgetter
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .contract import HistoryItem
from .signals_profile import (
    CHANNEL_HALF_LIFE_DAYS,
    DAY,
    HOUR_FACTOR_MAX,
    HOUR_FACTOR_MIN,
    HOUR_WINDOW,
    MIN_TIMED_WATCHES,
    PRIOR_WATCHES,
)


def _recency(now: int, at: int) -> float:
    return 0.5 ** (max(0, now - at) / DAY / CHANNEL_HALF_LIFE_DAYS)


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def hour_factor(watched_at: list[int], now: int, zone: ZoneInfo) -> float:
    """How much more an interest is watched around this hour than at any hour.

    YouTube pushes an interest at the time of day it is usually consumed. Here, the share of
    the interest's watches within HOUR_WINDOW hours of the current hour, against the share a
    uniform day would give, with PRIOR_WATCHES of uniform watches added so that a few watches
    cannot swing it; neutral (1) below MIN_TIMED_WATCHES watches with a known time.
    """
    if len(watched_at) < MIN_TIMED_WATCHES:
        return 1.0

    def hour(ms: int) -> float:
        local = datetime.fromtimestamp(ms / 1000, zone)
        return local.hour + local.minute / 60

    current = hour(now)
    near = sum(
        min(abs(hour(t) - current), 24 - abs(hour(t) - current)) <= HOUR_WINDOW for t in watched_at
    )
    uniform = (2 * HOUR_WINDOW) / 24
    share = (near + PRIOR_WATCHES * uniform) / (len(watched_at) + PRIOR_WATCHES)
    return max(HOUR_FACTOR_MIN, min(HOUR_FACTOR_MAX, share / uniform))


# A watch of unknown length (an imported Takeout) is taken for a Short when the next watch
# started less than this later, or when its channel's videos of known length are Shorts at
# least at CHANNEL_SHORTS_SHARE (3 of them at least). Measured on three people's Takeouts
# against looked-up lengths:
# 90 s: precision 93-96 %, recall 89-94 %. On the prediction bench (six cuts, frozen lengths)
# Signals' top 10/25/50/100 went from 16.4/27.0/36.1/43.9 % to 22.9/41.6/51.3/56.6 %, and
# with no length looked up at all to 22.4/39.8/47.1/55.4 %; on a real account's Takeout against
# YouTube, YouTube's channels on ours 9.5 -> 12.3 %, ours on YouTube's home 20.2 -> 25.5 %.
SHORT_GAP_S = 90
CHANNEL_SHORTS_SHARE = 0.9


# The "watch again" row (0: none): videos watched on WATCH_AGAIN_DAYS different days or
# more, actually played (the next watch started at least SHORT_GAP_S later, so Shorts scrolled
# past twice do not count), the most watched again and the most recent first. YouTube's home
# shows such videos in "Watch it again", not among its recommendations.
# On four real profiles (`discovery_bench.py`, same candidates): videos of the page then
# watched within 14 days 3.1 -> 6.4 %, 0.6 -> 1.3 %, 3.4 -> 9.5 %; 18.5 %, 6.2 % and 62.5 %
# of the row's videos were watched again; already seen videos elsewhere stay at 0-1.3 %.
WATCH_AGAIN = 8
WATCH_AGAIN_DAYS = 2
WATCH_AGAIN_HORIZON_DAYS = 180


# Exploration, measured on four real profiles: 12-19 %, 17-23 %, 13-19 % and 33-46 % (the
# last one's later watches come 30-40 % from new channels).
EXPLORATION_DAYS = 90
EXPLORATION_MIN_WATCHES = 50


def exploration_rate(history: list[HistoryItem], now: int) -> float | None:
    """`history`: long-form watches (Shorts already left out), newest first."""
    horizon = now - EXPLORATION_DAYS * DAY
    # Too short a history makes every channel look new (a replayed history spans days and
    # was taken for an explorer's: never-watched share 40 -> 63 %): EXPLORATION_DAYS before
    # the window are needed to tell.
    if not history or history[-1].watched_at > horizon - EXPLORATION_DAYS * DAY:
        return None
    end = next((i for i, item in enumerate(history) if item.watched_at < horizon), len(history))
    recent = [item for item in history[:end] if item.channel_url]
    if len(recent) < EXPLORATION_MIN_WATCHES:
        return None
    # Channels watched before the window: read back only until each recent one is found.
    wanted = {item.channel_url for item in recent}
    known: set[str] = set()
    for item in history[end:]:
        if item.channel_url in wanted and item.channel_url not in known:
            known.add(item.channel_url)
            if len(known) == len(wanted):
                break
    new = 0
    for item in reversed(recent):  # oldest first: a channel is new at its first watch
        if item.channel_url not in known:
            new += 1
            known.add(item.channel_url)
    return new / len(recent)


def watched_again(history: list[HistoryItem], now: int) -> list[HistoryItem]:
    # The collector paused for this short pass only: with a long history's objects alive, its
    # passes took 0.40 of the 0.45 s this took. Paused for the whole profile instead, the
    # service kept 300-360 MB more for no time saved on a real page.
    collecting = gc.isenabled()
    gc.disable()
    try:
        return _watched_again(history, now)
    finally:
        if collecting:
            gc.enable()


def _watched_again(history: list[HistoryItem], now: int) -> list[HistoryItem]:
    # Only the last WATCH_AGAIN_HORIZON_DAYS: a score fades by half every 30 days, and reading
    # 261,000 watches took 0.85 s of a page.
    horizon = now - WATCH_AGAIN_HORIZON_DAYS * DAY
    recent = sorted([h for h in history if h.watched_at >= horizon], key=attrgetter("watched_at"))
    pairs: set[tuple[str, int]] = set()
    newest: dict[str, HistoryItem] = {}
    for i, item in enumerate(recent):
        if now - item.watched_at < DAY or _unavailable(item):
            continue
        following = recent[i + 1].watched_at if i + 1 < len(recent) else None
        if following is not None and following - item.watched_at < SHORT_GAP_S * 1000:
            continue
        pairs.add((item.video_url, item.watched_at // DAY))
        newest[item.video_url] = item
    days = Counter(url for url, _ in pairs)
    scores = {
        url: n * _recency(now, newest[url].watched_at)
        for url, n in days.items()
        if n >= WATCH_AGAIN_DAYS
    }
    return [newest[url] for url in sorted(scores, key=lambda u: -scores[u])][:WATCH_AGAIN]


def inferred_shorts(history: list[HistoryItem]) -> set[str]:
    """Ids of watches of unknown length that are Shorts all the same, by how they were watched."""
    by_time = sorted(history, key=attrgetter("watched_at"))
    known: dict[str, list[bool]] = {}
    for item in history:
        if item.duration_s:
            known.setdefault(item.channel_url, []).append(item.is_short or item.duration_s <= 180)
    # Decided once per channel, not once per watch: a channel mostly made of Shorts.
    short_channels = {
        url
        for url, mates in known.items()
        if len(mates) >= 3 and sum(mates) / len(mates) >= CHANNEL_SHORTS_SHARE
    }
    gap = SHORT_GAP_S * 1000
    last = len(by_time) - 1
    found = set()
    for i, item in enumerate(by_time):
        if item.duration_s or item.is_short:
            continue
        swiped = i < last and by_time[i + 1].watched_at - item.watched_at < gap
        if swiped or item.channel_url in short_channels:
            found.add(item.id)
    return found


def _unavailable(item: HistoryItem) -> bool:
    return not item.channel_url and (
        not item.title or item.title.startswith(("http://", "https://"))
    )


def boost_strength(age_min: float, half_life: float, max_age: float) -> float:
    """1 when just watched, halving every half-life, 0 past the maximum age."""
    return 0.0 if age_min > max_age else 0.5 ** (age_min / half_life)
