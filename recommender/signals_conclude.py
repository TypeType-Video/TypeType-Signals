"""What Signals concludes from one snapshot: channels, interest groups, recent watches."""

from operator import attrgetter

from recommender.signals_groups import InterestGroups

from .contract import HistoryItem, Snapshot
from .embeddings import Embedder
from .parameters import Values
from .signals_history import (
    WATCH_AGAIN,
    _recency,
    _unavailable,
    boost_strength,
    exploration_rate,
    inferred_shorts,
    watched_again,
)
from .signals_profile import (
    CHANNEL_FLOOR,
    CHANNEL_HALF_LIFE_DAYS,
    CONTINUE_DAYS,
    CONTINUE_MIN_SHARE,
    DAY,
    FINISHED_SHARE,
    LIKE_WEIGHT,
    MAX_BOOSTS,
    MAX_LATEST,
    MINUTE,
    SHORT_CHANNEL_WEIGHT,
    SHORTS_ONLY_FOR_LONG_FORM_CHANNELS,
    Boost,
    Channel,
    Conclusions,
    Group,
    channel_id,
)


def _share(item: HistoryItem) -> float:
    """How much of the video was watched; unknown progress (e.g. a live) counts as half."""
    return 0.5 if item.watched_share is None else item.watched_share


class Concluding:
    """What Signals concludes from one snapshot, worked out in steps."""

    def __init__(self, snapshot: Snapshot, values: Values, embedder: Embedder):
        self.snapshot = snapshot
        self.values = values
        self.embedder = embedder
        self.now = self.snapshot.context.now
        self.overrides = self.snapshot.overrides
        self.ignored = set(self.overrides.ignored_signals)
        self.shorts = self.values.flag("use_shorts_history")
        self.seed_share = self.values.number("seed_min_watched")
        self.lasting_share = self.values.number("long_term_min_watched")

        # A removed or private video comes from a Takeout with no channel and its own URL as
        # title (13 % of a measured account's watches): it says nothing about anyone's interests.
        self.inferred = set() if self.shorts else inferred_shorts(self.snapshot.history)
        self.history = [
            item
            for item in self.snapshot.history
            if (self.shorts or not (item.is_short or item.id in self.inferred))
            and (item.channel_url or not _unavailable(item))  # with a channel it is available
        ]
        self.history.sort(key=attrgetter("watched_at"), reverse=True)

    def conclusions(self) -> Conclusions:
        self._channels()
        self._groups()
        self._boosts()
        return self._rest()

    def _channels(self) -> None:
        """The channels the person watches: one point per different video, fading with time."""
        self.channels: dict[str, Channel] = {}
        self._count_watches()
        self._count_likes()
        if SHORT_CHANNEL_WEIGHT and not self.shorts:
            self._count_shorts()
        self.ranked = sorted(self.channels.values(), key=lambda c: (-c.score, c.id))
        for channel in self.ranked:
            channel.ignored = channel.id in self.ignored
        counted = [c for c in self.ranked if not c.ignored]
        total = sum(c.score for c in counted) or 1.0
        for channel in counted:
            channel.weight = channel.score / total

    def _channel(self, url: str, name: str = "") -> Channel:
        """The channel of this address, opened at zero the first time it is met."""
        channel = self.channels.get(url)
        if channel is None:
            channel = self.channels[url] = Channel(channel_id(url), url, name, 0.0, [])
        return channel

    def _groups(self) -> None:
        self.groups: list[Group] = InterestGroups(self).groups()

    def _count_watches(self) -> None:
        """Each different video watched far enough gives its channel points.

        This loop and the next run once per watch and per like of the whole history: what
        they need is read once before them and nothing is called per item that can be written
        in place (`_share`, `_recency` and `_channel` are spelled out here, measured)."""
        now, seed_share, lasting_share = self.now, self.seed_share, self.lasting_share
        channels = self.channels
        seen_videos: set[str] = set()
        for item in self.history:
            share = 0.5 if item.watched_share is None else item.watched_share
            url = item.channel_url
            if share < seed_share or not url or item.video_url in seen_videos:
                continue
            seen_videos.add(item.video_url)
            age = now - item.watched_at
            recency = 0.5 ** ((age if age > 0 else 0) / DAY / CHANNEL_HALF_LIFE_DAYS)
            if recency < CHANNEL_FLOOR:
                recency = CHANNEL_FLOOR
            channel = channels.get(url)
            if channel is None:
                channel = channels[url] = Channel(channel_id(url), url, item.channel_name, 0.0, [])
            channel.score += (1.0 if share >= lasting_share else 0.5) * recency
            if not channel.name:
                channel.name = item.channel_name
            channel.refs.append(item.id)

    def _count_likes(self) -> None:
        """A like gives its channel points too. A liked Short says as little as a watched one
        (on the measured account, 80-100 % of the likes whose length is known before each cut
        were Shorts, and channels of liked Shorts held most of Signals' top 10 where it lost)."""
        now, channels = self.now, self.channels
        self.favorites = [f for f in self.snapshot.favorites if self.shorts or not f.is_short]
        for favorite in self.favorites:
            url = favorite.channel_url
            if not url:
                continue
            channel = channels.get(url)
            if channel is None:
                channel = channels[url] = Channel(channel_id(url), url, "", 0.0, [])
            at = favorite.added_at
            # A dated like fades like a watch; an undated one (older Server) counts in full.
            liked = 1.0 if at is None else 0.5 ** (max(0, now - at) / DAY / CHANNEL_HALF_LIFE_DAYS)
            channel.score += LIKE_WEIGHT * liked
            channel.refs.append(favorite.id)

    def _count_shorts(self) -> None:
        """Shorts left out of the watches still say a little about their channel: one person's
        channels watched mostly through Shorts were then watched in long form, and Signals had
        them at rank 43-344 where counting every watch had them in the top 25."""
        seen_shorts: set[str] = set()
        for item in self.snapshot.history:
            if not (item.is_short or item.id in self.inferred) or not item.channel_url:
                continue
            if item.video_url in seen_shorts or _unavailable(item):
                continue
            seen_shorts.add(item.video_url)
            if item.channel_url not in self.channels and SHORTS_ONLY_FOR_LONG_FORM_CHANNELS:
                continue
            channel = self._channel(item.channel_url, item.channel_name)
            channel.score += SHORT_CHANNEL_WEIGHT * max(
                _recency(self.now, item.watched_at), CHANNEL_FLOOR
            )
            channel.name = channel.name or item.channel_name

    def _boosts(self) -> None:
        """What was just watched, fading by the half-life."""
        # --- Recent boosts: what was just watched, fading by the half-life.
        self.boosts: list[Boost] = []
        if self.values.flag("session_boost"):
            half_life = self.values.number("session_half_life")
            max_age = self.values.number("session_max_age")
            reset = self.overrides.session_reset_at or 0
            picked: set[str] = set()
            for item in self.history:  # newest first: past the session's age, nothing is left
                if self.now - item.watched_at > max_age * MINUTE:
                    break
                if (
                    item.watched_at < reset
                    or item.video_url in picked
                    or _share(item) < self.seed_share
                ):
                    continue
                if item.id in self.ignored:
                    continue
                strength = boost_strength((self.now - item.watched_at) / MINUTE, half_life, max_age)
                if strength <= 0:
                    continue
                picked.add(item.video_url)
                # A full watch pulls harder than a partial one.
                strength *= 1.0 if _share(item) >= self.lasting_share else 0.6
                self.boosts.append(Boost(item, strength, int(item.watched_at + max_age * MINUTE)))
            self.boosts = sorted(self.boosts, key=lambda b: -b.strength)[:MAX_BOOSTS]

    def _rest(self) -> Conclusions:
        """Videos started and not finished, the latest watches, and the whole put together."""
        # --- Started and not finished.
        started: list[HistoryItem] = []
        urls: set[str] = set()
        for item in self.history:  # newest first: past CONTINUE_DAYS, nothing is fresh any more
            if self.now - item.watched_at > CONTINUE_DAYS * DAY:
                break
            if item.video_url in urls or item.is_short:
                continue
            urls.add(item.video_url)
            watched = item.watched_share
            if watched is not None and CONTINUE_MIN_SHARE <= watched < FINISHED_SHARE:
                started.append(item)

        # --- Latest watches: YouTube's home keeps drawing on them after the boost has faded.
        latest: list[HistoryItem] = []
        reset = self.overrides.session_reset_at or 0
        for item in self.history:
            if len(latest) >= MAX_LATEST:
                break
            eligible = (
                not item.is_short
                and _share(item) >= self.seed_share
                and item.id not in self.ignored
            )
            if (
                eligible
                and item.watched_at >= reset
                and item.video_url not in {i.video_url for i in latest}
            ):
                latest.append(item)

        rewatched = watched_again(self.snapshot.history, self.now) if WATCH_AGAIN else []
        exploration = exploration_rate(self.history, self.now)
        known = {item.channel_url for item in self.snapshot.history if item.channel_url} | {
            s.channel_url for s in self.snapshot.subscriptions
        }
        return Conclusions(
            channels=self.ranked,
            groups=self.groups,
            boosts=self.boosts,
            started=started,
            latest=latest,
            known_channels=known,
            rewatched=rewatched,
            exploration=exploration,
        )


def conclude(snapshot: Snapshot, values: Values, embedder: Embedder) -> Conclusions:
    return Concluding(snapshot, values, embedder).conclusions()
