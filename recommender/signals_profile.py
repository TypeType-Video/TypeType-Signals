"""What Signals concludes from a snapshot: channels, interest groups, recent boosts.

This is the single place those conclusions are computed. The profile
endpoint shows them, and ranking uses them, so what the user sees on the
"what it remembers" screen is exactly what shapes their page.

What was measured on a real account shaped
the rules below:
- the channel matters more than the watch time: a channel counts by the
  number of different videos watched from it, not by minutes;
- a video just watched steers the page for a few hours, then fades;
- a short watch brings the video back, a full watch brings its channel;
- Shorts shape a profile of their own, so they are left out by default.
"""

from dataclasses import dataclass, field

import numpy as np

from . import settings
from .contract import HistoryItem, canonical_channel_url

MINUTE = 60_000
DAY = 24 * 60 * MINUTE
# A channel's weight halves every 30 days without a new video watched from it.
CHANNEL_HALF_LIFE_DAYS = 30
# The least a watch keeps for its channel however old it is, so channels watched for years stay near
# the top. Measured on six cut dates of one person's 26 months: with 0.1 (and every watch counting),
# Signals' top 10/25/50/100 held 14.1/21.3/31.0/40.9 % of the next two weeks' long-form watches,
# against 9.3/19.0/25.6/38.4 % fading to nothing and 13.0/18.8/26.3/34.7 % for "most watched
# channels"; 0.2 and 0.3 lost at 25 and 50. Neutral on a real account against YouTube.
CHANNEL_FLOOR = 0.1
# What a like adds to its channel, against 1 for a watch seen through (a watch of unknown
# progress adds 0.5). A like is weak evidence of what someone goes on to watch: over six cut
# dates Signals' top
# 10/25/50/100 held 14.1/21.3/31.2/41.3 % of the next two weeks' long-form watches with 1,
# 16.3/25.9/34.8/42.8 % with 0.25 and 16.3/26.6/35.4/43.6 % with 0.1 (0: the same). Kept
# above 0 so a history with few watches still has its likes to go on; likes still form
# interest groups whatever this weight.
LIKE_WEIGHT = 0.1
# What an imported watch (progress unknown) weighs in the interest groups, against 1 for a
# watch seen through; 0 keeps them out of the groups (they still count for their channel).
# A rule let them in for someone with few likes (under 100). It gave such a person interests,
# and pages of new channels they never went on to watch: on the page bench without it, known
# channels then watched +2.4 points (a Spanish profile with 8 likes: +12.0), real discoveries
# unchanged. Removed.
IMPORTED_GROUP_WEIGHT = 0.0
# What a watched Short adds to its channel (not to interests), against 1 for a long video
# seen through. 0: Shorts say nothing.
SHORT_CHANNEL_WEIGHT = 0.0
# Measured on three profiles (#236-#241): 0.1 and 0.25 for every channel lost 1-9 points at
# every depth. With this, Shorts only add to a channel also watched in long form.
SHORTS_ONLY_FOR_LONG_FORM_CHANNELS = True
# Titles this close (cosine) belong to the same interest group.
GROUP_SIMILARITY = 0.55
MAX_BOOSTS = 5
MAX_CHANNELS = 8
# Videos started this recently and left unfinished are offered again.
CONTINUE_DAYS = 14
CONTINUE_MIN_SHARE = 0.05
FINISHED_SHARE = 0.9
MAX_GROUP_REFS = 12
MAX_LATEST = 3
# Time of day: only watches whose time is known within this many seconds count.
TIME_PRECISION_S = 2 * 3600
HOUR_WINDOW = 2
MIN_TIMED_WATCHES = 3
PRIOR_WATCHES = 4
HOUR_FACTOR_MIN = 0.3
HOUR_FACTOR_MAX = 4.0
MAX_LASTING = 1000
# Titles clustered at most (lasting watches and likes, the most recent first): the clustering
# keeps an n x n table, so this bounds its memory (~8 bytes a pair) and time for any history.
MAX_TITLES = settings.whole("RECOMMENDER_MAX_TITLES", 2000)


def channel_id(url: str) -> str:
    return "channel:" + canonical_channel_url(url).rstrip("/").rsplit("/", 1)[-1]


def ref_tail(ref: str) -> str:
    return ref.split(":", 1)[-1]


@dataclass
class Channel:
    id: str
    url: str
    name: str
    score: float
    refs: list[str]
    weight: float = 0.0
    ignored: bool = False


@dataclass
class Group:
    id: str
    label: str
    members: list[str]
    centroid: np.ndarray
    score: float
    named: bool = True
    # The words its titles share, most typical first; what a topic search asks for.
    keywords: list[str] = field(default_factory=list)
    weight: float = 0.0
    pinned: bool = False
    muted: bool = False
    ignored: bool = False
    # How much it counts at this hour of the day compared with any other (1: no difference).
    hour_factor: float = 1.0

    @property
    def weight_now(self) -> float:
        return self.weight * self.hour_factor


@dataclass
class Boost:
    item: HistoryItem
    strength: float
    expires_at: int


@dataclass
class Conclusions:
    channels: list[Channel] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    boosts: list[Boost] = field(default_factory=list)
    started: list[HistoryItem] = field(default_factory=list)
    # The newest videos watched enough to count, whatever their age.
    latest: list[HistoryItem] = field(default_factory=list)
    # Every channel the user watched or follows, to tell new creators apart.
    known_channels: set[str] = field(default_factory=set)
    # Videos watched again on other days (music, recitations): the "watch again" row.
    rewatched: list[HistoryItem] = field(default_factory=list)
    # How much this person explores: the share of their long-form watches of the last
    # EXPLORATION_DAYS from a channel they had never watched before (None: too few).
    exploration: float | None = None
    _by_url: dict[str, Channel] | None = field(default=None, repr=False, compare=False)

    def channel(self, url: str) -> Channel | None:
        # Asked once per candidate: an index, not a scan of thousands of channels.
        if self._by_url is None:
            self._by_url = {}
            for c in self.channels:
                self._by_url.setdefault(c.url, c)
        return self._by_url.get(url)
