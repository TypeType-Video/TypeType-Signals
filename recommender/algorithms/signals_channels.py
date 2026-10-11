"""Signals, channels: the creators you watch, and nothing else.

Reads your history the way Signals does (the same weights per channel: what you watch most,
fading with time but never to nothing, Shorts left out) and fills the page with the newest
videos of the channels you watch most, then with what is new from your subscriptions. No
search, no related videos, no new creators: every video comes from a channel you know.

With "vary the page" off, the same data gives the same page; on (the default), chance
weighs a little against the score, so a refresh shows another selection. Shorts, videos you
finished, muted channels and "not interested" videos are left out.
"""

from ..base import Algorithm, Pool, Run, drop_unwanted, make_row, score
from ..common_specs import max_per_channel, page_size, random_enabled
from ..contract import Snapshot, Source
from ..contract_algorithms import Traits
from ..contract_profile import Profile, ProfileChannel
from ..contract_rank import Because, Reason, Row
from ..parameters import Number, Values
from ..signals_profile import MAX_CHANNELS, Conclusions
from .signals import Signals

DAY_MS = 86_400_000
# Channels whose page is asked for, heaviest first.
CHANNELS = 20
FEED_LIMIT = 60
# A video's age halves what its freshness adds every this many days.
FRESH_HALF_LIFE_DAYS = 30
# How much chance weighs against the score when the page varies on refresh.
DRAW = 0.1
W_CHANNEL = 0.8
W_FRESH = 0.2
# What a subscribed channel never watched is worth against the most watched one.
SUBSCRIBED_ONLY = 0.1

TITLES = {
    "channels": {"en": "From the creators you watch", "fr": "Des créateurs que vous regardez"},
    "subscriptions": {"en": "New from your subscriptions", "fr": "Nouveau dans vos abonnements"},
}
WHY = {
    "channels": {
        "en": "Newest videos of the {channels} channels you watch most, the most watched first",
        "fr": "Dernières vidéos des {channels} chaînes que vous regardez le plus, les plus "
        "regardées d'abord",
    },
    "subscriptions": {
        "en": "Newest videos of subscriptions not shown above",
        "fr": "Dernières vidéos des abonnements absents de la ligne du dessus",
    },
}


class ChannelsRanking:
    """One page of Signals by channels being ranked, step by step."""

    def __init__(self, algorithm: "SignalsChannels", run: Run):
        self.algorithm = algorithm
        self.run = run
        self.conclusions = self.algorithm.conclude(self.run.snapshot)
        self.now = self.run.snapshot.context.now
        self.pool = Pool.gather(self.run.groups)
        drop_unwanted(self.run, self.pool, seen=True, shorts=False)
        self.subscribed = {sub.channel_url: sub.id for sub in self.run.snapshot.subscriptions}
        self.heaviest = (
            max((ch.weight for ch in self.conclusions.channels if not ch.ignored), default=0.0)
            or 1.0
        )

        self.varies = self.run.values.flag("random_enabled")
        self.rng = self.run.rng(self.varies)

    def rows(self) -> list[Row]:
        self._score()
        self._place()
        return self._build()

    def _score(self) -> None:
        """The score of each candidate, from how much its channel is watched."""
        self.totals: dict[str, float] = {}
        for url in self.pool.urls():
            video = self.pool.videos[url]
            channel = self.conclusions.channel(video.uploader_url)
            if channel is not None and not channel.ignored:
                weight = channel.weight / self.heaviest
            elif video.uploader_url in self.subscribed:
                weight = SUBSCRIBED_ONLY
            else:
                continue  # a channel the person does not know: not this algorithm's business
            age_days = (
                max(0, self.now - video.uploaded_at) / DAY_MS
                if video.uploaded_at
                else FRESH_HALF_LIFE_DAYS
            )
            self.totals[url] = score(
                self.run,
                url,
                {
                    "channel": W_CHANNEL * weight,
                    "freshness": W_FRESH * 0.5 ** (age_days / FRESH_HALF_LIFE_DAYS),
                    **({"draw": DRAW * self.rng.random()} if self.varies else {}),
                },
            )

    def _place(self) -> None:
        """The best scores first, under the cap per channel: watched channels, then the feed."""
        size = self.run.values.whole("page_size")
        cap = self.run.values.whole("max_per_channel")
        per_channel: dict[str, int] = {}
        self.watched_row: list[str] = []
        self.feed_row: list[str] = []
        for url in sorted(self.totals, key=lambda u: -self.totals[u]):
            if len(self.watched_row) + len(self.feed_row) >= size:
                break
            channel_url = self.pool.videos[url].uploader_url
            if per_channel.get(channel_url, 0) >= cap:
                self.run.drop(url, "channel_cap")
                continue
            per_channel[channel_url] = per_channel.get(channel_url, 0) + 1
            (
                self.watched_row
                if self.conclusions.channel(channel_url) is not None
                else self.feed_row
            ).append(url)

    def _build(self) -> list[Row]:
        """The row of watched channels, then the row of the subscription feed."""
        rows: list[Row] = []
        if self.watched_row:
            rows.append(
                make_row(
                    "channels",
                    self.run.text(TITLES["channels"]),
                    Reason(kind="same_channel"),
                    [
                        (
                            url,
                            [
                                Because(
                                    kind="same_channel",
                                    ref=self.conclusions.channel(
                                        self.pool.videos[url].uploader_url
                                    ).id,
                                    weight=round(self.totals[url], 3),
                                )
                            ],
                        )
                        for url in self.watched_row
                    ],
                )
            )
            self.run.explain(
                "channels",
                WHY["channels"],
                channels=len({self.pool.videos[u].uploader_url for u in self.watched_row}),
            )
        if self.feed_row:
            rows.append(
                make_row(
                    "channels-subscriptions",
                    self.run.text(TITLES["subscriptions"]),
                    Reason(kind="new_from_subscriptions"),
                    [
                        (
                            url,
                            [
                                Because(
                                    kind="subscriptions",
                                    ref=self.subscribed.get(self.pool.videos[url].uploader_url),
                                    weight=1.0,
                                )
                            ],
                        )
                        for url in self.feed_row
                    ],
                )
            )
            self.run.explain("channels-subscriptions", WHY["subscriptions"])
        return rows


class SignalsChannels(Algorithm):
    id = "signals-channels"
    name = {"en": "Signals, channels", "fr": "Signals, chaînes"}
    description = {
        "en": "The newest videos of the creators you watch most, and of your subscriptions.",
        "fr": "Les dernières vidéos des créateurs que vous regardez le plus, et de vos "
        "abonnements.",
    }
    uses = ["history", "subscriptions", "favorites", "not_interested", "shown"]
    sources = ["channel_videos", "subscription_feed"]
    random = True
    traits = Traits(personal=3, discovery=0, variety=1)
    guide = {
        "for_you_if": {
            "en": "You come for the creators you already follow or watch.",
            "fr": "Vous venez pour les créateurs que vous suivez ou regardez déjà.",
        },
        "you_will_see": {
            "en": "Only channels you know: the ones you watch most first.",
            "fr": "Uniquement des chaînes que vous connaissez : celles que vous regardez le plus "
            "d'abord.",
        },
        "good_to_know": {
            "en": "It never shows a creator you have not watched or followed.",
            "fr": "Il ne montre jamais un créateur que vous n'avez ni regardé ni suivi.",
        },
    }
    specs = [
        Number(
            id="videos_per_channel",
            group="sources",
            label={"en": "Videos asked per channel", "fr": "Vidéos demandées par chaîne"},
            help={
                "en": "Newest videos asked from each of the channels you watch most.",
                "fr": "Dernières vidéos demandées à chacune des chaînes que vous regardez le plus.",
            },
            min=2,
            max=20,
            step=1,
            default=8,
        ),
        max_per_channel(default=5, maximum=12),
        page_size(),
        random_enabled(
            "Vary the page on every refresh", "Varier la page à chaque rafraîchissement"
        ),
    ]

    def __init__(self, signals: Signals | None = None):
        # The channel weights are Signals' own: one profile for both, worked out once a snapshot.
        self._signals = signals or Signals()

    def conclude(self, snapshot: Snapshot) -> Conclusions:
        return self._signals.conclude(snapshot, Values(self._signals.specs, {}))

    def profile(self, snapshot: Snapshot, values: Values) -> Profile:
        c = self.conclude(snapshot)
        return Profile(
            channels=[
                ProfileChannel(id=ch.id, name=ch.name, weight=round(ch.weight, 4), from_=ch.refs)
                for ch in c.channels[:MAX_CHANNELS]
            ]
        )

    def plan(self, snapshot: Snapshot, values: Values) -> list[Source]:
        c = self.conclude(snapshot)
        muted = set(snapshot.overrides.muted_channels)
        channels = sorted(
            (ch for ch in c.channels if not ch.ignored and ch.url and ch.url not in muted),
            key=lambda ch: (-ch.weight, ch.id),
        )
        sources = [
            Source(
                kind="channel_videos",
                channel_url=ch.url,
                sort="latest",
                limit=values.whole("videos_per_channel"),
                ref=ch.id,
            )
            for ch in channels[:CHANNELS]
        ]
        if snapshot.subscriptions:
            sources.append(Source(kind="subscription_feed", limit=FEED_LIMIT))
        return sources

    def rank(self, run: Run) -> list[Row]:
        return ChannelsRanking(self, run).rows()
