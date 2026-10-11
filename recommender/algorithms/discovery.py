"""Discovery: your subscriptions, what is close to them and to your favorites, and trends.

Nothing is learned from what you watch. Half the page is new videos from
your subscriptions, `trending_share` is trending videos, and the rest is
videos YouTube relates to your newest subscription videos and to a few of
your favorites. With `random_enabled`, every part is drawn again on each
refresh from a larger set, so the page changes; without it, the same data
always gives the same page. Videos you finished or were already shown
recently are left out, as are muted channels and "not interested" videos.
"""

import random

from ..base import Algorithm, Pool, Run, cap_per_channel, drop_unwanted, make_row
from ..common_specs import max_per_channel, random_enabled, related_per_seed, trending_share
from ..contract import Snapshot, Source
from ..contract_algorithms import Traits
from ..contract_rank import Because, Reason, Row
from ..parameters import Values

PAGE = 80
SUBSCRIPTIONS_SHARE = 0.5
FEED_LIMIT = 120
FAVORITE_SEEDS = 3
# Starting videos drawn from the newest subscription videos at each refresh. Measured on a
# real account: 3 fixed starts with 10 related videos in all gave 7 related videos and a page
# that never changed.
FEED_SEEDS = 6
FEED_SEEDS_AMONG = 18
TRENDING_LIMIT = 40
# With randomness, each part is drawn from this many times the videos it shows.
DRAW_FROM = 2

TITLES = {
    "subscriptions": {"en": "New from your subscriptions", "fr": "Nouveau dans vos abonnements"},
    "related_feed": {"en": "Close to your subscriptions", "fr": "Proche de vos abonnements"},
    "related_favorite": {"en": "Close to {title}", "fr": "Proche de {title}"},
    "trending": {"en": "Trending", "fr": "Tendances"},
}
WHY = {
    "drawn": {
        "en": "{shown} drawn at random among the {pool} newest",
        "fr": "{shown} tirées au sort parmi les {pool} plus récentes",
    },
    "ordered": {
        "en": "the {shown} newest",
        "fr": "les {shown} plus récentes",
    },
    "subscriptions": {
        "en": "Videos from your subscriptions: {how}",
        "fr": "Vidéos de vos abonnements : {how}",
    },
    "related_feed": {
        "en": "Videos YouTube relates to your newest subscription videos: {how}",
        "fr": "Vidéos que YouTube associe à vos dernières vidéos d'abonnements : {how}",
    },
    "related_favorite": {
        "en": "Videos YouTube relates to your favorite {title}: {how}",
        "fr": "Vidéos que YouTube associe à votre favori {title} : {how}",
    },
    "trending": {
        "en": "Trending videos in your country: {how}",
        "fr": "Vidéos en tendance dans votre pays : {how}",
    },
}


class DiscoveryRanking:
    """One page of Discovery being ranked, step by step."""

    def __init__(self, algorithm: "Discovery", run: Run):
        self.algorithm = algorithm
        self.run = run
        self.values = self.run.values
        self.randomized = self.values.flag("random_enabled")
        self.rng = self.run.rng(self.randomized)
        self.pool = Pool.gather(self.run.groups)
        drop_unwanted(self.run, self.pool, seen=True, shorts=False)

    def rows(self) -> list[Row]:
        self._share()
        return self._build()

    def _share(self) -> None:
        """The parts of the page, each with its share: a video goes to the first that claims it."""

        def newest(urls: list[str]) -> list[str]:
            return sorted(urls, key=lambda u: self.pool.videos[u].uploaded_at or 0, reverse=True)

        def take(urls: list[str], count: int) -> tuple[list[str], str]:
            """The part's videos, and how they were chosen, for the trace."""
            if not self.randomized or len(urls) <= count:
                chosen = urls[:count]
                return chosen, self.run.text(WHY["ordered"], shown=len(chosen))
            candidates = urls[: count * DRAW_FROM]
            picked = set(self.rng.sample(candidates, count))
            chosen = [u for u in candidates if u in picked]
            return chosen, self.run.text(WHY["drawn"], shown=len(chosen), pool=len(candidates))

        trending_count = round(PAGE * self.values.number("trending_share"))
        subscriptions_count = (
            round(PAGE * SUBSCRIPTIONS_SHARE) if self.run.snapshot.subscriptions else 0
        )
        related_count = PAGE - trending_count - subscriptions_count

        # Each video belongs to the first part that claims it: subscriptions, related, trending.
        claimed: set[str] = set()

        def unclaimed(urls: list[str]) -> list[str]:
            return [u for u in urls if u not in claimed]

        self.parts: list[tuple[str, str | None, Reason, str, list[str]]] = []

        feed = newest(self.pool.from_kind("subscription_feed"))
        chosen, how = take(feed, subscriptions_count)
        claimed.update(chosen)
        self.parts.append(
            ("subscriptions", None, Reason(kind="new_from_subscriptions"), how, chosen)
        )

        related_groups = [g for g in self.run.groups if g.source.kind == "related"]
        # Close to your subscriptions means other channels: the subscribed ones have their part.
        subscribed = {sub.channel_url for sub in self.run.snapshot.subscriptions}
        per_group = max(1, related_count // max(1, len(related_groups)))
        for group in related_groups:
            urls = unclaimed(
                [
                    v.url
                    for v in group.videos
                    if v.url in self.pool.videos and v.uploader_url not in subscribed
                ]
            )
            chosen, how = take(urls, per_group)
            claimed.update(chosen)
            key = "related_feed" if group.source.of else "related_favorite"
            self.parts.append(
                (key, group.source.ref, Reason(kind="discovery", ref=group.source.ref), how, chosen)
            )

        trending = unclaimed(self.pool.from_kind("trending"))
        chosen, how = take(trending, trending_count)
        claimed.update(chosen)
        self.parts.append(("trending", None, Reason(kind="trending"), how, chosen))

    def _build(self) -> list[Row]:
        """One row per part that got videos, under the cap per channel."""
        cap = self.values.whole("max_per_channel")
        ordered = cap_per_channel(
            self.run, [u for *_, urls in self.parts for u in urls], self.pool, cap
        )
        allowed = set(ordered)
        favorite_titles = {f.id: f.title or f.video_url for f in self.run.snapshot.favorites}
        subscription_of = {sub.channel_url: sub.id for sub in self.run.snapshot.subscriptions}

        rows: list[Row] = []
        for index, (key, ref, reason, how, urls) in enumerate(self.parts):
            urls = [u for u in urls if u in allowed]
            if not urls:
                continue
            row_id = f"discovery-{key}-{index}"
            title_values = {"title": favorite_titles.get(ref or "", "")}
            if key == "subscriptions":
                videos = [
                    (
                        u,
                        [
                            Because(
                                kind="subscriptions",
                                ref=subscription_of.get(self.pool.videos[u].uploader_url),
                                weight=1.0,
                            )
                        ],
                    )
                    for u in urls
                ]
            else:
                videos = [(u, [Because(kind=reason.kind, ref=ref, weight=1.0)]) for u in urls]
            rows.append(
                make_row(row_id, self.run.text(TITLES[key], **title_values), reason, videos)
            )
            self.run.explain(row_id, WHY[key], how=how, **title_values)
        return rows


class Discovery(Algorithm):
    id = "discovery"
    name = {"en": "Discovery", "fr": "Découverte"}
    description = {
        "en": "Your subscriptions, videos related to them and to your favorites, and trending "
        "videos, rotated on every refresh.",
        "fr": "Les abonnements, des vidéos liées aux abonnements et aux favoris, et les "
        "tendances. La sélection change à chaque rafraîchissement.",
    }
    uses = ["subscriptions", "favorites", "not_interested", "shown"]
    sources = ["subscription_feed", "related", "trending"]
    random = True
    traits = Traits(personal=1, discovery=3, variety=3)
    guide = {
        "for_you_if": {
            "en": "You like your subscriptions but also want to come across other things, "
            "without the home page learning from what you watch.",
            "fr": "Vous aimez vos abonnements mais voulez aussi tomber sur autre chose, sans que "
            "l'accueil apprenne de ce que vous regardez.",
        },
        "you_will_see": {
            "en": "Half subscriptions, the rest videos related to them and to your favorites, "
            "plus trends, drawn again at every refresh.",
            "fr": "Moitié abonnements, le reste en vidéos liées à vos abonnements et favoris, plus "
            "des tendances, retirées au sort à chaque rafraîchissement.",
        },
        "good_to_know": {
            "en": "It reads your subscriptions and favorites, never your history. Suggestions "
            "stay broad.",
            "fr": "Il lit vos abonnements et vos favoris, jamais votre historique. Les suggestions "
            "restent larges.",
        },
    }
    specs = [
        related_per_seed(
            default=10,
            maximum=30,
            label_en="Related videos per starting video",
            label_fr="Vidéos liées par vidéo de départ",
        ),
        trending_share(default=0.2, maximum=0.5, featured=True),
        max_per_channel(default=6, maximum=12),
        random_enabled("Rotate on refresh", "Varier à chaque rafraîchissement"),
    ]

    def plan(self, snapshot: Snapshot, values: Values) -> list[Source]:
        related = values.whole("related_per_seed")
        sources: list[Source] = []
        if snapshot.subscriptions:
            sources.append(Source(kind="subscription_feed", limit=FEED_LIMIT))
            if related:
                among = FEED_SEEDS_AMONG if values.flag("random_enabled") else None
                sources.append(
                    Source(
                        kind="related",
                        of="subscription_feed",
                        count=FEED_SEEDS,
                        among=among,
                        limit=related,
                    )
                )
        if related and snapshot.favorites:
            favorites = list(snapshot.favorites)
            if values.flag("random_enabled"):
                random.Random(snapshot.context.seed).shuffle(favorites)
            for favorite in favorites[:FAVORITE_SEEDS]:
                sources.append(
                    Source(
                        kind="related", video_url=favorite.video_url, limit=related, ref=favorite.id
                    )
                )
        if values.number("trending_share") > 0:
            sources.append(Source(kind="trending", limit=TRENDING_LIMIT))
        return sources

    def rank(self, run: Run) -> list[Row]:
        return DiscoveryRanking(self, run).rows()
