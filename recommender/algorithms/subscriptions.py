"""Subscriptions: the newest videos of the channels you follow, newest first.

Nothing is deduced about you. The page is the subscription feed Server
fetched, without muted channels or videos marked "not interested", with at
most `max_per_channel` videos per channel, split into three rows by upload
day in your time zone: today, the last seven days, and earlier.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..base import Algorithm, Pool, Run, cap_per_channel, drop_unwanted, make_row
from ..common_specs import include_shorts, max_per_channel
from ..contract import Snapshot, Source
from ..contract_algorithms import Traits
from ..contract_rank import Because, Reason, Row
from ..parameters import Values

FEED_LIMIT = 150

ROWS = [
    ("today", {"en": "Today", "fr": "Aujourd'hui"}),
    ("week", {"en": "This week", "fr": "Cette semaine"}),
    ("earlier", {"en": "Earlier", "fr": "Plus ancien"}),
]
ROW_WHY = {
    "today": {
        "en": "{count} videos from your subscriptions uploaded since midnight",
        "fr": "{count} vidéos de vos abonnements publiées depuis minuit",
    },
    "week": {
        "en": "{count} videos from your subscriptions uploaded in the last 7 days",
        "fr": "{count} vidéos de vos abonnements publiées ces 7 derniers jours",
    },
    "earlier": {
        "en": "{count} older videos from your subscriptions",
        "fr": "{count} vidéos plus anciennes de vos abonnements",
    },
}


def day_start(now_ms: int, timezone: str) -> int:
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        zone = ZoneInfo("UTC")
    now = datetime.fromtimestamp(now_ms / 1000, zone)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.timestamp() * 1000)


class Subscriptions(Algorithm):
    id = "subscriptions"
    name = {"en": "Subscriptions", "fr": "Abonnements"}
    description = {
        "en": "Newest videos from the channels you follow, in upload order.",
        "fr": "Les dernières vidéos des chaînes suivies, dans l'ordre de publication.",
    }
    uses = ["subscriptions", "not_interested"]
    sources = ["subscription_feed"]
    random = False
    traits = Traits(personal=0, discovery=0, variety=0)
    guide = {
        "for_you_if": {
            "en": "You only want the channels you chose, in chronological order.",
            "fr": "Vous voulez seulement les chaînes que vous avez choisies, dans l'ordre "
            "chronologique.",
        },
        "you_will_see": {
            "en": "New videos from your subscriptions, newest first, and nothing else.",
            "fr": "Les nouvelles vidéos de vos abonnements, les plus récentes d'abord, et rien "
            "d'autre.",
        },
        "good_to_know": {
            "en": "No discovery: if you follow few channels, the page stays short. Nothing about "
            "what you watch is used.",
            "fr": "Aucune découverte : si vous suivez peu de chaînes, la page reste courte. Rien "
            "de ce que vous regardez n'est utilisé.",
        },
    }
    # YouTube's chronological subscription feed has no cap per channel: a cap of 20 lost 30 of
    # 375 videos on a real feed. The default is high enough to lose none; it can be lowered.
    specs = [
        max_per_channel(default=100, maximum=100),
        include_shorts(default=False, featured=True),
    ]

    def plan(self, snapshot: Snapshot, values: Values) -> list[Source]:
        if not snapshot.subscriptions:
            return []
        return [Source(kind="subscription_feed", limit=FEED_LIMIT)]

    def rank(self, run: Run) -> list[Row]:
        pool = Pool.gather([g for g in run.groups if g.source.kind == "subscription_feed"])
        drop_unwanted(run, pool, seen=False, shorts=run.values.flag("include_shorts"))
        newest = sorted(
            pool.urls(), key=lambda url: pool.videos[url].uploaded_at or 0, reverse=True
        )
        kept = cap_per_channel(run, newest, pool, run.values.whole("max_per_channel"))

        subscription_of = {s.channel_url: s.id for s in run.snapshot.subscriptions}
        context = run.snapshot.context
        today = day_start(context.now, context.timezone)
        week = today - int(timedelta(days=6).total_seconds() * 1000)
        buckets: dict[str, list[str]] = {key: [] for key, _ in ROWS}
        for url in kept:
            uploaded = pool.videos[url].uploaded_at
            key = (
                "earlier"
                if not uploaded
                else "today"
                if uploaded >= today
                else ("week" if uploaded >= week else "earlier")
            )
            buckets[key].append(url)

        rows: list[Row] = []
        for key, title in ROWS:
            if not buckets[key]:
                continue
            row_id = f"subscriptions-{key}"
            videos = [
                (
                    url,
                    [
                        Because(
                            kind="subscriptions",
                            ref=subscription_of.get(pool.videos[url].uploader_url),
                            weight=1.0,
                        )
                    ],
                )
                for url in buckets[key]
            ]
            rows.append(
                make_row(
                    row_id,
                    run.text(title),
                    Reason(kind="new_from_subscriptions"),
                    videos,
                )
            )
            run.explain(row_id, ROW_WHY[key], count=len(videos))
        return rows
