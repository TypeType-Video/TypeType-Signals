"""Signals, mix: part channels, part subjects.

Asks for what "Signals, channels" and "Signals, videos" each ask for, lets each rank its own
candidates, and gives `videos_share` of the page to the subjects and the rest to the channels
you watch. A video both would show is shown once, with the channels. If one of the two has
nothing to show (no subject found, no known channel), the other takes the whole page.
"""

from ..base import Algorithm, Run
from ..common_specs import page_size, random_enabled
from ..contract import Snapshot, Source
from ..contract_algorithms import Traits
from ..contract_profile import Profile
from ..contract_rank import Row
from ..parameters import Number, Values
from .signals_channels import SignalsChannels
from .signals_videos import SignalsVideos


class SignalsMix(Algorithm):
    id = "signals-mix"
    name = {"en": "Signals, mix", "fr": "Signals, mélange"}
    description = {
        "en": "The creators you watch, and videos on your subjects from any creator.",
        "fr": "Les créateurs que vous regardez, et des vidéos sur vos sujets de tout créateur.",
    }
    uses = ["history", "subscriptions", "favorites", "not_interested", "shown"]
    sources = ["channel_videos", "subscription_feed", "search", "related"]
    random = True
    traits = Traits(personal=3, discovery=2, variety=2)
    guide = {
        "for_you_if": {
            "en": "You want your usual creators first, and room for other videos on what you like.",
            "fr": "Vous voulez vos créateurs habituels d'abord, et de la place pour d'autres "
            "vidéos sur ce que vous aimez.",
        },
        "you_will_see": {
            "en": "A row of the creators you watch, then one row per subject, in the share you "
            "set.",
            "fr": "Une ligne des créateurs que vous regardez, puis une ligne par sujet, dans la "
            "part que vous réglez.",
        },
        "good_to_know": {
            "en": "It is the two other Signals side by side: each can be tried alone.",
            "fr": "Ce sont les deux autres Signals côte à côte : chacun peut être essayé seul.",
        },
    }
    specs = [
        Number(
            id="videos_share",
            group="rows",
            label={"en": "Share of the page for subjects", "fr": "Part de la page pour les sujets"},
            help={
                "en": "The rest goes to the creators you watch.",
                "fr": "Le reste va aux créateurs que vous regardez.",
            },
            min=0,
            max=1,
            step=0.1,
            default=0.4,
            unit="%",
            featured=True,
        ),
        page_size(),
        random_enabled(
            "Vary the page on every refresh", "Varier la page à chaque rafraîchissement"
        ),
    ]

    def __init__(
        self, channels: SignalsChannels | None = None, videos: SignalsVideos | None = None
    ):
        self._channels = channels or SignalsChannels()
        self._videos = videos or SignalsVideos()

    def _parts(self, values: Values) -> list[tuple[Algorithm, float]]:
        share = values.number("videos_share")
        return [(self._channels, 1 - share), (self._videos, share)]

    def profile(self, snapshot: Snapshot, values: Values) -> Profile:
        """What each part keeps of the person: the creators of one, the subjects of the other."""
        return Profile(
            groups=self._videos.profile(snapshot, Values(self._videos.specs, {})).groups,
            channels=self._channels.profile(snapshot, Values(self._channels.specs, {})).channels,
        )

    def plan(self, snapshot: Snapshot, values: Values) -> list[Source]:
        sources: list[Source] = []
        for part, share in self._parts(values):
            if share > 0:
                varies = {"random_enabled": values.flag("random_enabled")}
                sources += part.plan(snapshot, Values(part.specs, varies))
        return sources

    def rank(self, run: Run) -> list[Row]:
        size = run.values.whole("page_size")
        ranked: list[tuple[list[Row], float]] = []
        for part, share in self._parts(run.values):
            if share <= 0:
                continue
            # Each part sees only the candidates of the sources it uses, and may fill the page.
            groups = [g for g in run.groups if g.source.kind in part.sources]
            inner = Run(
                snapshot=run.snapshot,
                values=Values(
                    part.specs,
                    {"page_size": size, "random_enabled": run.values.flag("random_enabled")},
                ),
                groups=groups,
            )
            rows = part.rank(inner)
            run.dropped += inner.dropped
            run.scores += inner.scores
            run.row_reasons += inner.row_reasons
            ranked.append((rows, share))
        if not any(rows for rows, _ in ranked):
            return []
        # A part with nothing to show leaves its room to the other.
        live = [(rows, share) for rows, share in ranked if rows]
        total = sum(share for _, share in live)
        shown: set[str] = set()
        page: list[Row] = []
        for rows, share in live:
            room = round(size * share / total)
            for row in rows:
                videos = [v for v in row.videos if v.url not in shown][: max(0, room)]
                if not videos:
                    continue
                room -= len(videos)
                shown.update(v.url for v in videos)
                page.append(row.model_copy(update={"videos": videos}))
        return page
