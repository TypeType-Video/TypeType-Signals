"""The rows a page of Signals can have, each with its share of the page."""

from dataclasses import dataclass, field

from ..contract_rank import Because, Reason
from ..signals_clusters import CUSTOM_PREFIX
from ..signals_profile import MINUTE
from .signals_constants import (
    MIN_ROW,
    PINNED_MIN,
    TITLES,
    WHY,
)
from .signals_scoring import Scoring


@dataclass
class Slot:
    """A row being filled: its share of the page and the videos it got."""

    id: str
    title: str
    reason: Reason
    why: str
    quota: int = 0
    minimum: int = 1
    extra: Because | None = None
    urls: list[str] = field(default_factory=list)


# An explorer gets more room for new creators, in proportion to how much they explore beyond
# EXPLORATION_REFERENCE; someone who explores less keeps the share they set.
EXPLORATION_REFERENCE = 0.25
MAX_DISCOVERY_SHARE = 0.6


def discovery_share(setting: float, exploration: float | None) -> float:
    if not exploration or exploration <= EXPLORATION_REFERENCE or not EXPLORATION_REFERENCE:
        return setting
    return min(MAX_DISCOVERY_SHARE, setting * exploration / EXPLORATION_REFERENCE)


class Rows:
    """The rows the page can have, opened from the scored candidates."""

    def __init__(self, scored: Scoring):
        self.scored = scored

    def opened(self) -> "Rows":
        """The same object once every row has its share of the page."""
        self._open_rows()
        return self

    def _open_rows(self) -> None:
        """The rows the page can have, each with its share of the page. Their order here is the
        order they are filled in."""
        self.size = self.scored.values.whole("page_size")
        self.subscription_of = {s.channel_url: s.id for s in self.scored.snapshot.subscriptions}
        self.slots: dict[str, Slot] = {}
        self._resume_rows()
        self._recent_rows()
        if self.scored.snapshot.subscriptions:
            self.slots["subscriptions"] = Slot(
                "subscriptions",
                self.scored.run.text(TITLES["subscriptions"]),
                Reason(kind="new_from_subscriptions"),
                "",
                quota=round(self.size * self.scored.values.number("subscriptions_share")),
            )
        share = discovery_share(
            self.scored.values.number("discovery_share"), self.scored.conclusions.exploration
        )
        self.slots["discovery"] = self._row(
            "discovery",
            "discovery",
            quota=round(self.size * share),
            extra=Because(kind="discovery", weight=1.0),
        )
        self._topic_rows()
        self.slots["same_channel"] = self._row(
            "same_channel", "same_channel", row_id="same-channel"
        )
        self.slots["for_you"] = self._row("for_you", "recommended", row_id="for-you")
        self.slots["trending"] = self._row(
            "trending",
            "trending",
            quota=round(self.size * self.scored.values.number("trending_share")),
            extra=Because(kind="trending", weight=1.0),
        )

    def _row(self, key: str, kind: str, row_id: str | None = None, **fields) -> Slot:
        """A row whose title and explanation are fixed texts."""
        reason = Reason(kind=kind)
        return Slot(
            row_id or key,
            self.scored.run.text(TITLES[key]),
            reason,
            self.scored.run.text(WHY[key]),
            **fields,
        )

    def _resume_rows(self) -> None:
        """Videos started and not finished, and videos the person watches again."""
        for key, urls in (
            ("continue", self.scored.continue_urls),
            ("watch_again", self.scored.watch_again_urls),
        ):
            if urls:
                self.slots[key] = Slot(
                    key,
                    self.scored.run.text(TITLES[key]),
                    Reason(kind=key),
                    self.scored.run.text(WHY[key], count=len(urls)),
                    urls=list(urls),
                )

    def _recent_rows(self) -> None:
        """One row per video just watched, bigger while its effect is strong."""
        max_age = self.scored.values.number("session_max_age")
        strength = sum(b.strength for b in self.scored.conclusions.boosts)
        # The room recent watches get follows their remaining strength, so it shrinks as
        # their effect fades instead of staying full until it suddenly stops.
        recent_total = round(
            self.size * self.scored.values.number("recent_share") * min(1.0, strength)
        )
        self.boost_row: dict[str, str] = {}
        for boost in self.scored.conclusions.boosts:
            item = boost.item
            key = f"watched:{item.id}"
            ago = round((self.scored.now - item.watched_at) / MINUTE)
            self.slots[key] = Slot(
                f"watched-{item.id.split(':', 1)[-1]}",
                self.scored.run.text(TITLES["watched"], title=item.title or item.video_url),
                Reason(kind="watched", ref=item.id),
                self.scored.run.text(
                    WHY["watched"],
                    percent=round(100 * (item.watched_share or 0)),
                    ago=ago,
                    strength=round(100 * boost.strength),
                    left=max(0, round(max_age - ago)),
                ),
                quota=round(recent_total * boost.strength / strength) if strength else 0,
            )
            # Its related videos, and the catalogue of its channel when watched through.
            self.boost_row[item.id] = key
            channel = self.scored.conclusions.channel(item.channel_url)
            if channel and (item.watched_share or 0) >= self.scored.lasting:
                self.boost_row.setdefault(channel.id, key)

    def _topic_rows(self) -> None:
        """One row per interest that counts, pinned ones first: they share what the other
        rows leave of the page, by their weight."""
        self.active = sorted(
            (
                g
                for g in self.scored.conclusions.groups
                if not g.muted and not g.ignored and g.weight > 0 and (g.named or g.pinned)
            ),
            key=lambda g: (not g.pinned, -g.weight_now),
        )
        topics_total = (
            self.size
            - sum(slot.quota for slot in self.slots.values())
            - len(self.scored.continue_urls)
            - len(self.scored.watch_again_urls)
        )
        topics_total -= round(self.size * self.scored.values.number("trending_share"))
        total_weight = sum(g.weight_now for g in self.active) or 1
        for group in self.active:
            quota = round(max(0, topics_total) * group.weight_now / total_weight)
            why = self.scored.run.text(WHY["custom"])
            if not group.id.startswith(CUSTOM_PREFIX):
                pinned = self.scored.run.text(WHY["pinned"]) if group.pinned else ""
                why = self.scored.run.text(WHY["topic"], count=len(group.members), pinned=pinned)
            self.slots[group.id] = Slot(
                group.id.replace(":", "-"),
                group.label,
                Reason(kind="topic", ref=group.id),
                why,
                quota=max(quota, PINNED_MIN) if group.pinned else quota,
                minimum=1 if group.pinned else MIN_ROW,
            )
