"""How Signals turns scored candidates into a page: each placed in a row, then the rows built."""

import math
from collections import Counter
from itertools import zip_longest
from typing import TYPE_CHECKING

from ..base import (
    Run,
    cap_per_channel,
    make_row,
)
from ..contract_rank import Because, Reason, Row
from .signals_constants import (
    CHANNEL_FIRST_SHARE,
    CHANNEL_ROW_MIN,
    DISCOVERY_BUCKET_SHARE,
    DISCOVERY_ECHO_FIRST,
    DISCOVERY_MIN_VIEWS,
    DISCOVERY_OWN_CAP,
    TITLES,
    WHY,
    share_sized,
)
from .signals_rows import Rows
from .signals_scoring import Scoring

if TYPE_CHECKING:
    from .signals import Signals


class Ranking:
    """One page being ranked: the scored candidates shared out between the rows, then built."""

    def __init__(self, scored: Scoring, layout: Rows):
        self.scored = scored
        self.layout = layout

    def rows(self) -> list[Row]:
        self._fill()
        return self._build()

    def _fill(self) -> None:
        """Puts each candidate in a row it belongs to, best scores first, under the caps."""
        self.by_score = sorted(self.scored.ranked, key=lambda u: -self.scored.totals[u])
        # After a full watch YouTube's reaction is mostly the channel itself (The B1M: 4–5 of
        # its videos out of about 6): a recent watch's row takes its channel first, up to
        # CHANNEL_FIRST_SHARE of its room, then related videos.
        watched_channel = {
            self.layout.boost_row[b.item.id]: b.item.channel_url
            for b in self.scored.conclusions.boosts
            if (b.item.watched_share or 0) >= self.scored.lasting
        }
        mixed = self._new_creators_in_turn()
        self.placed: set[str] = set(self.scored.continue_urls) | set(self.scored.watch_again_urls)
        self.cap = self.scored.values.whole("max_per_channel")
        self.topic_cap = self.scored.values.whole("max_per_topic")
        self.per_channel = Counter(
            self.scored.pool.videos[u].uploader_url for u in self.scored.continue_urls
        )
        self.per_topic: Counter[str] = Counter()

        # First pass: each row takes its share, best scores first.
        for key, slot in self.layout.slots.items():
            order = mixed if key == "discovery" else self.by_score
            if key in watched_channel:
                order = self._channel_first(watched_channel[key], slot.quota)
            for url in order:
                if len(slot.urls) >= slot.quota:
                    break
                if self._free(url, key) and key in self._homes(url):
                    self._place(url, key)
        # Second pass: rows that could not fill their share leave room; the best remaining
        # candidates fill it, each in the first row it belongs to. Rows sized by a share the
        # user sets (subscriptions, new creators, trending) never grow past it: a share of
        # 0 % means none.
        for url in self.by_score:
            if len(self.placed) >= self.layout.size:
                break
            home = next((key for key in self._homes(url) if not share_sized(key)), None)
            if home and self._free(url, home):
                self._place(url, home)
        # A topic row of one or two videos reads as noise: its videos join a broader row.
        for key in [k for k in self.layout.slots if k.startswith("group:")]:
            slot = self.layout.slots[key]
            if 0 < len(slot.urls) < slot.minimum:
                for url in slot.urls:
                    self.layout.slots[self._homes(url)[-1]].urls.append(url)
                slot.urls = []
        if "subscriptions" in self.layout.slots:
            count = len(self.layout.slots["subscriptions"].urls)
            self.layout.slots["subscriptions"].why = self.scored.run.text(
                WHY["subscriptions"], count=count
            )

    def _arrived_from(self, url: str, kind: str) -> bool:
        return any(s.kind == kind for s in self.scored.pool.sources[url])

    def _is_new_creator(self, url: str) -> bool:
        return (
            self.scored.pool.videos[url].uploader_url not in self.scored.conclusions.known_channels
        )

    def _homes(self, url: str) -> list[str]:
        """The rows a candidate belongs to, from how it arrived and what it is close to."""
        # A video brought by a recent watch belongs to that watch's row only, so the room
        # set for recent watches is all the room they take.
        if self.scored.seed_of.get(url) in self.layout.boost_row:
            return [self.layout.boost_row[self.scored.seed_of[url]]]
        found = []
        if self._arrived_from(url, "subscription_feed") and "subscriptions" in self.layout.slots:
            found.append("subscriptions")
        if url in self.scored.group_of and self.scored.group_of[url].id in self.layout.slots:
            found.append(self.scored.group_of[url].id)
        # "Creators you never watched, close to your interests" (the setting's own words): a
        # new creator close to none of them (a film trailer, a music video) is not one.
        # Someone with no interest at all (few likes, an imported history) would get none:
        # for them alone the row takes any new creator the sources bring.
        if (
            self._is_new_creator(url)
            and (url in self.scored.group_of or not self.scored.conclusions.groups)
            and (self.scored.pool.videos[url].view_count or DISCOVERY_MIN_VIEWS)
            >= DISCOVERY_MIN_VIEWS
            and not all(s.kind == "trending" for s in self.scored.pool.sources[url])
        ):
            found.append("discovery")
        if self._arrived_from(url, "trending"):
            found.append("trending")
        known = self.scored.conclusions.channel(self.scored.pool.videos[url].uploader_url)
        found.append("same_channel" if known else "for_you")
        return found

    def _channel_first(self, channel_url: str, quota: int) -> list[str]:
        own = [u for u in self.by_score if self.scored.pool.videos[u].uploader_url == channel_url]
        own = own[: math.ceil(quota * CHANNEL_FIRST_SHARE)]
        return own + [u for u in self.by_score if u not in set(own)]

    def _new_creators_in_turn(self) -> list[str]:
        """The candidates taken in turn from each interest, so one topic cannot fill the new
        creators row."""
        buckets: dict[str, list[str]] = {}
        for url in self.by_score:
            interest = self.scored.group_of[url].id if url in self.scored.group_of else ""
            buckets.setdefault(interest, []).append(url)
        if DISCOVERY_ECHO_FIRST:
            echo = Counter(
                self.scored.pool.videos[u].uploader_url
                for u in self.by_score
                if self._is_new_creator(u)
            )
            for bucket in buckets.values():
                # A stable sort: within the same echo, score still decides.
                bucket.sort(key=lambda u: -echo[self.scored.pool.videos[u].uploader_url])
        most = max(1, math.ceil(self.layout.slots["discovery"].quota * DISCOVERY_BUCKET_SHARE))
        turns = zip_longest(*(b[:most] for b in buckets.values()))
        return [u for turn in turns for u in turn if u]

    def _place(self, url: str, key: str) -> None:
        self.layout.slots[key].urls.append(url)
        self.placed.add(url)
        self.per_channel[self.scored.pool.videos[url].uploader_url] += 1
        if url in self.scored.topic_of and not key.startswith("watched:"):
            topic = self.scored.topic_of[url]
            self.per_topic[self.scored.family.get(topic.id, topic.id)] += 1

    def _free(self, url: str, key: str = "") -> bool:
        """True when the caps per channel and per interest still let the candidate in."""
        if (
            url in self.placed
            or self.per_channel[self.scored.pool.videos[url].uploader_url] >= self.cap
        ):
            return False
        # A recent watch's row and pinned interests are not held back by the cap. The new
        # creators row is: measured, its new channels cluster in the biggest interests.
        topic = self.scored.topic_of.get(url)
        if topic is None or topic.pinned or key.startswith("watched:"):
            return True
        if key == "discovery" and DISCOVERY_OWN_CAP:
            return True
        # The cap follows the time of day: an interest usually watched at this hour gets
        # more room, one never watched at this hour less (at least one video).
        allowed = max(1, round(self.topic_cap * topic.hour_factor))
        return self.per_topic[self.scored.family.get(topic.id, topic.id)] < allowed

    def _build(self) -> list[Row]:
        """The rows in their order, each video with the reasons it is there."""
        order = self._order()
        # Within a row, best scores first; videos to resume keep their watch order.
        for key, slot in self.layout.slots.items():
            if key not in ("continue", "watch_again"):
                slot.urls.sort(key=lambda u: -self.scored.totals.get(u, 0))
        page = cap_per_channel(
            self.scored.run,
            [u for key in order for u in self.layout.slots[key].urls],
            self.scored.pool,
            self.scored.values.whole("max_per_channel"),
        )
        allowed = set(page)
        rows: list[Row] = []
        for key in order:
            slot = self.layout.slots[key]
            videos = [(url, self._reasons(key, url)) for url in slot.urls if url in allowed]
            if key == "same_channel":
                videos = self._channel_rows(videos, rows)
            if not videos:
                continue
            rows.append(make_row(slot.id, slot.title, slot.reason, videos))
            self.scored.run.explain(slot.id, slot.why)
        return rows

    def _order(self) -> list[str]:
        """The rows from top to bottom: the first interest, then new creators, then the rest."""
        topic_keys = [g.id for g in self.layout.active if self.layout.slots[g.id].urls]
        order = [
            "continue",
            *[self.layout.boost_row[b.item.id] for b in self.scored.conclusions.boosts],
            "subscriptions",
        ]
        order += topic_keys[:1] + ["discovery"] + topic_keys[1:]
        order += ["watch_again", "same_channel", "for_you", "trending"]
        return [key for key in dict.fromkeys(order) if key in self.layout.slots]

    def _reasons(self, key: str, url: str) -> list[Because]:
        """Why this video is in this row, strongest first."""
        slot = self.layout.slots[key]
        if key == "continue":
            item = next(i for i in self.scored.conclusions.started if i.video_url == url)
            reasons = [Because(kind="continue", ref=item.id, weight=1.0)]
        elif key == "watch_again":
            item = next(i for i in self.scored.conclusions.rewatched if i.video_url == url)
            reasons = [Because(kind="watch_again", ref=item.id, weight=1.0)]
        else:
            reasons = list(self.scored.because.get(url, []))
        if key == "subscriptions":
            channel_url = self.scored.pool.videos[url].uploader_url
            subscription = self.layout.subscription_of.get(channel_url)
            reasons.insert(0, Because(kind="subscriptions", ref=subscription, weight=1.0))
        if slot.extra:
            reasons.insert(0, slot.extra)
        if not reasons:
            weight = round(self.scored.totals.get(url, 0), 4)
            reasons = [Because(kind="recommended", weight=weight)]
        return reasons

    def _channel_rows(self, videos: list[tuple], rows: list[Row]) -> list[tuple]:
        """YouTube gives a channel it knows you like a row of its own (its "Cars and more" was
        five videos of one channel): a channel with CHANNEL_ROW_MIN videos here gets "More from
        <channel>", added to the rows; the videos that keep the shared row are returned."""
        by_channel: dict[str, list] = {}
        for url, reasons in videos:
            channel_url = self.scored.pool.videos[url].uploader_url
            by_channel.setdefault(channel_url, []).append((url, reasons))
        shared = []
        for channel_url, own in by_channel.items():
            channel = self.scored.conclusions.channel(channel_url)
            if len(own) < CHANNEL_ROW_MIN or channel is None:
                shared += own
                continue
            row_id = f"channel-{channel.id.split(':', 1)[-1]}"
            name = channel.name or self.scored.pool.videos[own[0][0]].uploader_name
            title = self.scored.run.text(TITLES["channel"], name=name)
            rows.append(make_row(row_id, title, Reason(kind="same_channel", ref=channel.id), own))
            self.scored.run.explain(row_id, WHY["channel"], name=name, count=len(channel.refs))
        return shared


def rank_page(algorithm: "Signals", run: Run) -> list[Row]:
    """The page of one run: candidates scored, rows opened, then each candidate placed."""
    scored = Scoring(algorithm, run).scored()
    return Ranking(scored, Rows(scored).opened()).rows()
