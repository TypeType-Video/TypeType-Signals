"""A channel is one channel however its URL is written (imports and clients differ)."""

import pytest

from recommender import engine
from recommender.contract import HistoryItem, Overrides, canonical_channel_url
from recommender.signals_profile import channel_id

from .factories import snapshot, watched

UC = "UCabcdefghijklmnopqrstuv"
OTHER = "UCzyxwvutsrqponmlkjihgfe"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/channel/{UC}",
        f"https://www.youtube.com/channel/{UC}/",
        f"https://www.youtube.com/channel/{UC}/videos",
        f"http://youtube.com/channel/{UC}/featured",
        f"https://m.youtube.com/channel/{UC}/shorts?view=0",
        f" https://www.youtube.com/channel/{UC} ",
    ],
)
def test_every_spelling_of_a_channel_is_the_same_channel(url):
    assert canonical_channel_url(url) == f"https://www.youtube.com/channel/{UC}"
    assert channel_id(url) == f"channel:{UC}"


def test_handles_keep_their_name_and_other_urls_are_only_trimmed():
    assert (
        canonical_channel_url("https://youtube.com/@Some.Creator/videos")
        == "https://www.youtube.com/@Some.Creator"
    )
    assert canonical_channel_url(" https://example.org/x ") == "https://example.org/x"
    assert canonical_channel_url("") == ""


def test_channels_written_with_a_tab_keep_their_own_id():
    assert channel_id(f"https://www.youtube.com/channel/{UC}/videos") != channel_id(
        f"https://www.youtube.com/channel/{OTHER}/videos"
    )


def test_watches_of_one_channel_under_two_spellings_count_together():
    first = watched("a", UC, title="Camping alone in the snowy forest")
    second = watched("b", UC, title="Winter camping shelter in the forest")
    second = HistoryItem(
        **{**second.model_dump(), "channel_url": f"http://youtube.com/channel/{UC}/videos"}
    )
    profile = engine.profile(snapshot("signals", history=[first, second]))
    channels = [c for c in profile.channels if c.id == f"channel:{UC}"]
    assert len(channels) == 1
    assert set(channels[0].from_) == {"history:a", "history:b"}


def test_a_muted_channel_stays_muted_under_another_spelling():
    overrides = Overrides(muted_channels=[f"http://youtube.com/channel/{UC}/videos"])
    assert overrides.muted_channels == [f"https://www.youtube.com/channel/{UC}"]
