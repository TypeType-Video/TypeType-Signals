"""The fixed numbers of Signals: what is not a setting, and why each has its value."""

from ..signals_profile import MINUTE

MAX_CONTINUE = 4
RECENT_SHARE = 0.12
# Measured on five points: 0.2 brings median views from ~490K to ~630K (YouTube ~1.1M) and
# nudges measure 1 up, keeping renewal at 44 % (0.35: 42 %).
POPULARITY_BONUS = 0.2
CHANNEL_ROW_MIN = 3
# Measured: YouTube gives each interest 1 to 3.6 videos of a page; with no cap, Signals gave
# one interest 25.
MAX_PER_TOPIC = 3
CHANNEL_FIRST_SHARE = 0.6
# A latest watch past its boost still seeds related videos, weaker, fading over a day.
LATEST_SEED = 0.3
LATEST_HALF_LIFE_H = 24
BOOST_SEEDS = 3
GROUP_SEEDS = 6
# Measured on a real account: channel catalogues are the source YouTube agrees with most
# (77 % of their channels were on its home), so Signals asks for more of them.
CATALOGUE_CHANNELS = 8
SEARCH_TOPICS = 4
# Sources asked at every refresh; the others of each kind rotate, drawn from this many times
# more candidates, as YouTube rotates its occasional topics.
CORE_GROUPS = 3
CORE_TOPICS = 2
CORE_CHANNELS = 4
ROTATION_POOL = 3
FEED_LIMIT = 60
TRENDING_LIMIT = 30
# Below this closeness a title does not match an interest group at all.
TOPIC_FLOOR = 0.45
# A video related to a group's anchor joins the group's row only if it is at least
# this close to the group; otherwise it is simply "related to a video you watched".
PROVENANCE_FLOOR = 0.4
# Closeness a video needs to be placed in a topic's row (it scores from TOPIC_FLOOR).
ROW_FLOOR = 0.5
# Rows whose size is a share set by the user: the fill pass never grows them.
SHARED_ROWS = {"subscriptions", "discovery", "trending"}


def share_sized(key: str) -> bool:
    """Rows sized by a user setting: the shares, and the rows of recent watches."""
    return key in SHARED_ROWS or key.startswith("watched:")


# Videos asked by search for a subject the person added themselves.
CUSTOM_SEARCH = 10
# One interest can take at most this share of the new creators row.
DISCOVERY_BUCKET_SHARE = 0.5
# The new creators row is held back only by its own share per interest (above), not by the cap per
# interest the topic rows share; what it places still counts against that cap. With the shared cap
# it filled 11 of its 24 places on a page of a real account while 163 of 385 candidates were from
# channels never watched. On a real account against YouTube (7 points, 4 runs, both orders): never-
# watched share +12.3 points (≈23 → 36 %, YouTube 53 %), YouTube's channels on ours +0.2, ours on
# YouTube's home -1.9: of 558 never-watched candidate channels only 7 are on YouTube's home, so any
# channel added there costs a little of that.
DISCOVERY_OWN_CAP = True
# New creators whose channel comes back several times among the candidates go first in their
# row (False: by score alone). Among the new-channel candidates of five real profiles, a
# channel with one video in the pool was then really watched by the person 5.4 % of the time,
# with two 12.6 %, with three or four 14.2 %, with five or more 37.2 %. On the page bench
# (30 cuts): real discoveries on the page +0.67 points [+0.14, +1.28], known channels then
# watched unchanged.
DISCOVERY_ECHO_FIRST = True
# A new creator's video needs this many views for the new creators row (unknown: allowed). Among the
# new-channel candidates of five real profiles, a video under 10,000 views was from a channel the
# person then really watched 2.3 % of the time, against 6 to 10 % above. On the page bench (30
# cuts): real discoveries on the page +0.21 points [+0.04, +0.39], known channels unchanged; on a
# real account against YouTube no page changed. 100,000 had been measured worse on that account.
DISCOVERY_MIN_VIEWS = 10_000
# A candidate this close to a muted group is dropped.
MUTED_SIMILARITY = 0.55
# New candidate titles embedded at most per page; the others in the background, ready for the next
# page, and meanwhile matched to no interest (their channel and source still count). A page brings
# ~350 candidates and the fp16 model takes ~14 ms a title on the reference machine (350 in 4.5-5.5
# s, whatever the threads), so embedding them all took rank to 3 s on a first page.
RANK_EMBED_BUDGET = 40
# The share watched that makes a watch's title worth embedding ahead (warm): long_term_min_watched.
WARM_MIN_WATCHED = 0.3
CONCLUSIONS_KEPT = 4
FRESHNESS_HALF_LIFE_DAYS = 14
# Chosen on a real account: with the rotating sources, 0.35 gives about as many new videos per
# refresh as YouTube.
DRAW_WEIGHT = 0.35
PINNED_MIN = 4
# A topic row needs at least this many videos to be shown as its own row.
MIN_ROW = 3
DAY_MS = 24 * 60 * MINUTE

TITLES = {
    "continue": {"en": "Continue watching", "fr": "Reprendre"},
    "watch_again": {"en": "Watch it again", "fr": "Revoir"},
    "watched": {"en": "Because you watched {title}", "fr": "Parce que vous avez regardé {title}"},
    "subscriptions": {"en": "New from your subscriptions", "fr": "Nouveau dans vos abonnements"},
    "same_channel": {"en": "From creators you watch", "fr": "De créateurs que vous regardez"},
    "discovery": {"en": "New creators for you", "fr": "Nouveaux créateurs pour vous"},
    "trending": {"en": "Trending", "fr": "Tendances"},
    "for_you": {"en": "Recommended for you", "fr": "Recommandé pour vous"},
    "channel": {"en": "More from {name}", "fr": "Plus de {name}"},
}
WHY = {
    "continue": {
        "en": "{count} videos started in the last 14 days and not finished",
        "fr": "{count} vidéos commencées ces 14 derniers jours et non terminées",
    },
    "watch_again": {
        "en": "{count} videos you came back to on several days",
        "fr": "{count} vidéos que vous avez regardées plusieurs jours différents",
    },
    "watched": {
        "en": "Watched {percent} % {ago} min ago: its effect is at {strength} % and stops in "
        "{left} min",
        "fr": "Regardée à {percent} % il y a {ago} min : son effet est à {strength} % et "
        "s'arrête dans {left} min",
    },
    "subscriptions": {
        "en": "{count} videos from your subscriptions, best scores first",
        "fr": "{count} vidéos de vos abonnements, meilleurs scores d'abord",
    },
    "topic": {
        "en": "Interest group deduced from {count} videos you watched through{pinned}",
        "fr": "Centre d'intérêt déduit de {count} vidéos regardées jusqu'au bout{pinned}",
    },
    "pinned": {"en": ", pinned by you", "fr": ", épinglé par vous"},
    "custom": {
        "en": "A subject you added yourself",
        "fr": "Un sujet que vous avez ajouté vous-même",
    },
    "same_channel": {
        "en": "Other videos from the channels you watch most",
        "fr": "D'autres vidéos des chaînes que vous regardez le plus",
    },
    "discovery": {
        "en": "Channels you have never watched nor followed, ranked by how close they are to "
        "what you watch",
        "fr": "Chaînes jamais regardées ni suivies, classées selon leur proximité avec ce que "
        "vous regardez",
    },
    "trending": {
        "en": "Popular right now in your country",
        "fr": "Populaires en ce moment dans votre pays",
    },
    "channel": {
        "en": "A channel you watch: {count} of your videos or subscriptions come from {name}",
        "fr": "Une chaîne que vous regardez : {count} de vos vidéos ou abonnements viennent de "
        "{name}",
    },
    "for_you": {
        "en": "The best remaining scores, whatever the source",
        "fr": "Les meilleurs scores restants, toutes sources confondues",
    },
}
