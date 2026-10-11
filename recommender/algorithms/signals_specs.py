"""The settings of Signals, as the settings screen shows them."""

from ..common_specs import (
    max_per_channel,
    page_size,
    random_enabled,
    related_per_seed,
    trending_share,
)
from ..parameters import Boolean, Number
from .signals_constants import DRAW_WEIGHT, MAX_PER_TOPIC, POPULARITY_BONUS, RECENT_SHARE

WEIGHT_HELP = {
    "w_watch_seed": (
        "Weight of a watched video",
        "Poids d'une vidéo regardée",
        "How much being close to a video you watched counts in the ranking. Higher, the home "
        "page looks more like what you just watched.",
        "Importance, dans le classement, d'être proche d'une vidéo que vous avez regardée. Plus "
        "la valeur est haute, plus l'accueil ressemble à ce que vous venez de voir.",
        0.6,
    ),
    "w_same_channel": (
        "Weight of the same creator",
        "Poids du même créateur",
        "How much being published by a creator you already watch counts in the ranking.",
        "Importance, dans le classement, d'être publiée par un créateur que vous regardez déjà.",
        0.25,
    ),
    "w_topic": (
        "Weight of a matching topic",
        "Poids d'un sujet proche",
        "How much matching one of your interests counts in the ranking.",
        "Importance, dans le classement, de correspondre à l'un de vos centres d'intérêt.",
        0.2,
    ),
    "w_freshness": (
        "Weight of freshness",
        "Poids de la fraîcheur",
        "How much being recent counts in the ranking. Higher, newly published videos come "
        "first.",
        "Importance, dans le classement, d'être récente. Plus la valeur est haute, plus les "
        "vidéos publiées depuis peu remontent.",
        # YouTube's home leans on proven videos (median 90 days old on a real account); a
        # weight of 0.1 made ours 14 days old, 0.05 keeps a little freshness.
        0.05,
    ),
}


def _weights() -> list[Number]:
    return [
        Number(id, "weights", {"en": en, "fr": fr}, {"en": hen, "fr": hfr}, 0, 1, 0.01, default)
        for id, (en, fr, hen, hfr, default) in WEIGHT_HELP.items()
    ]


SPECS = [
    Number(
        "seed_min_watched",
        "sources",
        {
            "en": "Watched share for a video to count",
            "fr": "Part regardée pour qu'une vidéo compte",
        },
        {
            "en": "A video only becomes a starting point for recommendations if you watched at "
            "least this share of it. Below that, it counts as skimmed.",
            "fr": "Une vidéo ne sert de point de départ aux recommandations que si vous en avez "
            "regardé au moins cette part. En dessous, elle est considérée comme survolée.",
        },
        0,
        1,
        0.05,
        # 0 since 2026-09-30: people go back to channels whose videos they did not finish.
        # Over six cut dates of one person's 26 months,
        # 0 instead of 0.3 put more of the next two weeks' channels in Signals' top 10/25.
        0.0,
        "%",
    ),
    related_per_seed(
        default=20,
        maximum=40,
        label_en="Related videos per watched video",
        label_fr="Vidéos liées par vidéo regardée",
    ),
    Number(
        "catalogue_per_channel",
        "sources",
        {
            "en": "Catalogue videos per watched creator",
            "fr": "Vidéos du catalogue par créateur regardé",
        },
        {
            "en": "When you watch a creator, how many of their other videos are offered as "
            "candidates.",
            "fr": "Quand vous regardez un créateur, combien de ses autres vidéos sont proposées "
            "comme candidates.",
        },
        0,
        30,
        1,
        10,
    ),
    Number(
        "search_per_topic",
        "sources",
        {"en": "Search results per interest", "fr": "Résultats de recherche par centre d'intérêt"},
        {
            "en": "For each of your main interests, how many videos found by searching its words "
            "are offered as candidates. They bring popular videos of the topic from channels you "
            "do not know yet, like the topic rows of YouTube. 0 turns the search off.",
            "fr": "Pour chacun de vos principaux centres d'intérêt, combien de vidéos trouvées en "
            "cherchant ses mots sont proposées comme candidates. Elles apportent des vidéos "
            "populaires du sujet, venant de chaînes que vous ne connaissez pas encore, comme les "
            "lignes par sujet de YouTube. 0 désactive la recherche.",
        },
        0,
        40,
        1,
        15,
    ),
    Boolean(
        "use_shorts_history",
        "sources",
        {"en": "Let Shorts shape the home page", "fr": "Laisser les Shorts influencer l'accueil"},
        {
            "en": "Your Shorts are tracked separately. On, they can also steer the regular videos "
            "of the home page. Off, they have no effect on them.",
            "fr": "Vos Shorts sont suivis à part. Activé, ils peuvent aussi orienter les vidéos "
            "classiques de l'accueil. Désactivé, ils n'ont aucun effet dessus.",
        },
        False,
        featured=True,
    ),
    *_weights(),
    Number(
        "popularity_penalty",
        "weights",
        {"en": "Penalty for very popular videos", "fr": "Pénalité pour les vidéos très populaires"},
        {
            "en": "Pushes down videos already seen by huge audiences, to leave room for smaller "
            "creators. Off by default: YouTube's home leans on proven, widely watched videos, "
            "and this follows it; raise it to favour small creators.",
            "fr": "Fait descendre les vidéos déjà vues par énormément de monde, pour laisser de la "
            "place aux créateurs plus petits. Désactivée par défaut : l'accueil de YouTube "
            "s'appuie sur des vidéos éprouvées et très vues, et celui-ci fait de même ; "
            "augmentez-la pour favoriser les petits créateurs.",
        },
        0,
        0.5,
        0.01,
        0,
    ),
    Number(
        "popularity_bonus",
        "weights",
        {"en": "Weight of proven popularity", "fr": "Poids de la popularité éprouvée"},
        {
            "en": "Pushes up videos many people already watched. YouTube's home leans on them "
            "(median a month old and a million views on a measured account).",
            "fr": "Fait remonter les vidéos déjà vues par beaucoup de monde. L'accueil de YouTube "
            "s'appuie sur elles (un mois et un million de vues en médiane sur un compte "
            "mesuré).",
        },
        0,
        0.5,
        0.01,
        POPULARITY_BONUS,
    ),
    Boolean(
        "session_boost",
        "decay",
        {
            "en": "Take recently watched videos into account",
            "fr": "Prendre en compte les vidéos regardées récemment",
        },
        {
            "en": "A video you just watched steers the home page for a few hours. Off, only your "
            "long-term habits count.",
            "fr": "Une vidéo que vous venez de regarder oriente l'accueil pendant quelques heures. "
            "Désactivé, seules vos habitudes de long terme comptent.",
        },
        True,
        featured=True,
    ),
    Number(
        "recent_share",
        "decay",
        {"en": "Room for what you just watched", "fr": "Place des vidéos regardées récemment"},
        {
            "en": "Largest share of the page given to videos related to what you just watched. "
            "It shrinks as the watch's effect fades.",
            "fr": "La plus grande part de la page donnée aux vidéos liées à ce que vous venez de "
            "regarder. Elle diminue à mesure que l'effet du visionnage s'estompe.",
        },
        0,
        0.5,
        0.01,
        RECENT_SHARE,
        "%",
    ),
    Number(
        "session_half_life",
        "decay",
        {"en": "Half-life of a recent watch", "fr": "Demi-vie d'un visionnage récent"},
        {
            "en": "Time after which the effect of a recent video is halved. Shorter, the home page "
            "moves on quickly.",
            "fr": "Temps au bout duquel l'effet d'une vidéo récente est divisé par deux. Plus il "
            "est court, plus l'accueil passe vite à autre chose.",
        },
        5,
        240,
        5,
        45,
        "min",
    ),
    Number(
        "session_max_age",
        "decay",
        {
            "en": "A recent watch stops counting after",
            "fr": "Un visionnage récent ne compte plus après",
        },
        {
            "en": "Past this delay, a watched video has no immediate effect on the home page "
            "anymore.",
            "fr": "Au-delà de ce délai, une vidéo regardée n'a plus d'effet immédiat sur "
            "l'accueil.",
        },
        30,
        720,
        30,
        180,
        "min",
    ),
    Number(
        "long_term_min_watched",
        "decay",
        {
            "en": "Watched share for a lasting interest",
            "fr": "Part regardée pour un intérêt durable",
        },
        {
            "en": "Share you need to watch for a video to count in your lasting interests, the "
            "ones kept from one day to the next.",
            "fr": "Part à regarder pour qu'une vidéo compte dans vos centres d'intérêt durables, "
            "ceux qui restent d'un jour à l'autre.",
        },
        0.3,
        1,
        0.05,
        0.8,
        "%",
    ),
    max_per_channel(default=6, maximum=12),
    Number(
        "max_per_topic",
        "rows",
        {"en": "Max videos per interest", "fr": "Vidéos max par centre d'intérêt"},
        {
            "en": "Caps how many videos about one of your interests appear on a page, so one "
            "big interest cannot take the whole page. Videos of what you just watched and "
            "pinned interests are not capped.",
            "fr": "Limite le nombre de vidéos d'un même centre d'intérêt sur une page, pour qu'un "
            "gros centre d'intérêt ne prenne pas toute la page. Les vidéos liées à ce que vous "
            "venez de regarder et les centres d'intérêt épinglés ne sont pas limités.",
        },
        1,
        80,
        1,
        MAX_PER_TOPIC,
    ),
    page_size(),
    Number(
        "subscriptions_share",
        "rows",
        {"en": "Share of subscriptions", "fr": "Part d'abonnements"},
        {
            "en": "Share of the home page kept for new videos from the channels you follow.",
            "fr": "Part de l'accueil réservée aux nouvelles vidéos des chaînes suivies.",
        },
        0,
        0.6,
        0.05,
        0.1,
        "%",
    ),
    Number(
        "discovery_share",
        "exploration",
        {"en": "Share of creators you never watched", "fr": "Part de créateurs jamais vus"},
        {
            "en": "Share of the home page kept for creators you never watched, close to your "
            "interests.",
            "fr": "Part de l'accueil réservée à des créateurs jamais regardés, proches de vos "
            "centres d'intérêt.",
        },
        0,
        0.6,
        0.05,
        0.3,
        "%",
        featured=True,
    ),
    # YouTube's home shows no trending row: 0 of 175 shelf titles over 1,177 home pages of
    # the measured account. Off by default, still a choice.
    trending_share(default=0.0, maximum=0.3),
    Boolean(
        "time_of_day",
        "weights",
        {"en": "Follow the time of day", "fr": "Suivre l'heure de la journée"},
        {
            "en": "An interest you usually watch at this hour of the day gets more room, one you "
            "never watch at this hour less. Only watches whose time is known count.",
            "fr": "Un centre d'intérêt que vous regardez d'habitude à cette heure-ci prend plus de "
            "place, un que vous ne regardez jamais à cette heure-ci en prend moins. Seuls les "
            "visionnages dont l'heure est connue comptent.",
        },
        True,
    ),
    random_enabled("Vary the page on every refresh", "Varier la page à chaque rafraîchissement"),
    Number(
        "draw_weight",
        "randomness",
        {"en": "Amount of variation", "fr": "Ampleur du hasard"},
        {
            "en": "How much chance weighs against the score when the page varies. Higher, more "
            "videos change from one refresh to the next; 0 keeps the best scores in place.",
            "fr": "Le poids du hasard face au score quand la page varie. Plus il est haut, plus de "
            "vidéos changent d'un rafraîchissement à l'autre ; 0 garde les meilleurs scores en "
            "place.",
        },
        0,
        0.6,
        0.05,
        DRAW_WEIGHT,
    ),
]
