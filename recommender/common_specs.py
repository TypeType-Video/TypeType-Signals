"""Parameters several algorithms share. Each algorithm picks its own default and bounds."""

from .parameters import Boolean, Number


def max_per_channel(default: int, maximum: int) -> Number:
    return Number(
        id="max_per_channel",
        group="rows",
        label={"en": "Max videos per channel", "fr": "Vidéos max par chaîne"},
        help={
            "en": "Caps how many videos from one channel appear on a page, so a very active "
            "channel cannot take all the room.",
            "fr": "Limite le nombre de vidéos d'une même chaîne sur une page, pour qu'une "
            "chaîne très active ne prenne pas toute la place.",
        },
        min=1,
        max=maximum,
        step=1,
        default=default,
    )


def include_shorts(default: bool, featured: bool = False) -> Boolean:
    return Boolean(
        id="include_shorts",
        group="sources",
        label={"en": "Include Shorts", "fr": "Inclure les Shorts"},
        help={
            "en": "Adds Shorts from the channels you follow to the home page. Off, only "
            "regular videos appear.",
            "fr": "Ajoute les Shorts des chaînes suivies à l'accueil. Désactivé, seules les "
            "vidéos classiques apparaissent.",
        },
        default=default,
        featured=featured,
    )


def trending_share(default: float, maximum: float, featured: bool = False) -> Number:
    return Number(
        id="trending_share",
        group="exploration",
        label={"en": "Share of trending videos", "fr": "Part de tendances"},
        help={
            "en": "Share of the home page kept for videos popular right now in your country, "
            "outside your habits.",
            "fr": "Part de l'accueil réservée aux vidéos populaires en ce moment dans votre "
            "pays, en dehors de vos habitudes.",
        },
        min=0,
        max=maximum,
        step=0.05,
        default=default,
        unit="%",
        featured=featured,
    )


def random_enabled(label_en: str, label_fr: str) -> Boolean:
    return Boolean(
        id="random_enabled",
        group="randomness",
        label={"en": label_en, "fr": label_fr},
        help={
            "en": "On, the home page draws a new selection from the candidates on every "
            "refresh. Off, it stays the same until your data changes.",
            "fr": "Activé, l'accueil tire une nouvelle sélection parmi les candidates à chaque "
            "rafraîchissement. Désactivé, il reste identique tant que vos données ne changent "
            "pas.",
        },
        default=True,
        featured=True,
    )


def related_per_seed(default: int, maximum: int, label_en: str, label_fr: str) -> Number:
    return Number(
        id="related_per_seed",
        group="sources",
        label={"en": label_en, "fr": label_fr},
        help={
            "en": "For each starting video, how many close videos (same topic, same audience) "
            "are fetched as candidates.",
            "fr": "Pour chaque vidéo de départ, combien de vidéos proches (même sujet, même "
            "public) sont récupérées comme candidates.",
        },
        min=0,
        max=maximum,
        step=1,
        default=default,
    )


def page_size(default: int = 80) -> Number:
    return Number(
        id="page_size",
        group="rows",
        label={"en": "Videos per home page", "fr": "Vidéos par page d'accueil"},
        help={
            "en": "Number of videos on a home page before scrolling loads more.",
            "fr": "Nombre de vidéos sur une page d'accueil avant que le défilement en charge "
            "d'autres.",
        },
        min=20,
        max=150,
        step=10,
        default=default,
    )
