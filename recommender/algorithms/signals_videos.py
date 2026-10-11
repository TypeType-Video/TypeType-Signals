"""Signals, videos: aims at subjects, not at channels.

What you watch is read by subject (recommender/subjects.py: tags and category of each video
when the catalogue has them, its title otherwise). For your main subjects the page asks for a
search on the subject's own tags and for the videos YouTube relates to its heaviest video; it
then keeps the candidates closest to your subjects, whoever made them. A channel you watch
gets no advantage and a channel you never watched no handicap: at most `max_per_channel`
videos each.

With "vary the page" off, the same data gives the same page. On (the default), each
subject starts from one of its few heaviest videos and chance weighs a little against the
score, so a refresh shows another selection. Shorts, videos you finished, muted channels
and "not interested" videos are left out. With no subject at all (an empty history) the page
is empty: this algorithm has nothing to go on.
"""

import random

from ..base import Algorithm, Run
from ..catalogue import Catalogue, default_catalogue, video_id
from ..common_specs import max_per_channel, page_size, random_enabled
from ..contract import Snapshot, Source
from ..contract_algorithms import Traits
from ..contract_profile import Profile, ProfileGroup
from ..contract_rank import Candidate, Row
from ..embeddings import Embedder, default_embedder
from ..parameters import Number, Values
from ..subjects import Subject, subjects
from .signals_videos_ranking import VideosRanking

# Subjects searched, and subjects whose heaviest video starts related videos, per page.
SEARCH_SUBJECTS = 8
RELATED_SUBJECTS = 6


def _number(
    id: str, default: float, maximum: float, en: str, fr: str, help_en: str, help_fr: str
) -> Number:
    return Number(
        id=id,
        group="sources",
        label={"en": en, "fr": fr},
        help={"en": help_en, "fr": help_fr},
        min=0,
        max=maximum,
        step=5,
        default=default,
    )


class SignalsVideos(Algorithm):
    id = "signals-videos"
    name = {"en": "Signals, videos", "fr": "Signals, vidéos"}
    description = {
        "en": "Follows the subjects of what you watch, whoever made the video.",
        "fr": "Suit les sujets de ce que vous regardez, quel que soit le créateur.",
    }
    uses = ["history", "favorites", "not_interested", "shown"]
    sources = ["search", "related"]
    random = True
    traits = Traits(personal=3, discovery=3, variety=3)
    guide = {
        "for_you_if": {
            "en": "You care about what a video is about more than about who made it.",
            "fr": "Le sujet d'une vidéo compte plus pour vous que son créateur.",
        },
        "you_will_see": {
            "en": "One row per subject you watch, with videos on that subject from any creator.",
            "fr": "Une ligne par sujet que vous regardez, avec des vidéos de ce sujet de tout "
            "créateur.",
        },
        "good_to_know": {
            "en": "It reads subjects best for videos played in TypeType, whose tags it knows.",
            "fr": "Il lit mieux les sujets des vidéos lues dans TypeType, dont il connaît les "
            "tags.",
        },
    }
    specs = [
        _number(
            "search_per_subject",
            15,
            40,
            "Search results per subject",
            "Résultats de recherche par sujet",
            "Videos asked from a search on each of your main subjects.",
            "Vidéos demandées à une recherche sur chacun de vos sujets principaux.",
        ),
        _number(
            "related_per_subject",
            20,
            40,
            "Related videos per subject",
            "Vidéos liées par sujet",
            "Videos YouTube relates to the main video of each subject.",
            "Vidéos que YouTube associe à la vidéo principale de chaque sujet.",
        ),
        max_per_channel(default=2, maximum=6),
        page_size(),
        random_enabled(
            "Vary the page on every refresh", "Varier la page à chaque rafraîchissement"
        ),
    ]

    def __init__(self, embedder: Embedder | None = None, catalogue: Catalogue | None = None):
        self._embedder = embedder
        self._catalogue = catalogue

    @property
    def embedder(self) -> Embedder:
        return self._embedder or default_embedder()

    @property
    def catalogue(self) -> Catalogue:
        return self._catalogue or default_catalogue()

    def subjects(self, snapshot: Snapshot) -> list[Subject]:
        """The subjects the page is made from: the hidden ones are left out."""
        return [s for s in self.all_subjects(snapshot) if not s.muted]

    def all_subjects(self, snapshot: Snapshot) -> list[Subject]:
        return subjects(snapshot, self.embedder, self.catalogue)

    def profile(self, snapshot: Snapshot, values: Values) -> Profile:
        return Profile(
            groups=[
                ProfileGroup(
                    id=subject.id,
                    label=subject.label,
                    weight=round(subject.weight, 4),
                    from_=list(subject.members),
                    muted=subject.muted,
                )
                for subject in self.all_subjects(snapshot)
            ]
        )

    def plan(self, snapshot: Snapshot, values: Values) -> list[Source]:
        found = self.subjects(snapshot)
        sources: list[Source] = []
        searched = values.whole("search_per_subject")
        if searched:
            for subject in found[:SEARCH_SUBJECTS]:
                if subject.query:
                    sources.append(
                        Source(kind="search", query=subject.query, limit=searched, ref=subject.id)
                    )
        related = values.whole("related_per_subject")
        if related:
            # A page that varies starts each subject from one of its few heaviest videos.
            rng = random.Random(snapshot.context.seed)
            for subject in found[:RELATED_SUBJECTS]:
                start = subject.seed_url
                if values.flag("random_enabled") and subject.seed_urls:
                    start = rng.choice(subject.seed_urls)
                sources.append(
                    Source(kind="related", video_url=start, limit=related, ref=subject.id)
                )
        return sources

    def text_of(self, video: Candidate, facts: dict) -> str:
        known = facts.get(video_id(video.url) or "")
        return (known.text() if known else "") or video.title

    def rank(self, run: Run) -> list[Row]:
        return VideosRanking(self, run).rows()
