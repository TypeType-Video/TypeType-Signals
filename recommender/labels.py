"""Names for interest groups, made from the words their titles share."""

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

# Words that say nothing about a topic, in the languages seen on real accounts.
STOPWORDS = set(
    """
    the and for with you your this that from are was were have has how what why when who
    not but all can will just its it's out get got into more most best new top one two
    i'm we our they them his her she him my me day days year years time first last full
    live stream streaming video videos vlog official episode part ep shorts short edit
    les des une pour avec dans sur par pas est qui que quoi son ses aux ces cette mon ton
    vous nous ils elle elles tout tous plus très fait faire comme mais ou où leur leurs
    j'ai c'est ça sont été être avoir chez sans entre après avant encore déjà
    el la los las con por para una uno del que más como
    yang dan di ke dari ini itu untuk dengan
    """.split()
)
MIN_TITLES = 2
MIN_SPREAD_TITLES = 6
MIN_CONCENTRATION = 0.2
# Words that describe the format of a video, not its topic.
FORMAT_NOISE = re.compile(
    r"#\w+|\bis live\b|\best en direct\b|\bestá en vivo\b|\blive\b|\bstream\b|\bshorts?\b",
    re.IGNORECASE,
)


# Signals reads titles in every language. A word is a run of letters with the marks that belong to
# them: Hindi, Thai or Arabic vowel signs are marks, and splitting on them left Hindi titles with no
# word at all. Scripts written without spaces (Chinese, Japanese, Thai, Lao, Khmer, Myanmar) are
# read two characters at a time. Where a letter is a syllable (Indic scripts, Hangul, those without
# spaces) two letters make a word; elsewhere three, as before.
_MARKS = "".join(
    chr(c) for c in range(0x300, 0x10000) if unicodedata.category(chr(c)) in ("Mn", "Mc", "Me")
)
_WORD = re.compile(f"(?:[^\\W\\d_]|[{re.escape(_MARKS)}])+")
_UNSPACED = re.compile(
    "[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf"
    "\u4e00-\u9fff\uf900-\ufaff]"
)
_SYLLABIC = re.compile("[\u0900-\u0dff\u1100-\u11ff\uac00-\ud7af]")
# Arabic and Hebrew vowel points are optional: the same word is written with or without them.
_OPTIONAL_POINTS = dict.fromkeys(
    [*range(0x0591, 0x05C8), *range(0x064B, 0x0660), 0x0670, *range(0x06D6, 0x06EE)]
)


def _letters(word: str) -> int:
    return sum(1 for ch in word if not unicodedata.category(ch).startswith("M"))


def _split(token: str) -> list[str]:
    if not _UNSPACED.search(token):
        least = 2 if _SYLLABIC.search(token) else 3
        return [token] if _letters(token) >= least else []
    base = [ch for ch in token if not unicodedata.category(ch).startswith("M")]
    return ["".join(base[i : i + 2]) for i in range(len(base) - 1)]


_ASCII_WORD = re.compile(r"[a-z]{3,}")
# Characters that need the careful path: marks, scripts without spaces, syllabic scripts.
_SPECIAL = re.compile(
    f"[{re.escape(_MARKS)}]|{_UNSPACED.pattern}|{_SYLLABIC.pattern}|[\u0590-\u06ff]"
)


def words(title: str) -> set[str]:
    text = title.lower()
    if text.isascii():  # most titles: no need for the rest
        return {w for w in _ASCII_WORD.findall(text) if w not in STOPWORDS}
    if not _SPECIAL.search(text):
        return {w for w in _WORD.findall(text) if len(w) >= 3 and w not in STOPWORDS}
    if not unicodedata.is_normalized("NFC", text):
        text = unicodedata.normalize("NFC", text)
    text = text.translate(_OPTIONAL_POINTS)
    return {w for token in _WORD.findall(text) for w in _split(token) if w not in STOPWORDS}


# The same titles come back at every page of the same person: both answers depend on the
# title alone, so they are kept (half the time of reading a 97,000-watch history went here).
@lru_cache(maxsize=200_000)
def topic_text(title: str) -> str:
    """The title with its format words, hashtags and emojis removed (letters keep their
    marks: a Hindi or Thai word without its vowel signs is another word)."""
    kept: list[str] = []
    for ch in FORMAT_NOISE.sub(" ", title):
        if ch.isalnum() or ch.isspace() or ch in "'-":
            kept.append(ch)
        elif unicodedata.category(ch).startswith("M") and kept and kept[-1].isalnum():
            # A mark belongs to the letter before it; one after an emoji (the invisible
            # U+FE0F after "❤") is dropped with the emoji.
            kept.append(ch)
        elif (
            kept
            and unicodedata.category(kept[-1]).startswith("M")
            and unicodedata.category(ch).startswith("M")
        ):
            kept.append(ch)  # a letter may carry several marks
    return " ".join("".join(kept).split())


@lru_cache(maxsize=200_000)
def informative(title: str) -> bool:
    # A title that is only a link (a removed video) names no topic.
    if title.strip().startswith(("http://", "https://")):
        return False
    return len(words(topic_text(title))) >= 2


def group_label(
    titles: list[str], all_titles: list[str], channel_names: list[str]
) -> tuple[str, bool]:
    label, named, _ = group_words(titles, all_titles, channel_names)
    return label, named


def title_counts(all_titles: list[str]) -> Counter:
    """In how many titles each word appears (computed once for every group of a profile)."""
    return Counter(w for t in all_titles for w in words(t))


def generic_words(groups_titles: list[list[str]]) -> set[str]:
    """Words spread over many interest groups rather than gathered in one: in any language,
    "je suis", "jamais", "の" are in every topic, a topic's own words in one or two. Measured
    on two real profiles: "jeu"
    had 17 % of its titles in its main group, "anyme" 8 %, against 33-92 % for "minecraft",
    "memes", "kebab"; a word in fewer than MIN_SPREAD_TITLES titles is not judged."""
    per_group = [Counter(w for t in titles for w in words(t)) for titles in groups_titles]
    total: Counter = Counter()
    best: Counter = Counter()
    for counts in per_group:
        total.update(counts)
        for w, n in counts.items():
            best[w] = max(best[w], n)
    return {
        w for w, n in total.items() if n >= MIN_SPREAD_TITLES and best[w] / n < MIN_CONCENTRATION
    }


def group_words(
    titles: list[str],
    all_titles: list[str],
    channel_names: list[str],
    everywhere: Counter | None = None,
    generic: set[str] | frozenset[str] = frozenset(),
) -> tuple[str, bool, list[str]]:
    """The two words most typical of these titles, or their two main channels, or else
    their first title: a group always has a readable name.

    The flag says whether the name comes from shared words. A group named
    after its channels has titles that share nothing readable: it is kept
    in the profile, but gets no row of its own on the home page.
    """
    inside = Counter(w for t in titles for w in words(t) if w not in generic)
    everywhere = title_counts(all_titles) if everywhere is None else everywhere
    total = max(1, len(all_titles))
    ranked = sorted(
        (w for w, n in inside.items() if n >= MIN_TITLES),
        key=lambda w: (-inside[w] * math.log(1 + total / max(1, everywhere[w])), w),
    )
    if len(ranked) >= 2:
        return f"{ranked[0].capitalize()} & {ranked[1]}", True, ranked[:3]
    top = [name for name, _ in Counter(channel_names).most_common(2) if name]
    if ranked and top:
        return f"{ranked[0].capitalize()} · {top[0]}", True, ranked[:1]
    if top or ranked:
        return " · ".join(top) or ranked[0].capitalize(), False, ranked[:1]
    first = topic_text(titles[0]) if titles else ""
    return (first[:40] + "…" if len(first) > 40 else first) or "?", False, []


# How long a text cut for a name or a search may be.
NAME_CHARS = 46


def shorten(text: str, limit: int = NAME_CHARS) -> str:
    """The text cut at a word, with an ellipsis, when it is longer than the limit."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] if " " in text[:limit] else text[:limit]
    return cut.rstrip(" ,;:-–—|(") + "…"
