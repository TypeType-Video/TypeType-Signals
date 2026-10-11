"""Signals reads titles in every language and script, and every group gets a readable name."""

from recommender.labels import generic_words, group_words, informative, topic_text, words


def test_every_script_gives_words():
    assert words("हिंदी गाने नए") == {"हिंदी", "गाने", "नए"}  # Hindi vowel signs stay in words
    assert words("한국 음식 먹방") == {"한국", "음식", "먹방"}  # two-syllable Korean words
    assert {"日本", "音楽"} <= words("日本の音楽ライブ")  # no spaces: read two at a time
    assert {"中国", "美食", "小吃"} <= words("中国美食 街头小吃")
    assert words("Música brasileira ao vivo") == {"música", "brasileira", "vivo"}
    assert words("Français : les recettes de grand-mère") == {
        "français",
        "recettes",
        "grand",
        "mère",
    }


def test_arabic_is_the_same_word_with_or_without_its_vowel_points():
    assert words("سورة البَقَرَة كَامِلَة") == words("سورة البقرة كاملة")


def test_titles_in_any_script_can_join_a_group():
    for title in ["हिंदी गाने नए", "日本の音楽ライブ", "한국 음식 먹방", "เพลงไทยใหม่ล่าสุด"]:
        assert informative(title), title


def test_topic_text_keeps_the_marks_of_a_word():
    assert topic_text("हिंदी गाने #trending") == "हिंदी गाने"


def test_a_word_spread_over_every_group_does_not_name_one():
    topics = ("kebab", "minecraft", "pêche", "tennis", "cuisine", "voyage")  # real profiles: 100+
    groups = [[f"je suis {topic} {i}" for i in range(3)] for topic in topics]
    assert "suis" in generic_words(groups)
    assert "kebab" not in generic_words(groups)
    label, _, _ = group_words(groups[0], sum(groups, []), [], None, generic_words(groups))
    assert label == "Kebab"


def test_a_group_always_has_a_readable_name():
    label, named, _ = group_words(["PRIME M'BOLHI 2021", "Quand Maes parle de Benab"], [], ["", ""])
    assert label != "?" and not named
    label, _, _ = group_words(["PRIME M'BOLHI 2021", "Quand Maes parle"], [], ["Maes", ""])
    assert label == "Maes"


def test_the_invisible_mark_after_an_emoji_goes_with_it():
    assert topic_text("MINECRAFT 🔥️ HARDCORE ❤️ #fyp") == "MINECRAFT HARDCORE"
    assert topic_text("سورة البَقَرَة") == "سورة البَقَرَة"
