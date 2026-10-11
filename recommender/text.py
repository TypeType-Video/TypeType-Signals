"""Every text the service returns, in the languages it knows (English as fallback)."""

Text = dict[str, str]


def pick(text: Text, locale: str) -> str:
    language = locale.split("-")[0].lower()
    return text.get(language, text["en"])


def fill(text: Text, locale: str, **values: object) -> str:
    return pick(text, locale).format(**values)
