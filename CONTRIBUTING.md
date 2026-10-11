# Contributing

The code here decides what people see on their home page, and the settings screen links to
it (the algorithms are open and documented in plain language). It is meant to be read by them as much as by us.

## Before and after a change

```sh
uv run ruff format recommender tests
uv run ruff check recommender tests
uv run pytest
uv run python -m tests.reference_pages write /tmp/pages.json   # before
uv run python -m tests.reference_pages check /tmp/pages.json   # after
```

`tests.reference_pages` builds 144 made-up pages (24 people, 6 algorithms) and tells whether
one of them changed. A refactor or a fix that is not meant to change a page must leave all of
them identical.

## Versions

The algorithms follow [semantic versioning](https://semver.org); the number is in
`recommender/__init__.py` and `pyproject.toml` (a test keeps them equal). A patch changes no
page, a minor version may change what pages show or add settings, a major version changes
the contract with TypeType-Server. A change of code does not change a result without
changing the version.

## How the code is written

- **One module, one subject**, named after it. There is no limit of lines: a file is cut
  when it speaks of two things, not at a number. Past about 400 lines, ask the question.
  No file is past it today.
- **A function fits on a screen**: 60 lines at most, checked by `tests/test_code_shape.py`.
  A longer one becomes named steps.
- **No function inside a function** when something else needs it: it is a method or a
  function of the module. A small helper used only by the function around it may stay there.
- **No frozen switch**: a constant that is always `True` or always `False` hides code that
  never runs. The exception is a constant that carries a measurement and is flipped to make
  the comparison again (`DISCOVERY_OWN_CAP`, `DISCOVERY_ECHO_FIRST`,
  `SHORTS_ONLY_FOR_LONG_FORM_CHANNELS`, `UNKNOWN_PROGRESS_IS_SEEN`): its comment says what
  was measured.
- **Types** on what a function takes and gives (ruff checks it). A record is a `dataclass`
  or a `NamedTuple` with named fields, not a tuple read by position or a dictionary of
  anything. The measured exception: the two records made once per watch of a whole history
  (`Entry` in `signals_conclude.py`, the entries of `subjects.py`) stay plain tuples, named
  fields cost 4 to 5 % there.
- **Objects hand each other their result** rather than inherit to share state: the ranking
  of Signals is `Scoring`, then `Rows`, then `Ranking`, each given the previous one.
- **ruff decides the style**, formatting included. Nobody argues about spaces.
- **The explanations stay in the algorithm files**: why a number has its value, and what was
  measured to choose it, is written where the number is, in plain language.
- **A loop that runs once per watch** of a whole history is written for speed, and a change
  to it is timed: one more tuple per watch cost half a second on 261,000 watches, and a
  small function called per watch a tenth of a second.

## The time of a page

`tests/test_scale.py` holds the profile of three times the measured account (261,000
watches, 104,000 likes) under a page's 1.5 s, and requires the time to grow no faster than
the history. The loops that run once per watch or per like (`_count_watches`, `_count_likes`
and `inferred_shorts`) are written for that: what they need is read once before the loop, and
nothing is called per item that can be written in place. Time a change to them before and
after (the best of several runs, on the same machine, one after the other).

## Commits

Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
`type: short description`, one line, imperative mood, no emoji. Types: `feat`, `fix`,
`refactor`, `chore`, `docs`, `test`, `perf`, `style`, `build`, `ci`. One concern per commit.
