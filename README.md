<div align="center">
  <img src="https://raw.githubusercontent.com/TypeType-Video/TypeType/main/assets/banner.svg" alt="TypeType" width="100%">
  <h1>TypeType Signals</h1>
  <p>An algorithm you control.</p>
</div>

> **Experimental.** Version 0.0.1. Nothing here is settled: the algorithms, their settings
> and what they show will change. Do not rely on it.

Today, every recommendation algorithm pushes you to watch without end, and none of them
tells you how. TypeType is proud to present our vision of something else: an algorithm that
is transparent, and that you control.

It is far from perfect. It will keep evolving for as long as the project goes on.

## Thank you

To the community, for its suggestions. And above all to my friends, who will recognise
themselves: recommendation is a craft of its own, and I did not have all the skills for it.
Their advice and their work are what made this possible, and still do. They do not want to be
named publicly, and that choice is respected.

## The algorithms

There is no single right way to fill a home page, so there are several, side by side. You
pick one, you tune it, and you can change your mind at any time.

| Algorithm | What it does |
|---|---|
| Subscriptions | The newest videos from the channels you follow, in upload order. |
| Discovery | Your subscriptions, videos related to them and to your favorites, and trending videos. |
| Signals | Reacts to what you watch: videos you started, more from creators you watched, rows by topic, and one row of creators you never watched. |
| Signals, channels | The newest videos of the creators you watch most, and of your subscriptions. |
| Signals, videos | Follows the subjects of what you watch, whoever made the video. |
| Signals, mix | The creators you watch, and videos on your subjects from any creator. |

Whichever you choose:

- it says which of your data it reads, and you decide what it may read;
- every row on your home page says why it is there;
- what it concluded about you (the creators you watch, your interests, what you watched
  last) is shown to you, and you can correct, pin or hide any of it;
- every setting is on the table, with its default;
- randomness, when an algorithm uses it, is said, and can be turned off.

In TypeType, open **Algorithm** in the menu (`/algorithm`). The guide there compares the
algorithms side by side, and the other tabs show what yours keeps of you and how your last
home page was built. We let you discover it.

## If you use TypeType

There is nothing to install. If your instance offers the recommender, the Algorithm page is
in the menu and the option is off until you turn it on. Turn it off again and your home page
is the usual one, as if nothing had happened.

## If you host an instance

The recommender is optional. Without it, TypeType behaves exactly as before and shows no
option anywhere. With it, the administrator still has to switch it on, and each person then
chooses for themselves.

It needs:

- about 2 GB of memory (4 GB, with other settings, for a faster service under a crowd);
- about 1 GB of disk for its language model, downloaded once on first start;
- PostgreSQL with the `pgvector` extension, and the cache the stack already has.

It is added to the TypeType stack with one file. The steps, and how to remove it, are in the
[stack's README](https://github.com/TypeType-Video/TypeType#optional-recommender-experimental).
Every setting comes from the environment and is listed in the
[TypeType documentation](https://typetype-video.github.io/Docs-TypeType/).

What it keeps: public video titles and their vectors in its own database, and temporary
history snapshots sent by TypeType-Server, so a page does not need to send them again.
The default history TTL is 30 minutes and is configurable. Your data
stays in TypeType-Server, which also fetches the videos and applies blocking and family
filters to whatever comes back. This service only decides the order and the rows.

## If you are a developer

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing anything: it says how the
code is written, and how to prove that a change did not alter a single page.

```sh
docker compose up -d --build        # the service on :8190, with its database and cache
uv run pytest                       # tests, with a word-hashing stand-in for the model
```

The [TypeType documentation](https://typetype-video.github.io/Docs-TypeType/) holds the principles every algorithm
here must respect, each algorithm explained, what TypeType-Server sends and receives, and
what a person can see and change.

A new algorithm is one file in `recommender/algorithms/`. It must say what it reads, expose
every setting, and explain every row it makes.

## Versions

The algorithms follow [semantic versioning](https://semver.org). A patch changes no page, a
minor version may change what pages show or add settings, a major version changes the
contract with TypeType-Server. While the number starts with 0, anything may still change.
The version running on an instance is shown on its About page.

## License

GNU Affero General Public License, version 3 ([`LICENSE`](LICENSE)): anyone may use, study
and change the algorithms, and whoever runs a changed version for other people gives them
its source. What the service depends on is listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
