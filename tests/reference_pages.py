"""Pages of made-up people, to tell whether a change of code changes a result.

    uv run python -m tests.reference_pages write pages.json    # before the change
    uv run python -m tests.reference_pages check pages.json    # after it

Every algorithm plans, ranks and describes the profile of PEOPLE seeded people. Titles are
compared by their words (no model), so the same code gives the same file on any machine.
"""

import json
import os
import random
import sys
from pathlib import Path

os.environ["RECOMMENDER_EMBEDDER"] = "hashing"
os.environ.pop("RECOMMENDER_CACHE_URL", None)
os.environ.pop("RECOMMENDER_DATABASE_URL", None)

from fastapi.testclient import TestClient  # noqa: E402

from recommender.main import app  # noqa: E402

from .factories import DAY, HOUR, NOW, channel  # noqa: E402

PEOPLE = 24
SUBJECTS = {
    "bread": "sourdough bread baking oven crust flour starter recipe",
    "chess": "chess opening endgame gambit grandmaster tactics blunder",
    "linux": "linux kernel terminal distro package shell server",
    "volcano": "volcano eruption lava geology crater magma island",
    "guitar": "guitar chords solo acoustic lesson pedal amp",
    "trains": "train railway locomotive station steam track journey",
    "ramen": "ramen broth noodles tokyo recipe pork egg",
    "football": "football match goal league highlights transfer derby",
    "cuisine": "recette cuisine facile gâteau four beurre maison",
    "histoire": "histoire guerre empire roi révolution documentaire siècle",
    "料理": "料理 レシピ 簡単 ラーメン 東京 美味しい 作り方",
    "история": "история война империя документальный фильм век царь",
}
ALGORITHMS = [a["id"] for a in TestClient(app).get("/v1/algorithms").json()["algorithms"]]


def title(rng: random.Random, subject: str) -> str:
    words = SUBJECTS[subject].split()
    return (
        " ".join(rng.sample(words, k=min(len(words), rng.randint(3, 5)))) + f" {rng.randint(1, 99)}"
    )


def person(seed: int) -> dict:
    rng = random.Random(seed)
    mine = rng.sample(sorted(SUBJECTS), k=rng.randint(2, 5))
    makers = {s: [f"{s}-maker-{k}" for k in range(rng.randint(1, 4))] for s in mine}
    history = []
    for k in range(rng.choice([0, 3, 25, 80, 240])):
        subject = rng.choice(mine)
        maker = rng.choice(makers[subject])
        duration = rng.choice([45, 300, 600, 1800, 5400])
        history.append(
            {
                "id": f"history:{seed}-{k}",
                "video_url": f"https://www.youtube.com/watch?v=h{seed:02d}{k:04d}xxxx"[:43],
                "channel_url": channel(maker),
                "channel_name": maker,
                "title": title(rng, subject),
                "duration_s": duration,
                "progress_s": int(duration * rng.choice([0.05, 0.4, 0.9, 1.0])),
                "watched_at": NOW - int(rng.expovariate(1 / (20 * DAY))) - rng.randint(0, 6) * HOUR,
            }
        )
    follows = sorted({m for s in mine for m in makers[s] if rng.random() < 0.6})
    return {
        "context": {
            "now": NOW,
            "timezone": rng.choice(["Europe/Paris", "Asia/Tokyo", "UTC"]),
            "locale": rng.choice(["en", "fr"]),
            "seed": seed,
        },
        "history": history,
        "subscriptions": [
            {"id": f"subscription:{m}", "channel_url": channel(m), "name": m} for m in follows
        ],
        "_subjects": mine,
        "_makers": makers,
    }


def candidates(seed: int, who: dict, sources: list[dict]) -> list[dict]:
    rng = random.Random(seed * 7919 + 1)
    groups = []
    for n, source in enumerate(sources):
        videos = []
        for k in range(rng.randint(0, min(12, source.get("limit") or 12))):
            subject = rng.choice(who["_subjects"] if rng.random() < 0.7 else sorted(SUBJECTS))
            known = who["_makers"].get(subject, [])
            maker = (
                rng.choice(known)
                if known and rng.random() < 0.5
                else f"{subject}-new-{rng.randint(0, 9)}"
            )
            videos.append(
                {
                    "url": f"https://www.youtube.com/watch?v=c{seed:02d}{n:02d}{k:02d}xxxxx"[:43],
                    "title": title(rng, subject),
                    "uploader_url": channel(maker),
                    "uploader_name": maker,
                    "uploaded_at": NOW - int(rng.expovariate(1 / (30 * DAY))),
                    "view_count": int(rng.lognormvariate(9, 2)),
                    "duration_s": rng.choice([40, 400, 900, 3600]),
                }
            )
        groups.append({"source": source, "videos": videos})
    return groups


def pages() -> dict:
    client = TestClient(app)
    found = {}
    for seed in range(PEOPLE):
        who = person(seed)
        for algorithm in ALGORITHMS:
            snapshot = {k: v for k, v in who.items() if not k.startswith("_")}
            snapshot["algorithm"] = {
                "id": algorithm,
                "parameters": {"random_enabled": seed % 3 == 0} if seed % 2 else {},
            }
            plan = client.post("/v1/plan", json=snapshot)
            sources = plan.json().get("sources", []) if plan.status_code == 200 else []
            rank = client.post(
                "/v1/rank",
                json={
                    "snapshot": snapshot,
                    "candidates": candidates(seed, who, sources),
                    "trace": True,
                },
            )
            profile = client.post("/v1/profile", json=snapshot)
            found[f"{seed}:{algorithm}"] = {
                "plan": [plan.status_code, sources],
                "rank": [rank.status_code, rank.json()],
                "profile": [profile.status_code, profile.json()],
            }
    return found


if __name__ == "__main__":
    action, path = sys.argv[1], sys.argv[2]
    now = json.loads(json.dumps(pages(), sort_keys=True))
    if action == "write":
        Path(path).write_text(json.dumps(now, sort_keys=True))
        ranked = sum(1 for v in now.values() if v["rank"][0] == 200 and v["rank"][1].get("rows"))
        print(f"{len(now)} pages written, {ranked} with rows")
    else:
        before = json.loads(Path(path).read_text())
        different = sorted(k for k in before.keys() | now.keys() if before.get(k) != now.get(k))
        print(
            f"{len(now)} pages, {len(different)} different"
            + (f": {different[:8]}" if different else "")
        )
        sys.exit(1 if different else 0)
