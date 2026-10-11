"""Goal criterion 4, memory: the whole service within its budget, measured as the resident
memory of a real process, not traced Python allocations (the request's pydantic-core
objects and onnxruntime's buffers are invisible to tracemalloc).

Budgets set on 2026-09-30: 2 GB for everyone, 4 GB opted into by heavy users.
Child processes play pages as Server does (the history sent once by key, or the
whole snapshot on every page), plan then rank by the plan's id, for one or several users of
the measured account's size (87,064 watches, 34,577 likes) or a fraction of it, with the
hashing embedder; the model's share is added as measured on the reference machine with the
real model: whole account,
import cold, the real process at 1,755 MB (x1) and 2,439 MB (x3) against 346 and 903 MB
here, and 20 pages by key at 1,789 MB against 386 MB here, so the model, its imports
and its embedding buffers take 1,403 to 1,536 MB; the highest, rounded up, is used.
"""

import json
import os
import subprocess
import sys

import pytest

MODEL_SHARE_MB = 1_550
GENERAL_BUDGET_MB = 2_000
HEAVY_BUDGET_MB = 4_000

CHILD = """
import gc, json, os, sys
os.environ["RECOMMENDER_EMBEDDER"] = "hashing"
sys.path.insert(0, ".")
from tests.test_scale import big_snapshot
from recommender import engine, history_cache
from recommender.contract import HistoryUpload, Snapshot
from recommender.contract_rank import RankRequest

times = float(sys.argv[1])
pages, users, by_key = (int(a) for a in sys.argv[2:5])
base = big_snapshot(int(87_064 * times), int(34_577 * times)).model_dump_json()
# Other users: the same size, other videos.
texts = [base.replace("watch?v=", f"watch?v=u{user}") for user in range(users)]
del base
gc.collect()

def rss():
    with open("/proc/self/status") as file:
        return next(int(line.split()[1]) for line in file if line.startswith("VmRSS")) // 1024

before = rss()
light = []
for user, text in enumerate(texts):
    if by_key:  # sent once, as the history changes; pages name it
        whole = Snapshot.model_validate_json(text)
        key = f"u{user}"
        history_cache.remember_history(
            HistoryUpload(key=key, history=whole.history, favorites=whole.favorites)
        )
        empty = {"history": [], "favorites": [], "history_key": key}
        light.append(whole.model_copy(update=empty).model_dump_json())
        del whole
steps = []
for page in range(pages):
    for user in range(users):
        request = Snapshot.model_validate_json(light[user] if by_key else texts[user])
        plan = engine.plan(request)
        engine.rank(RankRequest(plan_id=plan.plan_id, candidates=[]))
        del request, plan
    steps.append(rss())
print(json.dumps({"before": before, "after_5": steps[min(4, pages - 1)], "after": steps[-1]}))
"""


def _pages(times: int, pages: int, users: int, by_key: bool, settings: dict | None = None) -> dict:
    """What pages add to a process (history, plans, profiles), in MB, after page 5 and at
    the end, with the model's share added."""
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(times), str(pages), str(users), str(int(by_key))],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **(settings or {})},
    )
    measured = json.loads(done.stdout.strip().splitlines()[-1])
    return {
        "after_5": measured["after_5"] - measured["before"] + MODEL_SHARE_MB,
        "after": measured["after"] - measured["before"] + MODEL_SHARE_MB,
    }


LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")


def _flat(used: dict) -> bool:
    return used["after"] - used["after_5"] < 100  # no page leaves memory behind


@LINUX
def test_twenty_pages_of_the_measured_account_fit_the_general_budget():
    used = _pages(1, 20, 1, by_key=True)
    assert used["after"] < GENERAL_BUDGET_MB and _flat(used), used


@LINUX
def test_three_ordinary_users_fit_the_general_budget():
    used = _pages(1 / 3, 20, 3, by_key=True)
    assert used["after"] < GENERAL_BUDGET_MB and _flat(used), used


@LINUX
def test_three_users_the_size_of_the_measured_account_fit_the_heavy_budget():
    used = _pages(1, 20, 3, by_key=True)
    assert used["after"] < HEAVY_BUDGET_MB and _flat(used), used


@LINUX
def test_a_client_sending_the_whole_history_every_page_stays_bounded():
    used = _pages(1, 20, 3, by_key=False)
    assert used["after"] < HEAVY_BUDGET_MB and _flat(used), used


@LINUX
def test_three_times_the_account_fits_the_heavy_budget():
    used = _pages(3, 10, 1, by_key=True)
    assert used["after"] < HEAVY_BUDGET_MB and _flat(used), used
