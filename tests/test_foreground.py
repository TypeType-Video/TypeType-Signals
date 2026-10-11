"""Background embedding waits while a request is being answered."""

from recommender import embeddings, engine

from .factories import snapshot


def test_background_waits_during_requests_and_resumes_after():
    assert embeddings._no_request.is_set()
    with embeddings.foreground():
        assert not embeddings._no_request.is_set()
        with embeddings.foreground():  # two requests at once
            pass
        assert not embeddings._no_request.is_set()
    assert embeddings._no_request.is_set()


def test_a_page_holds_the_background_while_it_runs(monkeypatch):
    seen = []
    real = engine._plan
    monkeypatch.setattr(
        engine, "_plan", lambda s: seen.append(embeddings._no_request.is_set()) or real(s)
    )
    engine.plan(snapshot("signals"))
    assert seen == [False]
    assert embeddings._no_request.is_set()


def test_a_failing_request_still_lets_the_background_go():
    try:
        with embeddings.foreground():
            raise RuntimeError
    except RuntimeError:
        pass
    assert embeddings._no_request.is_set()
