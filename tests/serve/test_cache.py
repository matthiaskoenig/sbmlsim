"""The cache of the responses of the simulations and the route which serves it.

The model, the panel and the application of the tests are the fixtures of
`tests/serve/conftest.py`, whose application and client are of the test, so
every test is answered by a cache of its own.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import sbmlsim
from sbmlsim.serve.app import CACHE_CONTROL, NOT_STORED, create_app
from sbmlsim.serve.cache import MAX_ENTRIES, ResponseCache
from sbmlsim.serve.payload import Provenance, SimulateResponse, Status
from sbmlsim.serve.runner import ModelRegistry, PanelRunner
from tests.serve.conftest import SIMULATE

#: the request of the tests, which is simulated once
BODY = {"values": {"dose": 25.0}, "grid": {"egfr": 3}}
#: another request, which is another simulation
OTHER = {"values": {"dose": 12.5}, "grid": {}}
#: the caching of a stored response, the contract of the route; the literal and
#: not `CACHE_CONTROL`, which is the code under test
IMMUTABLE = "public, max-age=31536000, immutable"
#: the seconds an interleaved `put` is given to run inside the window of
#: another one, see `test_a_put_which_interleaves_another_one_waits_for_it`
INTERLEAVE = 0.2
#: the seconds a wait which must succeed gives up after, i.e. a failed test
TIMEOUT = 60.0


class Counter:
    """How often a panel was run, see the fixture `counting_runner`."""

    def __init__(self) -> None:
        self.calls = 0


class Interleaved(OrderedDict[str, SimulateResponse]):
    """The entries of a cache which let a hook in while a `put` stores.

    The hook runs once, inside `__setitem__` and after the entry was added,
    which is the window in the middle of a `put`: the entry is in the mapping
    and the eviction has not run yet.
    """

    def __init__(self, hook: Callable[[], None]) -> None:
        super().__init__()
        self._hook: Callable[[], None] | None = hook

    def __setitem__(self, key: str, value: SimulateResponse) -> None:
        super().__setitem__(key, value)
        hook, self._hook = self._hook, None
        if hook is not None:
            hook()


@pytest.fixture
def counting_runner(monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Count the runs of a panel, which are what a cache saves."""
    counter = Counter()
    run = PanelRunner.run

    def counted(self: PanelRunner, *args: Any, **kwargs: Any) -> Any:
        counter.calls += 1
        return run(self, *args, **kwargs)

    monkeypatch.setattr(PanelRunner, "run", counted)
    return counter


def _response(spec_hash: str, *, seconds: float = 1.0) -> SimulateResponse:
    """Get a response of a hash, the body a cache keeps.

    Args:
        spec_hash: the hash the response is stored under.
        seconds: the seconds of its provenance.

    Returns:
        The response.
    """
    return SimulateResponse(
        spec_hash=spec_hash,
        x=[0.0, 1.0],
        dims=[],
        curves={"Cve_hctz": [[0.0, 1.0]]},
        units={"x": "min", "Cve_hctz": "umol"},
        provenance=Provenance(
            model="hctz",
            panel="renal_plasma",
            model_sha256="0" * 64,
            variant=None,
            engine="roadrunner",
            sbmlsim=sbmlsim.__version__,
            seconds=seconds,
            created="2026-11-12T10:00:00+00:00",
        ),
        status=Status(ok=True, points=1, failed=[]),
    )


def _hash(character: str) -> str:
    """Get a hash of one character, see `_response`."""
    return character * 64


def _put(
    cache: ResponseCache, response: SimulateResponse, failed: list[BaseException]
) -> None:
    """Store a response and record what the `put` raised, if anything."""
    try:
        cache.put(response)
    except BaseException as error:
        failed.append(error)


def test_the_same_request_twice_simulates_once(
    client: TestClient, counting_runner: Counter
) -> None:
    bodies = [client.post(SIMULATE, json=BODY).json() for _ in range(2)]
    assert counting_runner.calls == 1
    assert bodies[0] == bodies[1]


def test_a_hit_carries_the_provenance_of_the_run_which_filled_the_cache(
    client: TestClient,
) -> None:
    """A body which is addressed by its hash is the one which was computed."""
    first = client.post(SIMULATE, json=BODY).json()
    again = client.post(SIMULATE, json=BODY).json()
    # the seconds and the time of the run which filled the cache, and not the
    # ones of the request which was answered from it
    assert again["provenance"] == first["provenance"]
    assert first["provenance"]["seconds"] > 0.0


def test_another_request_is_simulated_of_its_own(
    client: TestClient, counting_runner: Counter
) -> None:
    one = client.post(SIMULATE, json=BODY).json()
    other = client.post(SIMULATE, json=OTHER).json()
    assert counting_runner.calls == 2
    assert one["spec_hash"] != other["spec_hash"]


def test_a_stored_response_is_served_by_its_hash_as_immutable(
    client: TestClient,
) -> None:
    body = client.post(SIMULATE, json=BODY).json()
    answer = client.get(f"/api/sim/{body['spec_hash']}")
    assert answer.status_code == 200
    assert answer.json()["curves"] == body["curves"]
    assert answer.json() == body
    assert "immutable" in answer.headers["cache-control"]
    assert answer.headers["cache-control"] == IMMUTABLE
    assert CACHE_CONTROL == IMMUTABLE
    # the hash of a response is its lower case hexadecimal digits and nothing
    # else, so the route answers that one spelling of it
    upper = client.get(f"/api/sim/{body['spec_hash'].upper()}")
    assert upper.status_code == 404


def test_an_unknown_hash_is_404_and_runs_nothing(
    client: TestClient, never_runs: None
) -> None:
    answer = client.get(f"/api/sim/{_hash('0')}")
    assert answer.status_code == 404
    assert answer.json() == {"detail": NOT_STORED}


@pytest.mark.parametrize(
    "spec_hash",
    [
        "abc",
        _hash("0"),
        _hash("A"),
        _hash("g"),
        "a" * 63,
        "a" * 65,
        "a" * 4096,
        "%2E%2E",
    ],
)
def test_a_hash_which_is_not_stored_says_nothing_of_the_service(
    client: TestClient, never_runs: None, spec_hash: str
) -> None:
    """A hash of another shape is answered as an unknown one, not as an error."""
    answer = client.get(f"/api/sim/{spec_hash}")
    assert answer.status_code == 404
    assert answer.json() == {"detail": NOT_STORED}


def test_a_run_which_fails_is_not_stored(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure is no answer, so the next request of it runs again."""
    calls = []

    def broken(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        raise RuntimeError("the model did not integrate")

    monkeypatch.setattr(PanelRunner, "run", broken)
    for _ in range(2):
        assert client.post(SIMULATE, json=BODY).status_code == 500
    assert len(calls) == 2


def test_a_request_of_an_unknown_knob_is_422_and_is_not_answered_from_the_cache(
    client: TestClient,
) -> None:
    """The knobs are checked before the cache, which must not change a status."""
    body = client.post(SIMULATE, json=BODY).json()
    assert len(body["spec_hash"]) == 64
    named = dict(BODY["values"]) | {"kidney": 1.0}
    answer = client.post(SIMULATE, json={"values": named, "grid": BODY["grid"]})
    assert answer.status_code == 422
    assert "kidney" in answer.json()["detail"]


def test_the_cache_of_an_app_is_not_the_cache_of_another(
    registry: ModelRegistry,
) -> None:
    """The cache belongs to the app, so a body of one is not a body of another."""
    with (
        TestClient(create_app(registry)) as one,
        TestClient(create_app(registry)) as two,
    ):
        body = one.post(SIMULATE, json=BODY).json()
        assert one.get(f"/api/sim/{body['spec_hash']}").status_code == 200
        assert two.get(f"/api/sim/{body['spec_hash']}").status_code == 404


def test_the_cache_answers_what_it_was_given() -> None:
    cache = ResponseCache()
    assert cache.max_entries == MAX_ENTRIES
    assert len(cache) == 0
    assert cache.get(_hash("a")) is None
    response = _response(_hash("a"))
    cache.put(response)
    assert len(cache) == 1
    assert cache.get(_hash("a")) is response
    assert cache.get(_hash("b")) is None


def test_the_cache_evicts_the_least_recently_used_entry() -> None:
    """A `get` is a use, so the order is not the one of the puts alone."""
    cache = ResponseCache(max_entries=2)
    first, second, third = (_response(_hash(c)) for c in "abc")
    cache.put(first)
    cache.put(second)
    assert cache.get(first.spec_hash) is first
    cache.put(third)
    assert len(cache) == 2
    # `second` was put last but used first, so it is the one which is gone
    assert cache.get(second.spec_hash) is None
    assert cache.get(first.spec_hash) is first
    assert cache.get(third.spec_hash) is third


def test_a_response_which_is_stored_again_is_one_entry_and_the_newer_one() -> None:
    cache = ResponseCache(max_entries=2)
    first = _response(_hash("a"), seconds=1.0)
    cache.put(first)
    cache.put(_response(_hash("b")))
    again = _response(_hash("a"), seconds=2.0)
    cache.put(again)
    assert len(cache) == 2
    assert cache.get(_hash("a")) is again
    # the newer entry is the most recently used one, so the other one is evicted
    cache.put(_response(_hash("c")))
    assert cache.get(_hash("b")) is None
    assert cache.get(_hash("a")) is again


def test_a_cache_of_no_entry_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ResponseCache(max_entries=0)


def test_a_put_which_interleaves_another_one_waits_for_it() -> None:
    """The lock of the cache, pinned where a `put` without it breaks.

    A `put` of a full cache adds its entry and then evicts, and the thread of
    the second `put` is let in exactly between the two (`Interleaved`): it
    parks on the lock until the first `put` is through, which is the only
    order in which both entries are stored and one of them is evicted. A cache
    whose `put` does not hold the lock runs the second one inside that window,
    which evicts the entry the first one is storing and leaves it to fail in
    `move_to_end`.
    """
    cache = ResponseCache(max_entries=1)
    first, second = _response(_hash("a")), _response(_hash("b"))
    failed: list[BaseException] = []
    threads: list[threading.Thread] = []

    def interleave() -> None:
        thread = threading.Thread(target=_put, args=(cache, second, failed))
        threads.append(thread)
        thread.start()
        # the lock parks it, so the wait is the whole of it; an unlocked cache
        # runs the `put` here, well inside this window
        thread.join(timeout=INTERLEAVE)

    cache._entries = Interleaved(interleave)
    _put(cache, first, failed)
    for thread in threads:
        thread.join(timeout=TIMEOUT)
        assert not thread.is_alive()
    assert failed == []
    assert len(cache) == 1
    # the second `put` is the later one, so it is the entry which is kept
    assert cache.get(second.spec_hash) is second
    assert cache.get(first.spec_hash) is None


def test_many_threads_put_and_get_without_breaking_the_cache() -> None:
    """A smoke test of the workload, which a cache without its lock passes too.

    The guarantee of the lock is pinned by
    `test_a_put_which_interleaves_another_one_waits_for_it`; this one is the
    broad workload, four threads over more hashes than the cache keeps, and
    only says that it runs through and ends with the entries it may hold.
    """
    entries = 16
    cache = ResponseCache(max_entries=entries)
    hashes = [f"{index:064x}" for index in range(256)]

    def fill(start: int) -> None:
        for index in range(start, start + 64):
            cache.put(_response(hashes[index % len(hashes)]))
            cache.get(hashes[(index * 7) % len(hashes)])

    threads = [
        threading.Thread(target=fill, args=(start,)) for start in range(0, 256, 64)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=TIMEOUT)
    assert not [thread for thread in threads if thread.is_alive()]
    assert len(cache) == entries
    stored = {
        key: response for key in hashes if (response := cache.get(key)) is not None
    }
    assert len(stored) == entries
    assert all(key == response.spec_hash for key, response in stored.items())
