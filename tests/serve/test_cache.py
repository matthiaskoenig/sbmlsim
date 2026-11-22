"""The cache of the responses of the simulations and the route which serves it.

The model, the panel and the application of the tests are the fixtures of
`tests/serve/conftest.py`, whose application and client are of the test, so
every test is answered by a cache of its own.
"""

import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import sbmlsim
from sbmlsim.serve.app import CACHE_CONTROL, NOT_STORED, create_app
from sbmlsim.serve.cache import MAX_BYTES, ResponseCache, encode_response
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


class Interleaved(OrderedDict[str, bytes]):
    """The entries of a cache which let a hook in while a `put` stores.

    The hook runs once, inside `__setitem__` and after the entry was added,
    which is the window in the middle of a `put`: the entry is in the mapping
    and the eviction has not run yet.
    """

    def __init__(self, hook: Callable[[], None]) -> None:
        super().__init__()
        self._hook: Callable[[], None] | None = hook

    def __setitem__(self, key: str, value: bytes) -> None:
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
            variant="default",
            engine="roadrunner",
            sbmlsim=sbmlsim.__version__,
            seconds=seconds,
            created="2026-11-12T10:00:00+00:00",
        ),
        status=Status(ok=True, points=1, failed=[]),
    )


def _bytes() -> int:
    """Get the bytes of a body of `_response`, which are the same for all."""
    return len(encode_response(_response(_hash("a"))))


def _hash(character: str) -> str:
    """Get a hash of one character, see `_response`."""
    return character * 64


def _put(
    cache: ResponseCache, response: SimulateResponse, failed: list[BaseException]
) -> None:
    """Store the body of a response and record what the `put` raised, if anything."""
    try:
        cache.put(response.spec_hash, encode_response(response))
    except BaseException as error:
        failed.append(error)


def test_the_same_request_twice_simulates_once(
    client: TestClient, counting_runner: Counter
) -> None:
    bodies = [client.post(SIMULATE, json=BODY).json() for _ in range(2)]
    assert counting_runner.calls == 1
    assert bodies[0] == bodies[1]


def test_a_hit_is_answered_with_the_bytes_which_were_stored(
    client: TestClient, counting_runner: Counter
) -> None:
    """A hit serializes nothing on the event loop of the server (#521).

    The cache keeps the encoded body and the route answers it as it is, so a
    hit is the bytes of the miss, byte for byte, and the server encodes a
    response once however many requests ask for it.
    """
    first = client.post(SIMULATE, json=BODY)
    second = client.post(SIMULATE, json=BODY)

    assert counting_runner.calls == 1
    assert second.content == first.content
    assert second.headers["content-type"] == "application/json"
    # and the same bytes again by the hash of the request
    stored = client.get(f"/api/sim/{first.json()['spec_hash']}")
    assert stored.content == first.content


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
    assert cache.max_bytes == MAX_BYTES
    assert len(cache) == 0
    assert cache.nbytes == 0
    assert cache.get(_hash("a")) is None
    response = _response(_hash("a"))
    body = encode_response(response)
    assert cache.put(response.spec_hash, body)
    assert len(cache) == 1
    assert cache.nbytes == len(body)
    assert cache.get(_hash("a")) == body
    assert cache.get(_hash("b")) is None


def test_the_cache_evicts_the_least_recently_used_entry() -> None:
    """A `get` is a use, so the order is not the one of the puts alone."""
    cache = ResponseCache(max_bytes=2 * _bytes())
    first, second, third = (_response(_hash(c)) for c in "abc")
    for response in (first, second):
        cache.put(response.spec_hash, encode_response(response))
    assert cache.get(first.spec_hash) == encode_response(first)
    cache.put(third.spec_hash, encode_response(third))
    assert len(cache) == 2
    # `second` was put last but used first, so it is the one which is gone
    assert cache.get(second.spec_hash) is None
    assert cache.get(first.spec_hash) == encode_response(first)
    assert cache.get(third.spec_hash) == encode_response(third)


def test_a_response_which_is_stored_again_is_one_entry_and_the_newer_one() -> None:
    cache = ResponseCache(max_bytes=2 * _bytes())
    first = _response(_hash("a"), seconds=1.0)
    cache.put(first.spec_hash, encode_response(first))
    cache.put(_hash("b"), encode_response(_response(_hash("b"))))
    again = _response(_hash("a"), seconds=2.0)
    cache.put(again.spec_hash, encode_response(again))
    assert len(cache) == 2
    assert cache.get(_hash("a")) == encode_response(again)
    # the newer entry is the most recently used one, so the other one is evicted
    cache.put(_hash("c"), encode_response(_response(_hash("c"))))
    assert cache.get(_hash("b")) is None
    assert cache.get(_hash("a")) == encode_response(again)


def test_a_cache_of_no_byte_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one byte"):
        ResponseCache(max_bytes=0)


def test_a_put_which_interleaves_another_one_waits_for_it() -> None:
    """The lock of the cache, pinned where a `put` without it breaks.

    A `put` of a full cache adds its entry, then counts its bytes and then
    evicts, and the thread of the second `put` is let in exactly between the
    first two steps (`Interleaved`): it parks on the lock until the first `put`
    is through, which is the only order in which both entries are counted and
    one of them is evicted. A cache whose `put` does not hold the lock runs the
    second one inside that window, which the test sees as a `put` which ended
    inside it.
    """
    cache = ResponseCache(max_bytes=_bytes())
    first, second = _response(_hash("a")), _response(_hash("b"))
    failed: list[BaseException] = []
    threads: list[threading.Thread] = []
    #: whether the second `put` ended inside the window of the first one
    inside: list[bool] = []

    def interleave() -> None:
        thread = threading.Thread(target=_put, args=(cache, second, failed))
        threads.append(thread)
        thread.start()
        # the lock parks it, so the wait is the whole of it; an unlocked cache
        # runs the `put` here, well inside this window
        thread.join(timeout=INTERLEAVE)
        inside.append(not thread.is_alive())

    cache._entries = Interleaved(interleave)
    _put(cache, first, failed)
    for thread in threads:
        thread.join(timeout=TIMEOUT)
        assert not thread.is_alive()
    assert failed == []
    assert inside == [False]
    assert len(cache) == 1
    assert cache.nbytes == len(encode_response(second))
    # the second `put` is the later one, so it is the entry which is kept
    assert cache.get(second.spec_hash) == encode_response(second)
    assert cache.get(first.spec_hash) is None


def test_many_threads_put_and_get_without_breaking_the_cache() -> None:
    """A smoke test of the workload, which a cache without its lock passes too.

    The guarantee of the lock is pinned by
    `test_a_put_which_interleaves_another_one_waits_for_it`; this one is the
    broad workload, four threads over more hashes than the cache keeps, and
    only says that it runs through and ends with the entries it may hold.
    """
    entries = 16
    cache = ResponseCache(max_bytes=entries * _bytes())
    hashes = [f"{index:064x}" for index in range(256)]

    def fill(start: int) -> None:
        for index in range(start, start + 64):
            key = hashes[index % len(hashes)]
            cache.put(key, encode_response(_response(key)))
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
    stored = {key: body for key in hashes if (body := cache.get(key)) is not None}
    assert len(stored) == entries
    assert all(json.loads(body)["spec_hash"] == key for key, body in stored.items())


def test_the_cache_evicts_until_the_entries_fit_its_bytes() -> None:
    """A bound of bytes: one large response takes the room of several small ones."""
    small = _bytes()
    cache = ResponseCache(max_bytes=4 * small)
    for character in "abcd":
        response = _response(_hash(character))
        assert cache.put(response.spec_hash, encode_response(response))
    assert len(cache) == 4
    large = _large(_hash("e"), nbytes=2 * small)
    body = encode_response(large)
    assert small < len(body) < 4 * small
    assert cache.put(large.spec_hash, body)
    assert cache.nbytes <= cache.max_bytes
    assert cache.get(large.spec_hash) == body
    # the least recently used ones made the room, the newest small one stays
    assert cache.get(_hash("a")) is None
    assert cache.get(_hash("d")) is not None


def test_a_response_larger_than_the_cache_is_not_kept() -> None:
    small = _bytes()
    cache = ResponseCache(max_bytes=2 * small)
    kept = _response(_hash("a"))
    assert cache.put(kept.spec_hash, encode_response(kept))
    large = _large(_hash("b"), nbytes=3 * small)
    assert not cache.put(large.spec_hash, encode_response(large))
    assert cache.get(_hash("b")) is None
    # nothing was evicted for it
    assert cache.get(_hash("a")) == encode_response(kept)
    assert cache.nbytes == small
    # and a key it replaces is no longer answered with an older body
    replacing = _large(_hash("a"), nbytes=3 * small)
    assert not cache.put(replacing.spec_hash, encode_response(replacing))
    assert cache.get(_hash("a")) is None
    assert cache.nbytes == 0


def test_an_answer_larger_than_the_cache_is_served_and_not_stored(
    registry: ModelRegistry,
    counting_runner: Counter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sbmlsim.serve.app.ResponseCache", lambda: ResponseCache(max_bytes=1024)
    )
    with TestClient(create_app(registry)) as client:
        answer = client.post(SIMULATE, json=BODY)
        assert answer.status_code == 200
        body = answer.json()
        assert client.post(SIMULATE, json=BODY).json()["curves"] == body["curves"]
        assert counting_runner.calls == 2
        assert client.get(f"/api/sim/{body['spec_hash']}").status_code == 404


@pytest.mark.parametrize(
    ("curves", "points", "times"),
    [(1, 1, 201), (2, 1, 201), (1, 64, 2), (4, 64, 501)],
)
def test_the_bytes_of_an_entry_are_the_bytes_of_the_body_it_serves(
    curves: int, points: int, times: int
) -> None:
    """The cache measures the body it keeps, which is what the route answers.

    It kept the response and estimated its objects before, see `cache` (#521).
    """
    response = _shaped(curves, points, times)
    body = encode_response(response)
    cache = ResponseCache()

    assert cache.put(response.spec_hash, body)

    assert cache.nbytes == len(body)
    assert cache.get(response.spec_hash) == body
    assert json.loads(body)["spec_hash"] == response.spec_hash


def test_a_number_of_a_body_is_about_twenty_bytes() -> None:
    """The figures the module of the cache states, measured once.

    The four curves of the glimepiride page over its largest grid, 64 points
    of 501 times, are about 2.6 MB of JSON, which is what the bound of a cache
    of 256 MiB is read with.
    """
    response = _shaped(curves=4, points=64, times=501)
    numbers = len(response.x) + 4 * 64 * 501

    per_number = len(encode_response(response)) / numbers

    assert 16.0 <= per_number <= 24.0
    assert 2.0e6 <= len(encode_response(response)) <= 3.2e6


def _large(spec_hash: str, nbytes: int) -> SimulateResponse:
    """Get a response of a hash whose body is at least a number of bytes."""
    numbers = max(1, nbytes // 20)
    while True:
        response = _response(spec_hash).model_copy(
            update={
                "curves": {
                    "Cve_hctz": [[k + 0.123456789012345 for k in range(numbers)]]
                }
            }
        )
        if len(encode_response(response)) >= nbytes:
            return response
        numbers *= 2


def _shaped(curves: int, points: int, times: int) -> SimulateResponse:
    """Get a response of curves of distinct floats, as a run gives them."""
    rng = np.random.default_rng(1)
    return SimulateResponse(
        spec_hash=_hash("f"),
        x=rng.random(times).tolist(),
        dims=[],
        curves={
            f"curve{k}": [rng.random(times).tolist() for _ in range(points)]
            for k in range(curves)
        },
        units={"x": "min", **{f"curve{k}": "uM" for k in range(curves)}},
        provenance=_response(_hash("f")).provenance,
        status=Status(ok=True, points=points, failed=[]),
    )
