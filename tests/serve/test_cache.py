"""The cache of the responses of the simulations and the route which serves it.

The fixtures are the ones of the tests of the app, built here again: the two
files share no fixture, so a cache of one of them is never the cache of the
other. The app of a test is its own, as the cache belongs to the app, and the
registry is the one of the HCTZ example with the panel of the tests.
"""

import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sbmlsim
from examples.hctz_fitting import MODEL_PATH
from examples.hctz_fitting.project import PROJECT
from sbmlsim.panel import Choice, Knob, Panel, PanelCurve, PanelSimulation
from sbmlsim.project import Project
from sbmlsim.serve.app import CACHE_CONTROL, NOT_STORED, create_app
from sbmlsim.serve.cache import MAX_ENTRIES, ResponseCache
from sbmlsim.serve.payload import Provenance, SimulateResponse, Status
from sbmlsim.serve.runner import ModelEntry, ModelRegistry, PanelRunner

#: one day in the time unit of the model
END = 24 * 60.0
#: the output points of the panel, fewer than a real panel so the tests are quick
POINTS = 20
#: the simulation of the panel of the tests
SIMULATE = "/api/models/hctz/panels/renal/simulate"
#: the request of the tests, which is simulated once
BODY = {"values": {"dose": 25.0}, "grid": {"egfr": 3}}
#: another request, which is another simulation
OTHER = {"values": {"dose": 12.5}, "grid": {}}

EGFR = Knob(
    id="egfr",
    label="renal function",
    target="f_renal_function",
    unit="dimensionless",
    kind="range",
    min=0.2,
    max=1.0,
    steps=5,
    default=1.0,
)
DOSE = Knob(
    id="dose",
    label="dose",
    target="PODOSE_hctz",
    unit="mg",
    kind="choice",
    choices=(
        Choice(label="12.5 mg", value=12.5),
        Choice(label="25 mg", value=25.0),
        Choice(label="50 mg", value=50.0),
    ),
    default=25.0,
)


class Counter:
    """How often a panel was run, see the fixture `counting_runner`."""

    def __init__(self) -> None:
        self.calls = 0


@pytest.fixture(scope="module")
def panel() -> Panel:
    return Panel(
        id="renal",
        title="Hydrochlorothiazide in renal impairment",
        subtitle="the plasma concentration over the renal function",
        x_label="time (min)",
        knobs=(EGFR, DOSE),
        curves=(
            PanelCurve(id="Cve_hctz", label="hydrochlorothiazide plasma", unit="umol"),
        ),
        simulation=PanelSimulation(
            end=END, points=POINTS, preinit={"PODOSE_hctz": 0.0, "BW": 75.0}
        ),
    )


@pytest.fixture(scope="module")
def project(panel: Panel) -> Project:
    """The project of the HCTZ example with the panel of the tests."""
    return PROJECT.replace(variants={"default": MODEL_PATH}, panels={"renal": panel})


@pytest.fixture(scope="module")
def registry(project: Project) -> ModelRegistry:
    return ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=project, title="Hydrochlorothiazide")}
    )


@pytest.fixture
def app(registry: ModelRegistry) -> FastAPI:
    """An app of its own, whose cache of the responses is empty."""
    return create_app(registry)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


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


@pytest.fixture
def never_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if a panel is run at all."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the request reached the runner")

    monkeypatch.setattr(PanelRunner, "run", refuse)


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
            panel="renal",
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
    assert answer.headers["cache-control"] == CACHE_CONTROL
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


def test_the_cache_is_safe_from_several_threads() -> None:
    """The route is a sync `def`, so two requests are genuinely concurrent."""
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
        thread.join(timeout=60.0)
    assert not [thread for thread in threads if thread.is_alive()]
    assert len(cache) == entries
    stored = {
        key: response for key in hashes if (response := cache.get(key)) is not None
    }
    assert len(stored) == entries
    assert all(key == response.spec_hash for key, response in stored.items())
