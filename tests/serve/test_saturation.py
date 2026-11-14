"""The requests of one model never take every thread of the server.

The route of a simulation runs in the thread pool of the server (anyio's, 40
threads), and a request which waits for the lock of its model holds one of
them; the runner admits at most `MAX_WAITING` requests of one model and
refuses the others at once with a 503, so the other models and routes keep
answering while one model is saturated.

The runs of the saturated model are held until the test releases them, so
the requests of the test are genuinely concurrent and none of them finishes
before the test looked at the other model.
"""

import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from sbmlsim.project import Project
from sbmlsim.serve import runner as runner_module
from sbmlsim.serve.app import OVERLOADED, RETRY_AFTER, create_app
from sbmlsim.serve.runner import (
    MAX_WAITING,
    ModelBusy,
    ModelEntry,
    ModelRegistry,
    PanelRunner,
)
from tests.serve.conftest import DATA, SIMULATE

#: the requests of the saturated model, more than the threads of the server
REQUESTS = 48
#: the seconds a request of the other model may take while one is saturated;
#: it reads a declaration, which is milliseconds
ANSWER = 5.0
#: the seconds a wait which must succeed gives up after, i.e. a failed test;
#: it covers the load of the model by the first request
TIMEOUT = 120.0
#: a route of the other model, a sync route which needs a thread of the server
OTHER_PANEL = "/api/models/other/panels/renal_plasma"


@pytest.fixture
def two_models(project: Project) -> ModelRegistry:
    """Two models of one project, of which the test saturates the first."""
    return ModelRegistry(
        {
            "hctz": ModelEntry(id="hctz", project=project, title="saturated"),
            "other": ModelEntry(id="other", project=project, title="other"),
        }
    )


@pytest.fixture
def held(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[threading.Event, Any]]:
    """Hold every run of the model `hctz` until the test releases it.

    Yields:
        The event which releases the runs and a semaphore which is released
        once per run which reached the integrator.
    """
    release = threading.Event()
    entered = threading.Semaphore(0)
    run = runner_module._run

    def hold(warm: Any, *args: Any, **kwargs: Any) -> Any:
        if warm.key[0] == "hctz":
            entered.release()
            if not release.wait(timeout=TIMEOUT):
                raise TimeoutError("the test did not release the run")
        return run(warm, *args, **kwargs)

    monkeypatch.setattr(runner_module, "_run", hold)
    yield release, entered
    release.set()


def test_a_saturated_model_leaves_the_other_models_answering(
    two_models: ModelRegistry, held: tuple[threading.Event, Any]
) -> None:
    release, entered = held
    with (
        TestClient(create_app(two_models)) as client,
        ThreadPoolExecutor(max_workers=REQUESTS + 1) as pool,
    ):
        try:
            futures: list[Future[Any]] = [
                pool.submit(client.post, SIMULATE, json={}) for _ in range(REQUESTS)
            ]
            # the requests beyond the bound are answered at once, while the
            # admitted ones wait for the run which holds the lock
            refused = REQUESTS - MAX_WAITING
            deadline = time.monotonic() + TIMEOUT
            while sum(future.done() for future in futures) < refused:
                assert time.monotonic() < deadline, "the refusals did not come"
                time.sleep(0.01)
            assert entered.acquire(timeout=TIMEOUT), "no run reached the integrator"
            assert not release.is_set()

            started = time.monotonic()
            other = pool.submit(client.get, OTHER_PANEL).result(timeout=ANSWER)
            seconds = time.monotonic() - started
            assert other.status_code == 200, other.text
            assert seconds < ANSWER
            # the saturated model still holds its runs: the other model was
            # answered while it was saturated, and not after
            assert not release.is_set()
            assert sum(future.done() for future in futures) == refused
        finally:
            release.set()
        answers = [future.result(timeout=TIMEOUT) for future in futures]
    statuses = sorted(answer.status_code for answer in answers)
    assert statuses == [200] * MAX_WAITING + [503] * (REQUESTS - MAX_WAITING)
    for answer in answers:
        if answer.status_code == 503:
            assert answer.json() == {"detail": OVERLOADED}
            assert answer.headers["retry-after"] == str(RETRY_AFTER)


def test_the_refusal_names_nothing_of_the_service() -> None:
    assert "hctz" not in OVERLOADED
    assert "/" not in OVERLOADED
    assert RETRY_AFTER >= 1


def test_a_runner_refuses_a_request_beyond_the_bound_before_it_loads(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = PanelRunner(registry, max_waiting=1)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the refused request loaded the model")

    with runner._admitted("hctz"):
        monkeypatch.setattr(PanelRunner, "_warm_model", refuse)
        panel = registry.panel("hctz", "renal_plasma")
        with pytest.raises(ModelBusy):
            runner.run("hctz", panel, {}, {})
        with pytest.raises(ModelBusy):
            runner.units(registry.entry("hctz"), panel)
    # the bound is per model and is free again once a request left
    assert runner._in_runner == {"hctz": 0}


def test_a_runner_takes_at_least_one_request_of_a_model(
    registry: ModelRegistry,
) -> None:
    with pytest.raises(ValueError, match="at least one request"):
        PanelRunner(registry, max_waiting=0)


def test_a_page_of_four_panels_of_a_cold_model_is_never_refused(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulation and the study data of four panels wait for one load.

    The page of glimepiride of pkdb_models is four panels of one model, which a
    shell asks for at once, each its simulation and its study data, and every
    one of them waits for the load of the model when it is cold.
    """
    release = threading.Event()
    load = runner_module._load

    def held(*args: Any) -> Any:
        if not release.wait(timeout=TIMEOUT):
            raise TimeoutError("the test did not release the load")
        return load(*args)

    monkeypatch.setattr(runner_module, "_load", held)
    with (
        TestClient(create_app(registry)) as client,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        try:
            futures = [
                *(pool.submit(client.post, SIMULATE, json={}) for _ in range(4)),
                *(pool.submit(client.get, DATA) for _ in range(4)),
            ]
            # a refusal would answer at once, while the load is held
            time.sleep(1.0)
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        statuses = [future.result(timeout=TIMEOUT).status_code for future in futures]
    assert statuses == [200] * 8
