"""The simulation experiments of a model as jobs of the service.

The model and its registry are the fixtures of `tests/serve/conftest.py`; the
application and its client are built here over a `JobStore` of the `tmp_path`
of the test, so that a test sees the jobs of its own store and nothing is
written outside of it.

One test runs a real experiment of the HCTZ example end to end, which is the
only way to see a job reach `done` with the figures it collected. Every other
test holds or replaces the run itself (`held`, `nothing_ran`), so the states of
a job, the bound of the jobs in flight and the paths of its figures are tested
without simulating anything.
"""

import logging
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from examples.hctz_fitting import MODEL_PATH
from sbmlsim.panel import CONTRACT
from sbmlsim.project import FIGURES_DIR, Project
from sbmlsim.serve.app import BUSY, JOBS_PREFIX, NO_FIGURE, NO_JOB, create_app
from sbmlsim.serve.jobs import MAX_EXPERIMENTS, RUN_FAILED, JobStore
from sbmlsim.serve.runner import ModelRegistry

#: the submission of the experiments of the model of the tests
EXPERIMENTS = "/api/models/hctz/experiments"

#: one experiment of the HCTZ example, the one the end to end test runs
ONE = "Patel1984"

#: the jobs the store of the tests keeps in flight; two, so that the bound is
#: reached with three submissions and one of them is genuinely queued
MAX_JOBS = 2

#: the seconds a test waits for a job; generous, since the end to end test
#: loads the model, runs the experiment and renders its figures
TIMEOUT = 600.0

#: an application of no model at all which writes into the directory of its
#: jobs and ends without the lifespan of a server, see the test of the
#: finalizer; it writes the directory it used on its stdout
FINALIZER = """
import sys
from pathlib import Path

from sbmlsim.serve.app import create_app
from sbmlsim.serve.runner import ModelRegistry

app = create_app(ModelRegistry({}))
root = Path(app.state.jobs.root)
(root / "a-job").mkdir(parents=True)
(root / "a-job" / "report.html").write_text("<p>a report</p>", encoding="utf-8")
sys.stdout.write(str(root))
"""


@pytest.fixture
def store(tmp_path: Path) -> Iterator[JobStore]:
    """A store of its own, below the `tmp_path` of the test."""
    store = JobStore(tmp_path / "jobs", max_jobs=MAX_JOBS)
    yield store
    store.close()


@pytest.fixture
def client(registry: ModelRegistry, store: JobStore) -> Iterator[TestClient]:
    """The client of an application over the store of the test."""
    with TestClient(create_app(registry, jobs=store)) as client:
        yield client


@pytest.fixture
def held(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """Hold every run of the experiments until the test releases it."""
    release = threading.Event()

    def wait(self: Project, experiments: Any, out: Any, **kwargs: Any) -> list[Any]:
        if not release.wait(timeout=TIMEOUT):
            raise TimeoutError("the test did not release the run")
        return []

    monkeypatch.setattr(Project, "run_experiments", wait)
    yield release
    release.set()


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run nothing, so that every job of the test finishes at once."""

    def nothing(self: Project, experiments: Any, out: Any, **kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr(Project, "run_experiments", nothing)


@pytest.fixture
def nothing_ran(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail the test if the experiments of a model are run at all.

    The run happens in the worker of the store and not in the thread of the
    test, where an exception of it would be swallowed by its future, so the
    run is recorded and the test fails here.
    """
    reached = threading.Event()

    def refuse(*args: Any, **kwargs: Any) -> Any:
        reached.set()
        return []

    monkeypatch.setattr(Project, "run_experiments", refuse)
    yield
    assert not reached.is_set(), "the submission reached the experiments"


@dataclass
class FailedResult:
    """One result of a run in which the experiment or its figures failed."""

    error: str | None = "the experiment did not run"
    failed_figures: dict[str, str] = field(default_factory=dict)
    experiment: object = None
    failed: bool = True


def submit(client: TestClient, *experiments: str) -> dict[str, Any]:
    """Submit experiments of the model of the tests and answer the job."""
    answer = client.post(EXPERIMENTS, json={"experiments": list(experiments)})
    assert answer.status_code == 202, answer.text
    body: dict[str, Any] = answer.json()
    return body


def finished(client: TestClient, job_id: str) -> dict[str, Any]:
    """Follow a job until it is finished and answer its body."""
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        answer = client.get(f"/api/jobs/{job_id}")
        assert answer.status_code == 200, answer.text
        body: dict[str, Any] = answer.json()
        if body["status"] in {"done", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"the job '{job_id}' did not finish in {TIMEOUT} s")


def test_submitting_an_experiment_answers_202_with_a_queued_job(
    client: TestClient, held: threading.Event
) -> None:
    answer = client.post(EXPERIMENTS, json={"experiments": ["all"]})
    assert answer.status_code == 202
    body = answer.json()
    assert body["status"] in {"queued", "running"}
    assert body["contract"] == CONTRACT
    assert body["model"] == "hctz"
    assert body["experiments"] == ["all"]
    assert body["message"] == ""
    assert body["figures"] == []
    assert body["id"]


def test_the_job_reaches_done_and_lists_its_figures(client: TestClient) -> None:
    """One experiment of the example runs end to end, as a caller sees it."""
    job = submit(client, ONE)
    body = finished(client, job["id"])
    assert body["status"] == "done", body["message"]
    assert body["message"] == ""
    assert body["id"] == job["id"]
    figures = body["figures"]
    assert figures, "the run collected no figure"
    assert all(figure.startswith(f"{FIGURES_DIR}/") for figure in figures)
    assert any(ONE in figure for figure in figures)
    figure = client.get(f"/api/jobs/{job['id']}/figures/{figures[0]}")
    assert figure.status_code == 200
    assert figure.content


def test_an_unknown_job_is_404(client: TestClient) -> None:
    answer = client.get("/api/jobs/nope")
    assert answer.status_code == 404
    assert answer.json() == {"detail": NO_JOB}


def test_an_unknown_experiment_name_is_422_and_submits_nothing(
    client: TestClient, store: JobStore
) -> None:
    answer = client.post(EXPERIMENTS, json={"experiments": ["Beermann1976", "Nope"]})
    assert answer.status_code == 422
    detail = answer.json()["detail"]
    assert "Nope" in detail
    assert "Beermann1976" in detail
    assert len(store) == 0
    assert not list(store.root.glob("*"))


def test_an_unknown_model_is_404_and_submits_nothing(
    client: TestClient, store: JobStore, nothing_ran: None
) -> None:
    answer = client.post("/api/models/nope/experiments", json={"experiments": [ONE]})
    assert answer.status_code == 404
    assert len(store) == 0


@pytest.mark.parametrize(
    "body",
    [
        {"experiments": []},
        {"experiments": [ONE] * (MAX_EXPERIMENTS + 1)},
        {"experiments": ONE},
        {"experiments": [ONE], "out": "/etc"},
        {},
    ],
)
def test_a_submission_which_is_no_request_is_422(
    client: TestClient, nothing_ran: None, body: dict[str, Any]
) -> None:
    answer = client.post(EXPERIMENTS, json=body)
    assert answer.status_code == 422


def test_a_failing_experiment_run_ends_as_failed_with_a_message_and_no_traceback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run which raises is the fixed message of the module, never its error."""

    def broken(self: Project, experiments: Any, out: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError(f"'{MODEL_PATH}' did not integrate")

    monkeypatch.setattr(Project, "run_experiments", broken)
    job = submit(client, ONE)
    body = finished(client, job["id"])
    assert body["status"] == "failed"
    assert body["message"] == RUN_FAILED
    assert body["figures"] == []
    text = client.get(f"/api/jobs/{job['id']}").text
    assert "Traceback" not in text
    assert "RuntimeError" not in text
    assert str(MODEL_PATH) not in text


@pytest.mark.parametrize(
    ("result", "logged"),
    [
        (FailedResult(), "the experiment did not run"),
        # an experiment whose figures alone failed has no `error` at all, so
        # the reason is in the figures and nowhere else
        (FailedResult(error=None, failed_figures={"Fig1": "no such curve"}), "Fig1"),
    ],
)
def test_an_experiment_which_failed_ends_the_job_as_failed(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    result: FailedResult,
    logged: str,
) -> None:
    """A run raises for nothing (`raise_on_failure=False`), its results say so."""

    def failing(self: Project, experiments: Any, out: Any, **kwargs: Any) -> list[Any]:
        assert kwargs["raise_on_failure"] is False
        return [result]

    monkeypatch.setattr(Project, "run_experiments", failing)
    with caplog.at_level(logging.ERROR, logger="sbmlsim.serve.jobs"):
        job = submit(client, ONE)
        body = finished(client, job["id"])
    assert body["status"] == "failed"
    assert body["message"] == "1 of 1 experiments failed."
    assert logged not in body["message"]
    assert logged in caplog.text


def test_a_job_runs_serially_into_its_own_directory(
    client: TestClient, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run in a thread of a server never starts a process pool."""
    recorded: dict[str, Any] = {}

    def record(self: Project, experiments: Any, out: Any, **kwargs: Any) -> list[Any]:
        recorded.update(kwargs, experiments=experiments, out=out)
        return []

    monkeypatch.setattr(Project, "run_experiments", record)
    job = submit(client, ONE)
    assert finished(client, job["id"])["status"] == "done"
    assert recorded["n_workers"] == 1
    assert recorded["scan_workers"] == 1
    assert recorded["progress"] is False
    assert recorded["raise_on_failure"] is False
    assert recorded["collect_figures"] is True
    assert recorded["experiments"] == [ONE]
    assert Path(recorded["out"]) == store.root / job["id"]


def test_more_than_max_jobs_in_flight_is_429(
    client: TestClient, held: threading.Event
) -> None:
    """The store runs one job at a time and keeps at most `max_jobs` of them."""
    for _ in range(MAX_JOBS):
        submit(client, ONE)
    answer = client.post(EXPERIMENTS, json={"experiments": [ONE]})
    assert answer.status_code == 429
    assert answer.json() == {"detail": BUSY}
    held.set()


def test_a_finished_job_leaves_room_for_the_next_one(
    client: TestClient, quick: None
) -> None:
    """The jobs in flight are the queued and the running ones, not every one."""
    for _ in range(MAX_JOBS + 2):
        job = submit(client, ONE)
        assert finished(client, job["id"])["status"] == "done"


def test_a_finished_job_over_the_cap_is_dropped_with_its_directory(
    registry: ModelRegistry, tmp_path: Path, quick: None
) -> None:
    """What a service keeps is bounded as what it runs at once is."""
    store = JobStore(tmp_path / "jobs", max_finished=2)
    with TestClient(create_app(registry, jobs=store)) as client:
        ids = []
        for _ in range(4):
            job = submit(client, ONE)
            assert finished(client, job["id"])["status"] == "done"
            ids.append(job["id"])
        assert len(store) == 2
        assert [one for one in ids if store.job(one) is not None] == ids[-2:]
        assert sorted(path.name for path in store.root.iterdir()) == sorted(ids[-2:])
        assert client.get(f"/api/jobs/{ids[0]}").status_code == 404
        assert client.get(f"/api/jobs/{ids[-1]}").status_code == 200
    store.close()


@pytest.mark.parametrize(
    "path",
    [
        # the dots are encoded: a client resolves a literal `..` away before
        # the request is sent, see the test below
        "%2e%2e/%2e%2e/etc/passwd",
        "%2e%2e",
        "/etc/passwd",
        FIGURES_DIR,
        "nope.svg",
    ],
)
def test_a_figure_path_outside_the_job_directory_is_404(
    client: TestClient, held: threading.Event, path: str
) -> None:
    job = submit(client, ONE)
    answer = client.get(f"/api/jobs/{job['id']}/figures/{path}")
    assert answer.status_code == 404
    assert answer.json() == {"detail": NO_FIGURE}


def test_a_traversal_in_the_url_of_a_figure_is_404(
    client: TestClient, held: threading.Event
) -> None:
    """A client resolves `..` away, so the route is not even reached."""
    job = submit(client, ONE)
    answer = client.get(f"/api/jobs/{job['id']}/figures/../../etc/passwd")
    assert answer.status_code == 404


def test_the_store_answers_a_file_of_the_job_and_nothing_else(
    store: JobStore, registry: ModelRegistry, held: threading.Event, tmp_path: Path
) -> None:
    """The one rule of a figure: a file of the directory of its job."""
    job = store.submit(registry.entry("hctz"), [ONE])
    directory = store.directory(job.id)
    assert directory is not None
    figure = directory / FIGURES_DIR / f"{ONE}_Fig1.svg"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_text("<svg />", encoding="utf-8")
    assert store.figures(job.id) == (f"{FIGURES_DIR}/{ONE}_Fig1.svg",)
    assert store.figure(job.id, f"{FIGURES_DIR}/{ONE}_Fig1.svg") == figure.resolve()
    assert store.figure(job.id, "../../etc/passwd") is None
    assert store.figure(job.id, f"{FIGURES_DIR}/../../{directory.name}") is None
    assert store.figure(job.id, str(tmp_path / "secret.svg")) is None
    assert store.figure(job.id, FIGURES_DIR) is None


def test_a_symbolic_link_out_of_the_job_directory_is_404(
    client: TestClient, store: JobStore, held: threading.Event, tmp_path: Path
) -> None:
    """Both sides are resolved, so a link cannot leave the directory either."""
    secret = tmp_path / "secret.svg"
    secret.write_text("<svg />", encoding="utf-8")
    job = submit(client, ONE)
    directory = store.directory(job["id"])
    assert directory is not None
    (directory / "link.svg").symlink_to(secret)
    answer = client.get(f"/api/jobs/{job['id']}/figures/link.svg")
    assert answer.status_code == 404
    assert answer.json() == {"detail": NO_FIGURE}


def test_an_id_of_a_request_never_reaches_the_filesystem(
    client: TestClient, store: JobStore
) -> None:
    assert store.figures("../..") == ()
    assert store.directory("../..") is None
    assert store.figure("../..", "nope.svg") is None
    answer = client.get(f"/api/jobs/nope/figures/{FIGURES_DIR}/one.svg")
    assert answer.status_code == 404
    assert answer.json() == {"detail": NO_FIGURE}


def test_an_application_which_runs_no_job_creates_no_directory(
    registry: ModelRegistry,
) -> None:
    """The root is created with the first job, never by `create_app` itself."""
    app = create_app(registry)
    root = Path(app.state.jobs.root)
    assert root.name.startswith(JOBS_PREFIX)
    assert not root.exists()
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
    assert not root.exists()


def test_the_application_removes_the_directory_of_the_jobs_it_created(
    registry: ModelRegistry, quick: None
) -> None:
    app = create_app(registry)
    root = Path(app.state.jobs.root)
    with TestClient(app) as client:
        job = submit(client, ONE)
        assert finished(client, job["id"])["status"] == "done"
        assert (root / job["id"]).is_dir()
    assert not root.exists()


def test_the_jobs_are_removed_when_the_process_ends_without_the_lifespan(
    tmp_path: Path,
) -> None:
    """The lifespan is the orderly way out, the finalizer covers the others.

    The child builds an application, writes into the directory of its jobs and
    ends without ever starting the server, so only the `weakref.finalize` of
    the store can remove it.
    """
    answer = subprocess.run(
        [sys.executable, "-c", FINALIZER],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    root = Path(answer.stdout.strip())
    assert root.name.startswith(JOBS_PREFIX)
    assert not root.exists()


def test_a_store_which_was_given_is_not_removed(
    registry: ModelRegistry, store: JobStore
) -> None:
    store.root.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(registry, jobs=store)):
        pass
    assert store.root.is_dir()


def test_a_store_of_fewer_than_one_job_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one job"):
        JobStore(tmp_path, max_jobs=0)
    with pytest.raises(ValueError, match="at least one finished job"):
        JobStore(tmp_path, max_finished=0)


def test_a_store_which_takes_no_job_creates_no_directory(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    store = JobStore(root)
    assert not root.exists()
    store.close()
    assert not root.exists()


def test_a_store_which_was_closed_keeps_its_records_and_takes_no_job(
    store: JobStore, registry: ModelRegistry, held: threading.Event
) -> None:
    entry = registry.entry("hctz")
    job = store.submit(entry, [ONE])
    held.set()
    store.close()
    assert store.job(job.id) is not None
    with pytest.raises(RuntimeError):
        store.submit(entry, [ONE])
    assert len(store) == 1
    # the job which was not taken left no directory behind either
    assert [path.name for path in store.root.iterdir()] == [job.id]


def test_the_repr_of_a_store_carries_no_record(store: JobStore) -> None:
    assert repr(store) == (
        f"JobStore(root={store.root}, max_jobs={MAX_JOBS}, "
        f"max_finished={store.max_finished})"
    )


def test_a_model_without_simulation_experiments_is_500(
    registry: ModelRegistry, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model which has no experiments at all is a failure of the service."""

    def refuse(self: Project, names: Any = None) -> Any:
        raise ValueError("the project 'hctz' has no simulation experiments")

    monkeypatch.setattr(Project, "experiment_classes", refuse)
    with TestClient(create_app(registry, jobs=store)) as client:
        answer = client.post(EXPERIMENTS, json={"experiments": [ONE]})
    assert answer.status_code == 500
    assert "simulation experiments" not in answer.text
    assert len(store) == 0
