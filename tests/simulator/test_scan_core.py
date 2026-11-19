"""A scan of the engine core runs its points on the threads of the engine.

The engine simulates a batch of requests on threads of its own inside one
python process, so a scan of it starts no pool of processes: the points of a
chunk are one batch (`sbmlsim.simulator.worker._run_chunk_core`) and the
workers of the run are its threads. The result does not depend on them: a
scan of one thread and a scan of many are the same dataset, bitwise.
"""

from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("sbmlsim_core")

from sbmlsim import parallel
from sbmlsim.engine import Engine, UnsupportedModelError, import_core
from sbmlsim.model import load_model
from sbmlsim.model.model_core import CoreSBMLModel
from sbmlsim.resources import HCTZ_SBML
from sbmlsim.result.scan import ScanResult
from sbmlsim.simulation import PK, Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import IntegratorSettings, ScanError, Simulator, executor_core
from sbmlsim.simulator import simulator as simulator_module
from sbmlsim.simulator.executor import SteadyStateError

#: the simulation of every scan, an oral dose of HCTZ over a day in minutes
SIMULATION: Final[Simulation] = Simulation(
    end=24 * 60, times=np.linspace(0, 24 * 60, 25).tolist()
)

#: the tolerances at which the engines agree, see `test_simulator_core.py`
TIGHT: Final[IntegratorSettings] = {
    "absolute_tolerance": 1e-12,
    "relative_tolerance": 1e-10,
}

#: the observables of a scan: a value per simulation of the analysis of
#: pkpdutils and a timecourse of a formula of the selections
OBSERVABLES: Final[tuple[Any, ...]] = (
    PK("pk", "[Cve_hctz]", dose="PODOSE_hctz", route="oral"),
    Formula("total", "Aurine_hctz + Afeces_hctz"),
)


def scan_of(n: int, doses: Sequence[float] | None = None) -> Scan:
    """Get a scan of `n` oral doses of HCTZ, or of the doses which are given."""
    values = np.linspace(5.0, 100.0, n).tolist() if doses is None else list(doses)
    return Scan(SIMULATION, [Dimension("dose", values={"PODOSE_hctz": values})])


def _fail(message: str) -> Callable[..., Any]:
    """Get a function which fails a test by its message."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(message)

    return refuse


class _Executor:
    """An executor of batches which records the ones it runs, see `_Batches`."""

    def __init__(self, spy: _Batches, executor: Any, threads: int | None) -> None:
        """Record the batches of an executor of the loaded model."""
        self._spy = spy
        self._executor = executor
        self._threads = threads

    def simulate_many(self, requests: Sequence[dict[str, Any]]) -> list[Any]:
        """Record the points and the threads of a batch and run it."""
        self._spy.batches.append((len(requests), self._threads))
        return self._executor.simulate_many(requests)


class _Batches:
    """A `LoadedModel` which records the batches of a run, see `_spy`."""

    def __init__(self, loaded: Any) -> None:
        """Record the batches of the loaded model of a core model."""
        self.loaded = loaded
        self.batches: list[tuple[int, int | None]] = []

    def batch_executor(self, *, threads: int | None = None) -> _Executor:
        """Get an executor which records the batches it runs."""
        return _Executor(self, self.loaded.batch_executor(threads=threads), threads)

    def simulate_many(
        self, requests: Sequence[dict[str, Any]], *, threads: int | None = None
    ) -> list[Any]:
        """Record the points and the threads of a batch and run it."""
        self.batches.append((len(requests), threads))
        return self.loaded.simulate_many(requests, threads=threads)


def _spy(model: CoreSBMLModel, monkeypatch: pytest.MonkeyPatch) -> _Batches:
    """Record the batches of the model of a scan, which no point may run alone."""
    batches = _Batches(model.loaded)
    monkeypatch.setattr(model, "loaded", batches)
    # the executors of the model are the ones of the spy, also when the model
    # ran a batch before it was watched
    monkeypatch.setattr(model, "_executors", {})
    monkeypatch.setattr(executor_core, "execute", _fail("a point ran on its own"))
    return batches


def _point_error(kind: str, message: str, flag: int | None = None) -> Any:
    """Get a point of a batch which failed, the fields of a `PointError`."""

    class _Point:
        """A point of a batch which failed, see `PointError` of the bindings."""

        def __init__(self) -> None:
            self.kind = kind
            self.message = message
            self.flag = flag
            self.time = None

    # the fields the executor reads are the ones of the bindings
    assert {"kind", "message", "flag", "time"} <= set(dir(import_core().PointError))
    return _Point()


@pytest.fixture(scope="module")
def model_core() -> CoreSBMLModel:
    """Get the HCTZ model in the engine core, the reference model of the package."""
    model = load_model(HCTZ_SBML, Engine.CORE)
    assert isinstance(model, CoreSBMLModel)
    return model


def test_a_scan_of_the_core_does_not_start_a_pool(
    model_core: CoreSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The points of a scan of the engine run on its threads and not in a pool."""
    monkeypatch.setattr(parallel, "pool", _fail("a pool was started"))
    monkeypatch.setattr(parallel, "start_pool", _fail("a pool was started"))
    result = Simulator(engine="core", n_workers=4).run(model_core, scan_of(32))
    assert result.ds.sizes["dose"] == 32
    assert np.isfinite(result.ds["[Cve_hctz]"].values).all()


def test_the_points_of_a_chunk_are_one_batch_of_the_threads_of_the_run(
    model_core: CoreSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunk is one batch of the engine, on the threads the workers of the run give it.

    The points of a chunk of such a run are every point the cap of the engine
    allows and not a quarter of them per worker, which is the chunking of a
    pool: the parallelism of a run of the engine is inside its chunk, where
    the engine spreads the points of the batch over its threads itself, so a
    narrower chunk runs those threads with fewer points than they are and
    costs the waves of the batches it was cut into.
    """
    batches = _spy(model_core, monkeypatch)
    result = Simulator(engine="core", n_workers=4).run(model_core, scan_of(32))
    assert result.ds.sizes["dose"] == 32
    assert batches.batches == [(32, 4)]


def test_a_scan_of_one_point_per_thread_runs_them_in_one_batch(
    model_core: CoreSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scan of as many points as the run has threads is one batch of all of them.

    The four chunks per worker of a pool cut such a scan into four batches of
    a quarter of its points, which left three of four threads of the engine
    idle and cost four waves of integration where one does: the parallel scan
    of the large model of `docs/benchmarks.md`, eight points on eight
    workers, was 2.7 times its own time that way.
    """
    batches = _spy(model_core, monkeypatch)
    result = Simulator(engine="core", n_workers=8).run(model_core, scan_of(8))
    assert result.ds.sizes["dose"] == 8
    assert batches.batches == [(8, 8)]


def test_the_chunks_of_a_wide_scan_are_cut_at_the_cap_of_the_core(
    model_core: CoreSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The points of a chunk of the engine are capped below the ones of a pool.

    A chunk of the engine is one batch, which answers with the table of every
    one of its points at once, so the points of a wide scan are cut into more
    chunks than the four per worker of a run in a pool, see
    `sbmlsim.simulator.worker._run_chunk_core`.
    """
    most = simulator_module.CORE_MAX_CHUNK
    assert most < simulator_module.MAX_CHUNK
    batches = _spy(model_core, monkeypatch)
    # more than the four chunks of the one process of the run, which would be
    # `most + 1` points each
    n_points = 4 * most + 3
    result = Simulator(engine="core", n_workers=8).run(model_core, scan_of(n_points))
    assert result.ds.sizes["dose"] == n_points
    assert [points for points, _ in batches.batches] == [most] * 4 + [3]


def test_a_scan_does_not_depend_on_the_number_of_threads(
    model_core: CoreSBMLModel,
) -> None:
    """The result of a scan is the same dataset bitwise whatever the threads are.

    Every thread of a batch simulates on an instance of its own and the
    engine holds no state between two requests, so the points of a scan never
    see the thread which ran them.
    """
    one = Simulator(engine="core", n_workers=1).run(model_core, scan_of(64))
    many = Simulator(engine="core", n_workers=8).run(model_core, scan_of(64))
    assert one.ds.attrs["engine"] == "core"
    xr.testing.assert_identical(one.ds, many.ds)


def test_a_scan_does_not_depend_on_the_chunking(
    model_core: CoreSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunks of a run cut its points into batches and never reach its numbers."""
    scan = scan_of(16)
    whole = Simulator(engine="core", n_workers=4).run(model_core, scan)
    monkeypatch.setattr(
        simulator_module,
        "_chunk_size",
        lambda n_points, workers, most, batched=False: 1,
    )
    singles = Simulator(engine="core", n_workers=4).run(model_core, scan)
    xr.testing.assert_identical(whole.ds, singles.ds)


def test_a_scan_of_one_point_is_the_simulation_of_that_point(
    model_core: CoreSBMLModel,
) -> None:
    """A batch of one request is the simulation of that request, bitwise."""
    simulator = Simulator(engine="core", n_workers=1)
    scanned = simulator.run(model_core, Scan(SIMULATION, []))
    simulated = simulator.simulate(model_core, SIMULATION)
    compared = 0
    for k, column in enumerate(simulated.columns):
        if column in scanned.ds.data_vars:
            compared += 1
            np.testing.assert_array_equal(
                scanned.ds[column].values, simulated.values[:, k], err_msg=column
            )
    # every column of the simulation but the time, which is the coordinate of
    # the result and no variable of it
    assert compared == len(simulated.columns) - 1
    assert "time" not in scanned.ds.data_vars


def test_a_failed_point_is_flagged(model_core: CoreSBMLModel) -> None:
    """A point of a batch which fails is flagged and keeps the other points.

    A dose which is not finite is a request the engine refuses, the point
    roadrunner flags as well, see `test_simulator_core.py`.
    """
    scan = scan_of(0, doses=[10.0, float("nan"), 50.0])
    result = Simulator(engine="core", n_workers=4).run(
        model_core, scan, on_error="flag"
    )
    assert result.ds["status"].values.tolist() == [0, 1, 0]
    values = result.ds["[Cve_hctz]"].values
    assert np.isfinite(values[0]).all()
    assert np.isnan(values[1]).all()
    assert np.isfinite(values[2]).all()
    (error,) = result.ds.attrs["errors"]
    assert error.startswith("dose=1,")
    assert "RuntimeError" in error


def test_a_batch_the_engine_refuses_names_the_workers_of_the_run(
    model_core: CoreSBMLModel,
) -> None:
    """A batch the engine refuses, e.g. for its threads, is no error of the bindings.

    The engine runs a batch on at most 1024 threads and refuses more, which
    `n_workers` of the simulator decides, so the error of the run names it,
    see `sbmlsim.simulator.worker._batch`.
    """
    simulator = Simulator(engine="core", n_workers=2000)
    with pytest.raises(ValueError, match="'n_workers' of the run") as refused:
        simulator.run(model_core, scan_of(4))
    assert "2000 threads" in str(refused.value)
    assert isinstance(refused.value.__cause__, ValueError)


def test_a_point_whose_request_fails_is_that_point(
    model_core: CoreSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A point whose request cannot be built fails alone, as a point of the loop does.

    The request of a point is built from its plan, so a plan the executor
    refuses is the failure of its point and not of the batch of its chunk,
    and `on_error` decides about it as it decides about a simulation which
    fails.
    """
    request_of = executor_core.request_of

    def refusing(plan: Any, *args: Any, **kwargs: Any) -> Any:
        """Refuse the request of the point of the second dose."""
        if any(assignment.value == 50.0 for assignment in plan.preinit):
            raise RuntimeError("no request of this point")
        return request_of(plan, *args, **kwargs)

    monkeypatch.setattr(executor_core, "request_of", refusing)
    # the three points in one chunk, so that the two which keep their request
    # are one batch and run
    monkeypatch.setattr(
        simulator_module,
        "_chunk_size",
        lambda n_points, workers, most, batched=False: n_points,
    )
    scan = scan_of(0, doses=[10.0, 50.0, 100.0])
    flagged = Simulator(engine="core", n_workers=2).run(
        model_core, scan, on_error="flag"
    )
    assert flagged.ds["status"].values.tolist() == [0, 1, 0]
    assert np.isnan(flagged.ds["[Cve_hctz]"].values[1]).all()
    (error,) = flagged.ds.attrs["errors"]
    assert error.startswith("dose=1,")
    assert "no request of this point" in error
    with pytest.raises(ScanError, match="no request of this point"):
        Simulator(engine="core", n_workers=2).run(model_core, scan)


def test_the_first_failure_in_scan_order_is_raised(model_core: CoreSBMLModel) -> None:
    """The error of a scan is the first point which fails in the order of the scan.

    Every point of a batch runs, so the failures of a chunk are sorted and
    the first of them is raised, whatever the threads ran them in.
    """
    scan = scan_of(0, doses=[10.0, float("nan"), 50.0, float("nan")])
    with pytest.raises(ScanError) as caught:
        Simulator(engine="core", n_workers=4).run(model_core, scan, on_error="raise")
    message = str(caught.value)
    assert "dose=1" in message
    assert "dose=3" not in message


def test_the_observables_of_a_core_scan_are_the_ones_of_roadrunner() -> None:
    """The observables of a batch are evaluated on its points as roadrunner's are."""
    scan = scan_of(8)
    core = Simulator(engine="core", n_workers=4, **TIGHT).run(
        HCTZ_SBML, scan, observables=OBSERVABLES
    )
    roadrunner = Simulator(n_workers=1, **TIGHT).run(
        HCTZ_SBML, scan, observables=OBSERVABLES
    )
    assert set(core.ds.data_vars) == set(roadrunner.ds.data_vars)
    for name in ("pk.cmax", "pk.auc_inf_obs", "total"):
        np.testing.assert_allclose(
            core.ds[name].values, roadrunner.ds[name].values, rtol=1e-6, err_msg=name
        )


def test_a_failed_point_of_a_chunk_with_observables_is_flagged(
    model_core: CoreSBMLModel,
) -> None:
    """The observables of a chunk are evaluated on the points of its batch which ran.

    A point which failed is no native solution of the chunk, so it is flagged
    and the observables of the points around it are the ones of a batch
    without it.
    """
    scan = scan_of(0, doses=[10.0, float("nan"), 50.0])
    result = Simulator(engine="core", n_workers=2).run(
        model_core, scan, observables=OBSERVABLES, keep=["pk.cmax"], on_error="flag"
    )
    values = result.ds["pk.cmax"].values
    assert result.ds["status"].values.tolist() == [0, 1, 0]
    assert np.isfinite(values[0]) and np.isfinite(values[2])
    assert np.isnan(values[1])


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("unsupported", UnsupportedModelError),
        ("integration", RuntimeError),
        ("request", RuntimeError),
        ("internal", RuntimeError),
    ],
)
def test_the_error_of_a_point_of_a_batch_is_the_error_of_its_cause(
    model_core: CoreSBMLModel, kind: str, expected: type[Exception]
) -> None:
    """A `PointError` of a batch becomes the error its simulation would raise.

    The kinds of a point of a batch are the errors of a simulation of the
    engine, see `executor_core._failed`: a feature the engine does not
    support is raised for the whole scan and every other failure is a point a
    run flags.
    """
    failed = executor_core.point_error(
        _point_error(kind, "the message of the engine"), model_core
    )
    assert type(failed) is expected
    assert "the message of the engine" in str(failed)
    assert model_core.label in str(failed)


def test_a_point_without_a_steady_state_is_a_steady_state_error(
    model_core: CoreSBMLModel,
) -> None:
    """The flag of a presimulation which found no steady state is its own error."""
    point = _point_error(
        "integration", "no steady state", flag=executor_core.STEADY_STATE_FLAG
    )
    assert type(executor_core.point_error(point, model_core)) is SteadyStateError


def test_the_errors_of_a_scan_of_the_core_are_in_the_order_of_the_scan(
    model_core: CoreSBMLModel,
) -> None:
    """The failures of a batch are reported in the order of the scan, not of the threads."""
    doses = [10.0 if k % 2 else float("nan") for k in range(16)]
    result: ScanResult = Simulator(engine="core", n_workers=8).run(
        model_core, scan_of(0, doses=doses), on_error="flag"
    )
    errors = result.ds.attrs["errors"]
    assert [error.split(",")[0] for error in errors] == [
        f"dose={k}" for k in range(16) if k % 2 == 0
    ]
