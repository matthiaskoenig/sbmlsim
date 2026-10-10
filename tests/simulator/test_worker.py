"""A chunk of a scan runs its points on one plan and one model."""

import ctypes
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulation.observables import Custom, Formula
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.observables import compile_observables, identity_graph
from sbmlsim.simulator.plan import Plan, compile_simulation
from sbmlsim.simulator.simulator import scan_point_plans
from sbmlsim.simulator.worker import (
    MAX_ERRORS,
    Chunk,
    ChunkResult,
    ModelSpec,
    OnError,
    ScanPointError,
    observe,
    run_chunk,
    run_chunk_in_worker,
)
from tests.simulator.models import BLOWUP, fails_for_large_k1, sbml, sbml_pk

posix = pytest.mark.skipif(
    sys.platform == "win32", reason="the streams of C are flushed with POSIX ctypes"
)

SEL = ("time", "[A]", "[B]", "k1")


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    return RoadrunnerSBMLModel(source=sbml())


def _chunk(
    model: RoadrunnerSBMLModel,
    simulation: Simulation,
    values: dict[str, np.ndarray] | None = None,
    timed: dict[float, dict[str, np.ndarray]] | None = None,
    time: np.ndarray | None = None,
    selections: tuple[str, ...] = SEL,
    on_error: OnError = "raise",
    indices: np.ndarray | None = None,
) -> Chunk:
    return Chunk(
        indices=np.arange(2) if indices is None else indices,
        plan=compile_simulation(simulation, model.symbols, model.uinfo),
        model=0,
        graph=identity_graph(selections),
        values=values or {},
        timed=timed or {},
        time=time,
        on_error=on_error,
    )


def test_a_chunk_without_observables_has_no_scalars(model: RoadrunnerSBMLModel) -> None:
    result = run_chunk(_chunk(model, Simulation(end=2, steps=4)), model)
    assert result.scalars.shape == (2, 0)


def test_every_point_is_its_simulation(model: RoadrunnerSBMLModel) -> None:
    chunk = _chunk(
        model,
        Simulation(end=2, steps=4),
        values={"b0": np.array([0.0, 2.0])},
        timed={1.0: {"k1": np.array([0.1, 3.0])}},
    )
    result = run_chunk(chunk, model)
    assert result.values.shape == (2, 5, 4)
    assert result.status.tolist() == [0, 0]
    for k, (b0, k1) in enumerate([(0.0, 0.1), (2.0, 3.0)]):
        simulation = Simulation(
            end=2,
            steps=4,
            preinit_changes={"b0": b0},
            changes=[Change(1.0, {"k1": k1})],
        )
        plan = compile_simulation(simulation, model.symbols, model.uinfo)
        np.testing.assert_allclose(
            result.values[k], execute(plan, model, SEL).values, rtol=1e-12
        )


def test_the_points_are_padded_to_the_longest(model: RoadrunnerSBMLModel) -> None:
    chunk = _chunk(model, Simulation(end=2), values={"k1": np.array([0.1, 30.0])})
    result = run_chunk(chunk, model)
    rows = [int(np.isfinite(result.values[k, :, 0]).sum()) for k in range(2)]
    assert rows[0] < rows[1] == result.values.shape[1]
    assert np.isnan(result.values[0, rows[0] :]).all()


def test_a_grid_of_times_is_interpolated(model: RoadrunnerSBMLModel) -> None:
    grid = np.array([0.0, 0.5, 1.0, 3.0])
    chunk = _chunk(
        model,
        Simulation(end=2, changes=[Change(1.0, {"[A]": 5.0})]),
        values={"k1": np.array([0.1, 0.2])},
        time=grid,
    )
    result = run_chunk(chunk, model)
    assert result.values.shape == (2, 4, 4)
    np.testing.assert_array_equal(result.values[0, :, 0], grid)
    # the value after the change and none after the end
    assert result.values[0, 2, 1] == pytest.approx(5.0)
    assert np.isnan(result.values[0, 3, 1])


def _blowup_chunk(
    on_error: OnError, k: list[float] | None = None, time: np.ndarray | None = None
) -> tuple[Chunk, RoadrunnerSBMLModel]:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    k = k or [0.1, 2.0, 0.1]
    chunk = _chunk(
        blowup,
        Simulation(end=1, steps=4),
        values={"k": np.array(k)},
        selections=("time", "S"),
        on_error=on_error,
        indices=np.arange(4, 4 + len(k)),
        time=time,
    )
    return chunk, blowup


def _run_failing(chunk: Chunk, model: RoadrunnerSBMLModel, capfd) -> ChunkResult:
    """Run a chunk with failing points and take the output of SUNDIALS.

    roadrunner does not log the error of a point, the result has it.
    """
    capfd.readouterr()
    try:
        return run_chunk(chunk, model)
    finally:
        ctypes.CDLL(None).fflush(None)
        captured = capfd.readouterr()
        assert "CVODE Error" not in captured.out + captured.err


@posix
def test_a_failed_point_is_flagged(capfd) -> None:
    chunk, blowup = _blowup_chunk("flag")
    result = _run_failing(chunk, blowup, capfd)
    assert result.status.tolist() == [0, 1, 0]
    assert np.isnan(result.values[1, :, 1]).all()
    assert np.isfinite(result.values[[0, 2]]).all()
    # the point after a failure is the point it would be alone
    np.testing.assert_array_equal(result.values[2], result.values[0])
    assert result.errors[0][0] == 5
    assert "CVODE" in result.errors[0][1]


@posix
def test_a_failed_point_raises_with_its_index(capfd) -> None:
    chunk, blowup = _blowup_chunk("raise")
    with pytest.raises(ScanPointError) as info:
        _run_failing(chunk, blowup, capfd)
    assert info.value.index == 5
    again = pickle.loads(pickle.dumps(info.value))
    assert (again.index, again.message) == (5, info.value.message)


@posix
def test_a_failed_point_has_the_grid_as_time(capfd) -> None:
    grid = np.array([0.0, 0.5, 1.0])
    chunk, blowup = _blowup_chunk("flag", time=grid)
    result = _run_failing(chunk, blowup, capfd)
    assert result.status.tolist() == [0, 1, 0]
    np.testing.assert_array_equal(result.values[1, :, 0], grid)
    assert np.isnan(result.values[1, :, 1]).all()


@posix
def test_the_errors_are_capped(capfd) -> None:
    chunk, blowup = _blowup_chunk("flag", k=[2.0] * (MAX_ERRORS + 2))
    result = _run_failing(chunk, blowup, capfd)
    assert result.status.sum() == MAX_ERRORS + 2
    assert [i for i, _ in result.errors] == list(range(4, 4 + MAX_ERRORS))


def test_a_wrong_definition_is_no_failed_point(model: RoadrunnerSBMLModel) -> None:
    chunk = _chunk(
        model,
        Simulation(end=1),
        timed={5.0: {"k1": np.array([1.0, 2.0])}},
        on_error="flag",
    )
    with pytest.raises(ValueError):
        run_chunk(chunk, model)


def test_a_chunk_and_its_result_pickle(model: RoadrunnerSBMLModel) -> None:
    chunk = _chunk(
        model,
        Simulation(end=1, steps=2),
        values={"k1": np.array([0.1, 0.2])},
        timed={0.5: {"b0": np.array([1.0, 2.0])}},
        time=np.array([0.0, 1.0]),
    )
    again = pickle.loads(pickle.dumps(chunk))
    first, second = run_chunk(chunk, model), run_chunk(again, model)
    again_result = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(again_result.values, first.values)
    assert again_result.errors == first.errors
    np.testing.assert_array_equal(again_result.status, first.status)


def test_a_worker_sees_a_rewritten_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    path = tmp_path / "blowup.xml"
    path.write_text(sbml(BLOWUP))
    first = RoadrunnerSBMLModel(source=path)
    spec = ModelSpec.of(first)
    path.write_text(sbml(BLOWUP.replace("species S in C = 1", "species S in C = 0.5")))
    second = RoadrunnerSBMLModel(source=path)
    other = ModelSpec.of(second)
    assert other.key != spec.key
    chunk = _chunk(
        second,
        Simulation(end=1, steps=2),
        values={"k": np.array([0.1])},
        selections=("time", "S"),
        indices=np.arange(1),
    )
    run_chunk_in_worker(spec, chunk)
    result = run_chunk_in_worker(other, chunk)
    assert result.values[0, 0, 1] == pytest.approx(0.5)


def test_the_spec_of_a_model_loads_it_with_the_settings(
    model: RoadrunnerSBMLModel,
) -> None:
    settings: dict[str, float | AbsoluteTolerance] = {
        "absolute_tolerance": AbsoluteTolerance(
            amount=1e-12, concentration=1e-9, ids={"A": 1e-14}
        ),
        "relative_tolerance": 1e-8,
    }
    model.set_integrator_settings(**settings)
    # settings of the model which no simulator sets
    model.set_integrator_settings(
        stiff=False, maximum_num_steps=50, initial_time_step=1e-3
    )
    spec = ModelSpec.of(model)
    assert spec.integrator == "cvode"
    loaded = spec.load()
    np.testing.assert_allclose(
        loaded.r_loaded.getIntegrator().getAbsoluteToleranceVector(),
        model.r_loaded.getIntegrator().getAbsoluteToleranceVector(),
    )
    integrator = loaded.r_loaded.getIntegrator()
    assert integrator.getValue("relative_tolerance") == 1e-8
    assert integrator.getValue("stiff") is False
    assert integrator.getValue("maximum_num_steps") == 50
    assert integrator.getValue("initial_time_step") == 1e-3
    # the loaded model has the state of the model of the parent
    assert ModelSpec.of(loaded) == spec
    assert ModelSpec.of(model).key == spec.key
    model.set_integrator_settings(relative_tolerance=1e-6)
    assert ModelSpec.of(model).key != spec.key
    assert pickle.loads(pickle.dumps(spec)) == spec


def test_a_worker_loads_a_model_once(
    model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    spec = ModelSpec.of(model)
    loads: list[str] = []
    load = ModelSpec.load

    def counting(self: ModelSpec) -> RoadrunnerSBMLModel:
        loads.append(self.key)
        return load(self)

    monkeypatch.setattr(ModelSpec, "load", counting)
    chunk = _chunk(
        model, Simulation(end=1, steps=2), values={"k1": np.array([0.1, 0.2])}
    )
    first = run_chunk_in_worker(spec, chunk)
    second = run_chunk_in_worker(spec, chunk)
    np.testing.assert_array_equal(first.values, second.values)
    assert loads == [spec.key]


def test_a_change_of_a_dimension_wins_at_its_time(model: RoadrunnerSBMLModel) -> None:
    chunk = _chunk(
        model,
        Simulation(end=2, steps=4),
        values={"k1": np.array([0.1, 0.1])},
        timed={1.0: {"k1": np.array([3.0, 0.1])}},
    )
    result = run_chunk(chunk, model)
    simulation = Simulation(
        end=2, steps=4, preinit_changes={"k1": 0.1}, changes=[Change(1.0, {"k1": 3.0})]
    )
    plan = compile_simulation(simulation, model.symbols, model.uinfo)
    np.testing.assert_allclose(
        result.values[0], execute(plan, model, SEL).values, rtol=1e-12
    )
    assert result.values[0, -1, 3] == pytest.approx(3.0)
    assert result.values[1, -1, 3] == pytest.approx(0.1)


def fails_for_middle_k(time: np.ndarray, values: dict[str, np.ndarray]) -> float:
    if 0.3 < values["k"][0] < 1.0:
        raise ValueError("k is in the middle")
    return float(values["k"][0])


def _observed_chunk(
    model: RoadrunnerSBMLModel,
    observables: list[Custom | Formula],
    values: dict[str, np.ndarray],
    on_error: OnError,
    indices: np.ndarray,
    simulation: Simulation,
) -> Chunk:
    return Chunk(
        indices=indices,
        plan=compile_simulation(simulation, model.symbols, model.uinfo),
        model=0,
        graph=compile_observables(observables, model),
        values=values,
        timed={},
        time=None,
        on_error=on_error,
    )


def _large_k1(model: RoadrunnerSBMLModel, on_error: OnError) -> Chunk:
    return _observed_chunk(
        model,
        [
            Custom("bad", fails_for_large_k1, "dimensionless", symbols=["k1"]),
            Formula("a", "[A]"),
        ],
        {"k1": np.array([0.5, 2.0, 0.7])},
        on_error,
        np.arange(10, 13),
        Simulation(end=2, steps=4),
    )


def test_a_custom_which_fails_flags_only_its_point(model: RoadrunnerSBMLModel) -> None:
    result = run_chunk(_large_k1(model, "flag"), model)
    assert result.status.tolist() == [0, 1, 0]
    assert [index for index, _ in result.errors] == [11]
    assert "k1 is too large" in result.errors[0][1]
    assert np.isnan(result.values[1]).all()
    assert np.isnan(result.scalars[1]).all()
    for k in (0, 2):
        assert np.isfinite(result.values[k]).all()
        assert result.scalars[k].tolist() == [0.0]


def test_the_points_which_ran_do_not_depend_on_a_failing_one(
    model: RoadrunnerSBMLModel,
) -> None:
    both = run_chunk(_large_k1(model, "flag"), model)
    chunk = _large_k1(model, "flag")
    chunk.values["k1"] = np.array([0.5, 0.7, 0.7])
    healthy = run_chunk(chunk, model)
    np.testing.assert_array_equal(both.values[0], healthy.values[0])
    np.testing.assert_array_equal(both.values[2], healthy.values[2])


def test_a_custom_which_fails_raises_its_point(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ScanPointError, match="k1 is too large") as info:
        run_chunk(_large_k1(model, "raise"), model)
    assert info.value.index == 11


def _ordered(blowup: RoadrunnerSBMLModel, k: list[float], first: int) -> Chunk:
    return _observed_chunk(
        blowup,
        [Custom("bad", fails_for_middle_k, "dimensionless", symbols=["k"])],
        {"k": np.array(k)},
        "raise",
        np.arange(first, first + len(k)),
        Simulation(end=1, steps=4),
    )


@posix
def test_the_first_point_to_fail_is_raised_whatever_the_chunking(
    capfd: pytest.CaptureFixture[str],
) -> None:
    # point 1 fails its observable, point 2 its simulation
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    k = [0.1, 0.5, 2.0]
    with pytest.raises(ScanPointError, match="middle") as one:
        run_chunk(_ordered(blowup, k, 0), blowup)
    assert one.value.index == 1
    with pytest.raises(ScanPointError, match="middle") as head:
        run_chunk(_ordered(blowup, k[:2], 0), blowup)
    assert head.value.index == 1
    with pytest.raises(ScanPointError) as tail:
        run_chunk(_ordered(blowup, k[2:], 2), blowup)
    assert tail.value.index == 2
    ctypes.CDLL(None).fflush(None)
    capfd.readouterr()


def test_the_scalars_of_a_chunk_are_filled(model: RoadrunnerSBMLModel) -> None:
    chunk = _observed_chunk(
        model,
        [
            Formula("a", "[A]"),
            Formula("amax", "max(a)"),
            Formula("alast", "at(a, 2)"),
        ],
        {"k1": np.array([0.5, 2.0])},
        "raise",
        np.arange(2),
        Simulation(end=2, steps=4),
    )
    result = run_chunk(chunk, model)
    assert chunk.graph.scalars == ("amax", "alast")
    assert result.scalars.shape == (2, 2)
    a = result.values[:, :, 1]
    np.testing.assert_allclose(result.scalars[:, 0], a.max(axis=1))
    np.testing.assert_allclose(result.scalars[:, 1], a[:, -1])


def _large_cmax(time: np.ndarray, values: dict[str, Any]) -> float:
    """A custom observable which fails for a point whose concentration peaks high."""
    if values["[C]"].max() > 100.0:
        raise ValueError("cmax is too large")
    return 0.0


def _pk_points(doses: list[float]) -> tuple[RoadrunnerSBMLModel, list[Plan], list[Any]]:
    simulator = Simulator()
    model = simulator.load(sbml_pk())
    simulation = Simulation(end=24, steps=48, changes=[Change(0, {"PODOSE": 0.0})])
    plan = simulator.compile(model, simulation)
    scan = Scan(simulation, [Dimension("dose", values={"PODOSE": np.array(doses)})])
    plans = scan_point_plans(scan, model, plan, np.arange(len(doses))[:, None])
    solutions = [execute(p, model, ["time", "[C]"]).values for p in plans]
    return model, plans, solutions


def test_observe_evaluates_the_graph_on_the_solutions() -> None:
    model, plans, solutions = _pk_points([100.0, 400.0])
    graph = compile_observables(
        [Formula("cmax", "max([C])")], model, keep=["cmax", "[C]"], plans=plans
    )
    time, outputs, failures = observe(graph, ("time", "[C]"), solutions, plans)
    assert failures == []
    assert time.shape == (2, solutions[0].shape[0])
    assert outputs["[C]"].shape == (2, solutions[0].shape[0])
    np.testing.assert_allclose(
        outputs["cmax"], [solution[:, 1].max() for solution in solutions]
    )


def test_observe_reports_a_failing_point_and_keeps_the_others() -> None:
    model, plans, solutions = _pk_points([100.0, 4000.0, 200.0])
    graph = compile_observables(
        [
            Custom("bad", _large_cmax, "dimensionless", symbols=["[C]"]),
            Formula("cmax", "max([C])"),
        ],
        model,
        keep=["bad", "cmax"],
        plans=plans,
    )
    time, outputs, failures = observe(graph, ("time", "[C]"), solutions, plans)
    assert [position for position, _, _ in failures] == [1]
    assert "too large" in failures[0][1]
    assert time.shape[0] == 2
    np.testing.assert_allclose(
        outputs["cmax"], [solutions[0][:, 1].max(), solutions[2][:, 1].max()]
    )
