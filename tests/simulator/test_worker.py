"""A chunk of a scan runs its points on one plan and one model."""

import ctypes
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.plan import compile_simulation
from sbmlsim.simulator.worker import (
    MAX_ERRORS,
    Chunk,
    ChunkResult,
    ModelSpec,
    OnError,
    ScanPointError,
    run_chunk,
    run_chunk_in_worker,
)
from tests.simulator.models import BLOWUP, sbml

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
        selections=selections,
        values=values or {},
        timed=timed or {},
        time=time,
        on_error=on_error,
    )


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
    """Run a chunk with failing points and take the output of CVODE."""
    capfd.readouterr()
    try:
        return run_chunk(chunk, model)
    finally:
        ctypes.CDLL(None).fflush(None)
        captured = capfd.readouterr()
        assert "CVODE" in captured.out + captured.err


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
    spec = ModelSpec.of(first, {})
    path.write_text(sbml(BLOWUP.replace("species S in C = 1", "species S in C = 0.5")))
    second = RoadrunnerSBMLModel(source=path)
    other = ModelSpec.of(second, {})
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
    spec = ModelSpec.of(model, settings)
    loaded = spec.load()
    np.testing.assert_allclose(
        loaded.r_loaded.getIntegrator().getAbsoluteToleranceVector(),
        model.r_loaded.getIntegrator().getAbsoluteToleranceVector(),
    )
    assert loaded.r_loaded.getIntegrator().getValue("relative_tolerance") == 1e-8
    assert ModelSpec.of(model, settings).key == spec.key
    assert ModelSpec.of(model, {**settings, "relative_tolerance": 1e-6}).key != spec.key
    assert pickle.loads(pickle.dumps(spec)) == spec


def test_a_worker_loads_a_model_once(
    model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallel, "_CACHE", OrderedDict())
    spec = ModelSpec.of(model, {})
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
