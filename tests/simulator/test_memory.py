"""Memory retained by scan outputs."""

import weakref

import numpy as np
import pytest

import sbmlsim.simulator.simulator as simulator_module
import sbmlsim.simulator.worker as worker_module
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Custom, Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.worker import Chunk, ChunkResult
from tests.simulator.models import fails_for_large_k1, sbml


@pytest.mark.parametrize("grid", [None, [0.0, 0.5, 1.0]])
@pytest.mark.parametrize("failed", [False, True])
def test_scalar_scan_discards_time_buffers(
    monkeypatch: pytest.MonkeyPatch, grid: list[float] | None, failed: bool
) -> None:
    chunks: list[ChunkResult] = []
    run_chunk = worker_module.run_chunk

    def capture(chunk: Chunk, model: RoadrunnerSBMLModel) -> ChunkResult:
        result = run_chunk(chunk, model)
        chunks.append(result)
        return result

    monkeypatch.setattr(simulator_module, "run_chunk", capture)

    def unwanted_interpolation(*args: object, **kwargs: object) -> np.ndarray:
        pytest.fail("A scalar scan must not interpolate discarded timecourses.")

    monkeypatch.setattr(worker_module, "_on_grid", unwanted_interpolation)
    values = [0.1, 2.0 if failed else 0.2]
    scan = Scan(
        Simulation(end=1, steps=1000),
        [Dimension("rate", values={"k1": np.array(values)})],
    )
    result = Simulator(n_workers=1).run(
        sbml(),
        scan,
        [
            Formula("last", "at([A], 1)"),
            Custom("check", fails_for_large_k1, "dimensionless", symbols=["k1"]),
        ],
        time=grid,
        on_error="flag",
    )
    assert "time" not in result.ds
    assert result["last"].dims == ("rate",)
    assert result["status"].values.tolist() == [0, int(failed)]
    assert np.isfinite(result["last"].values[0])
    assert np.isnan(result["last"].values[1]) == failed
    assert chunks and all(chunk.values.nbytes == 0 for chunk in chunks)


def test_all_failed_scalar_points_keep_flags_without_time() -> None:
    scan = Scan(
        Simulation(end=1, steps=1000),
        [Dimension("rate", values={"k1": np.array([2.0, 3.0])})],
    )
    result = Simulator(n_workers=1).run(
        sbml(),
        scan,
        [Custom("check", fails_for_large_k1, "dimensionless", symbols=["k1"])],
        time=[0.0, 0.5, 1.0],
        on_error="flag",
    )
    assert "time" not in result.ds
    assert result["status"].values.tolist() == [1, 1]
    assert np.isnan(result["check"].values).all()
    assert len(result.ds.attrs["errors"]) == 2


@pytest.mark.parametrize("grid", [None, [0.0, 0.25, 0.5, 1.0]])
def test_the_chunks_are_released_when_written(
    monkeypatch: pytest.MonkeyPatch, grid: list[float] | None
) -> None:
    """A run holds its result once (#411).

    A chunk on a grid is written when it arrives and released, the chunks of
    a ragged result are released while `finish` writes them.
    """
    chunks: list[weakref.ref[ChunkResult]] = []
    run_chunk = worker_module.run_chunk

    def capture(chunk: Chunk, model: RoadrunnerSBMLModel) -> ChunkResult:
        result = run_chunk(chunk, model)
        chunks.append(weakref.ref(result))
        return result

    alive: list[int] = []
    finish = simulator_module._Assembly.finish

    def counted(assembly: simulator_module._Assembly) -> np.ndarray | None:
        alive.append(sum(ref() is not None for ref in chunks))
        cube = finish(assembly)
        alive.append(sum(ref() is not None for ref in chunks))
        return cube

    monkeypatch.setattr(simulator_module, "run_chunk", capture)
    monkeypatch.setattr(simulator_module._Assembly, "finish", counted)
    # the points of a ragged result differ in their number of steps
    values = np.linspace(0.1, 20.0, 8)
    simulation = Simulation(end=1)
    scan = Scan(simulation, [Dimension("rate", values={"k1": values})])
    result = Simulator(n_workers=1).run(sbml(), scan, time=grid)

    assert len(chunks) == 4
    assert alive == ([0, 0] if grid else [4, 0])
    expected = [
        Simulator(n_workers=1).run(sbml(), simulation.with_values({"k1": k}), time=grid)
        for k in values
    ]
    tdim = "time" if grid else "_point"
    for k, single in enumerate(expected):
        point = result.ds.isel(rate=k)
        rows = single.ds.sizes[tdim]
        for name in single.ds.data_vars:
            got = point[name].values
            np.testing.assert_array_equal(got[:rows], single.ds[name].values)
            assert np.isnan(got[rows:]).all()
    if grid is None:
        lengths = {single.ds.sizes[tdim] for single in expected}
        assert len(lengths) > 1
        assert result.ds.sizes[tdim] == max(lengths)
