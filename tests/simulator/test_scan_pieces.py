"""The pieces of a scan: a run of given points and the merge of their results."""

import logging
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import ScanResult
from sbmlsim.result.scan import INDEX, MAX_ERRORS, POINT, STATUS
from sbmlsim.simulation import Dimension, Formula, Scan, Simulation
from sbmlsim.simulator import Simulator
from tests.simulator.models import sbml

#: two species, `S2` blows up at the time about `1 / kb` for `kb > 0`
PIECES = """
model pieces
  compartment c = 1
  species S1 in c; species S2 in c
  S1 = 10; S2 = 1
  k1 = 0.5; k2 = 0.1; kb = 0
  J1: S1 -> S2; k1 * S1
  J2: S2 -> ; k2 * S2
  J3: -> S2; kb * S2^2
end
"""

BLOCKS = (range(5), range(5, 12))


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    return RoadrunnerSBMLModel(source=sbml(PIECES))


@pytest.fixture
def simulator() -> Simulator:
    return Simulator(n_workers=1)


def _scan(k1: tuple[float, ...] = (0.1, 1.0, 5.0)) -> Scan:
    """Get a scan of 3 x 4 points with string labels, a quantity and a covariate."""
    return Scan(
        Simulation(end=10),
        [
            Dimension(
                "fast",
                values={"k1": list(k1)},
                labels=["slow", "medium", "fastest"],
                coordinates={"group": [1, 2, 3]},
            ),
            Dimension(
                "decay",
                values={"k2": Q([0.01, 0.1, 1.0, 10.0], "dimensionless")},
                coordinates={"weight": [1.0, 2.0, 3.0, 4.0]},
            ),
        ],
    )


def _parts(
    simulator: Simulator, model: RoadrunnerSBMLModel, scan: Scan, **kwargs: Any
) -> list[ScanResult]:
    return [simulator.run(model, scan, points=points, **kwargs) for points in BLOCKS]


def test_blocks_combine_to_the_scan_on_a_grid(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    scan = _scan()
    time = np.linspace(0, 10, 11)
    full = simulator.run(model, scan, time=time)
    parts = _parts(simulator, model, scan, time=time)
    assert parts[0].ds.sizes[INDEX] == 5
    assert parts[0].ds.attrs["block"] == {
        "start": 0,
        "stop": 5,
        "shape": [3, 4],
        "dims": ["fast", "decay"],
        "coords": {
            "fast": "fast",
            "fast.k1": "fast",
            "group": "fast",
            "decay": "decay",
            "decay.k2": "decay",
            "weight": "decay",
        },
    }
    np.testing.assert_array_equal(parts[1].ds[INDEX], np.arange(5, 12))
    # in reverse order: `combine` sorts the blocks
    xr.testing.assert_identical(ScanResult.combine(parts[::-1]).ds, full.ds)


def test_blocks_combine_to_the_scan_ragged(
    simulator: Simulator, model: RoadrunnerSBMLModel, tmp_path: Path
) -> None:
    scan = _scan()
    full = simulator.run(model, scan)
    assert POINT in full.ds.dims
    parts = _parts(simulator, model, scan)
    # the blocks pad to their own longest timecourse
    assert parts[0].ds.sizes[POINT] != parts[1].ds.sizes[POINT]
    read = []
    for k, part in enumerate(parts):
        path = tmp_path / f"block{k}.nc"
        part.to_netcdf(path)
        read.append(ScanResult.from_netcdf(path))
    assert read[1].ds.attrs["block"] == parts[1].ds.attrs["block"]
    xr.testing.assert_identical(ScanResult.combine(read).ds, full.ds)
    xr.testing.assert_identical(ScanResult.combine(parts).ds, full.ds)


def test_blocks_combine_with_values_per_simulation(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    scan = _scan()
    observables = [Formula("s1", "S1"), Formula("s1max", "max(S1)")]
    full = simulator.run(model, scan, observables)
    parts = _parts(simulator, model, scan, observables=observables)
    assert parts[0].ds["s1max"].dims == (INDEX,)
    xr.testing.assert_identical(ScanResult.combine(parts).ds, full.ds)


def test_blocks_combine_with_failed_points(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    # `kb = 1` blows up before the end, at the points 2, 6 and 10
    scan = Scan(
        Simulation(end=10),
        [
            Dimension("fast", values={"k1": [0.1, 1.0, 5.0]}),
            Dimension(
                "decay", values={"k2": [0.01, 0.1, 1.0, 10.0], "kb": [0, 0, 1, 0]}
            ),
        ],
    )
    time = np.linspace(0, 10, 11)
    full = simulator.run(model, scan, time=time, on_error="flag")
    parts = _parts(simulator, model, scan, time=time, on_error="flag")
    assert int(full.ds[STATUS].sum()) == 3
    assert len(parts[0].ds.attrs["errors"]) == 1
    combined = ScanResult.combine(parts)
    xr.testing.assert_identical(combined.ds, full.ds)
    assert combined.ds.attrs["errors"] == full.ds.attrs["errors"]


def test_combine_keeps_at_most_the_errors_of_a_result(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    n = MAX_ERRORS + 4
    scan = Scan(Simulation(end=10), [Dimension("blow", values={"kb": [1.0] * n})])
    full = simulator.run(model, scan, on_error="flag")
    parts = [
        simulator.run(model, scan, points=points, on_error="flag")
        for points in (range(3), range(3, n))
    ]
    combined = ScanResult.combine(parts)
    assert len(combined.ds.attrs["errors"]) == MAX_ERRORS
    xr.testing.assert_identical(combined.ds, full.ds)


def test_blocks_combine_with_a_dimension_of_simulations(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    scan = Scan(
        Simulation(end=10),
        [
            Dimension("fast", values={"k1": [0.1, 1.0, 5.0]}),
            Dimension(
                "sim",
                simulations={
                    "short": Simulation(end=2),
                    "long": Simulation(end=10),
                },
            ),
        ],
    )
    full = simulator.run(model, scan)
    parts = [
        simulator.run(model, scan, points=points)
        for points in (range(1), range(1, 4), range(4, 6))
    ]
    xr.testing.assert_identical(ScanResult.combine(parts).ds, full.ds)


def test_a_scan_without_dimensions_is_one_block(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    simulation = Simulation(end=10)
    full = simulator.run(model, simulation)
    part = simulator.run(model, simulation, points=range(1))
    assert part.ds.attrs["block"]["shape"] == []
    xr.testing.assert_identical(ScanResult.combine([part]).ds, full.ds)
    with pytest.raises(ValueError, match="range"):
        simulator.run(model, simulation, points=range(2))


def test_blocks_in_a_pool_equal_serial_blocks(model: RoadrunnerSBMLModel) -> None:
    scan = _scan()
    points = range(2, 9)
    serial = Simulator(n_workers=1).run(model, scan, points=points)
    pooled = Simulator(n_workers=2).run(model, scan, points=points)
    xr.testing.assert_identical(pooled.ds, serial.ds)


def test_combine_names_a_gap(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    scan = _scan()
    parts = [
        simulator.run(model, scan, points=points) for points in (range(5), range(6, 12))
    ]
    with pytest.raises(ValueError, match=r"miss.*5"):
        ScanResult.combine(parts)


def test_combine_names_missing_points_at_the_end(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    parts = [simulator.run(model, _scan(), points=range(5))]
    with pytest.raises(ValueError, match=r"miss.*5.*12"):
        ScanResult.combine(parts)


def test_combine_refuses_overlap(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    scan = _scan()
    parts = [
        simulator.run(model, scan, points=points) for points in (range(6), range(5, 12))
    ]
    with pytest.raises(ValueError, match=r"more than once.*5"):
        ScanResult.combine(parts)


def test_combine_refuses_blocks_of_other_scans(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    parts = [
        simulator.run(model, _scan(), points=range(5)),
        simulator.run(model, _scan(k1=(0.2, 1.0, 5.0)), points=range(5, 12)),
    ]
    with pytest.raises(ValueError, match="other scans"):
        ScanResult.combine(parts)


def test_combine_refuses_a_whole_result(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    full = simulator.run(model, _scan())
    with pytest.raises(ValueError, match="block"):
        ScanResult.combine([full])
    with pytest.raises(ValueError, match="no blocks"):
        ScanResult.combine([])


def test_points_outside_the_scan_raise(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    with pytest.raises(ValueError, match="range"):
        simulator.run(model, _scan(), points=range(10, 13))


def test_empty_points_raise(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="empty"):
        simulator.run(model, _scan(), points=range(3, 3))


def test_points_with_a_step_raise(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    with pytest.raises(ValueError, match="step"):
        simulator.run(model, _scan(), points=range(0, 6, 2))


def test_blocks_combine_over_a_dimension_of_models(simulator: Simulator) -> None:
    models = {
        "low": RoadrunnerSBMLModel(source=sbml(PIECES)),
        "high": RoadrunnerSBMLModel(source=sbml(PIECES.replace("S1 = 10", "S1 = 20"))),
    }
    scan = Scan(
        Simulation(end=10),
        [
            Dimension("fast", values={"k1": [0.1, 1.0, 5.0]}),
            Dimension("model", models=models),
        ],
    )
    full = simulator.run(None, scan)
    parts = [
        simulator.run(None, scan, points=points)
        for points in (range(3), range(3, 4), range(4, 6))
    ]
    np.testing.assert_array_equal(parts[1].ds["model"], ["high"])
    xr.testing.assert_identical(ScanResult.combine(parts).ds, full.ds)


def test_a_block_allocates_for_its_points_only(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    # 10^7 points; their positions alone would take 240 MB
    scan = Scan(
        Simulation(end=10),
        [
            Dimension("fast", values={"k1": np.linspace(0.1, 5.0, 100)}),
            Dimension("decay", values={"k2": np.linspace(0.01, 1.0, 1000)}),
            Dimension("blow", values={"kb": np.zeros(100)}),
        ],
    )
    assert scan.size == 10**7
    points = range(4_321_000, 4_321_003)
    tracemalloc.start()
    try:
        part = simulator.run(model, scan, points=points, time=np.linspace(0, 10, 11))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 10 * 2**20
    assert part.ds.sizes[INDEX] == 3
    np.testing.assert_array_equal(
        part.ds["fast.k1"], [scan.dimensions[0].values["k1"][43]] * 3
    )


def test_a_block_names_its_points_when_points_fail(
    simulator: Simulator, model: RoadrunnerSBMLModel, caplog: pytest.LogCaptureFixture
) -> None:
    scan = Scan(Simulation(end=10), [Dimension("blow", values={"kb": [0, 1, 1, 0]})])
    with caplog.at_level(logging.WARNING):
        simulator.run(model, scan, points=range(1, 3), on_error="flag")
    assert "2 of the points 1-3 of the 4 points of the scan failed" in caplog.text


def test_a_block_has_no_summary_and_no_interpolation(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    part = simulator.run(model, _scan(), points=range(3))
    assert part.is_block
    with pytest.raises(ValueError, match="combine the blocks first"):
        part.summary()
    with pytest.raises(ValueError, match="combine the blocks first"):
        part.interpolate([0, 1])
    assert not ScanResult.combine(
        [simulator.run(model, _scan(), points=range(12))]
    ).is_block


def test_index_is_a_reserved_name(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="_index"):
        Scan(Simulation(end=1), [Dimension(INDEX, values={"k1": [1.0, 2.0]})])
    scan = Scan(
        Simulation(end=1),
        [Dimension("fast", values={"k1": [1.0, 2.0]}, coordinates={INDEX: [1, 2]})],
    )
    with pytest.raises(ValueError, match="_index"):
        Simulator(n_workers=1).run(model, scan)
    with pytest.raises(ValueError, match="_index"):
        Formula(INDEX, "S1")
