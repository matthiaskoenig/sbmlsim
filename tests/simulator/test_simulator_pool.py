"""A scan gives the same result serially and in the pool."""

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import sbmlsim.simulator.simulator as simulator_module
from sbmlsim import parallel
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import ScanError, Simulator
from sbmlsim.simulator.worker import ModelSpec
from tests.simulator.models import BLOWUP, PROBE, sbml

SEL = ["time", "[A]", "[B]", "X", "k1"]


def _model() -> RoadrunnerSBMLModel:
    model = RoadrunnerSBMLModel(source=sbml())
    model.set_selections(SEL)
    return model


def _scan() -> Scan:
    """Twelve points with the steps of the integrator, a change and a value at a time."""
    return Scan(
        Simulation(end=2, changes=[Change(1.0, {"[A]": 2.0})]),
        [
            Dimension("a", values={"b0": [0.5, 1.0, 2.0]}),
            Dimension("b", values={"k2": [0.1, 0.2, 0.4, 0.8]}, at=0.5),
        ],
    )


def _c_output(capfd: pytest.CaptureFixture[str]) -> str:
    """Take the output of C of the calling process, see test_simulator."""
    if sys.platform != "win32":
        ctypes.CDLL(None).fflush(None)
    captured = capfd.readouterr()
    return captured.out + captured.err


@pytest.mark.parametrize("n_workers", [1, 2, 4])
@pytest.mark.parametrize("max_chunk", [1, 1000])
def test_the_result_does_not_depend_on_the_workers(
    n_workers: int, max_chunk: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    serial = Simulator(n_workers=1).run(_model(), _scan())
    monkeypatch.setattr(simulator_module, "MAX_CHUNK", max_chunk)
    pooled = Simulator(n_workers=n_workers).run(_model(), _scan())
    xr.testing.assert_identical(pooled.ds, serial.ds)


def test_a_scan_over_models_in_the_pool(tmp_path: Path) -> None:
    slow = tmp_path / "slow.xml"
    slow.write_text(sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")))
    scan = Scan(
        Simulation(end=1, steps=4),
        [
            Dimension("model", models={"fast": _model(), "slow": slow}),
            Dimension("d", values={"b0": [1.0, 2.0]}),
        ],
    )
    serial = Simulator(n_workers=1).run(None, scan)
    pooled = Simulator(n_workers=2).run(None, scan)
    xr.testing.assert_identical(pooled.ds, serial.ds)
    # the models differ, the pool did not run one model for both
    a = pooled["[A]"].values
    assert not np.allclose(a[0], a[1])


def test_a_failed_point_is_flagged_in_the_pool(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(simulator_module, "MAX_CHUNK", 1)
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(
        Simulation(end=1, steps=4),
        [Dimension("rate", values={"k": [0.1, 2.0, 0.2, 0.3]})],
    )
    try:
        serial = Simulator(n_workers=1).run(blowup, scan, on_error="flag")
    finally:
        _c_output(capfd)
    caplog.clear()
    pooled = Simulator(n_workers=2).run(blowup, scan, on_error="flag")
    assert pooled["status"].values.tolist() == [0, 1, 0, 0]
    assert np.isnan(pooled["S"].values[1]).all()
    assert np.isfinite(pooled["S"].values[[0, 2, 3]]).all()
    xr.testing.assert_identical(pooled.ds, serial.ds)
    assert pooled.ds.attrs["errors"][0].startswith("rate=1, k=2.0: RuntimeError")
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "1 of 4 points of the scan failed" in warnings[0].getMessage()
    # the workers print no messages of SUNDIALS, and the parent integrated nothing
    assert "cvodes" not in _c_output(capfd)


def test_a_failed_point_raises_in_the_pool() -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(
        Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0]})]
    )
    with pytest.raises(ScanError, match=r"The point rate=1, k=2\.0 of the scan"):
        Simulator(n_workers=2).run(blowup, scan)
    # the pool is fine after a point failed and runs the next scan
    assert 2 in parallel._POOLS
    res = Simulator(n_workers=2).run(blowup, scan, on_error="flag")
    assert res["status"].values.tolist() == [0, 1]


def test_the_first_failed_point_raises_in_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunks of one point finish in any order, the error is the first in the scan."""
    monkeypatch.setattr(simulator_module, "MAX_CHUNK", 1)
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    rates = [0.1, 0.2, 3.0, 0.3, 2.0, 4.0, 0.4, 5.0]
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": rates})])
    for _ in range(3):
        with pytest.raises(ScanError, match=r"The point rate=2, k=3\.0 of the scan"):
            Simulator(n_workers=4).run(blowup, scan)


def test_the_errors_are_in_the_order_of_the_scan_in_the_pool(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(simulator_module, "MAX_CHUNK", 1)
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    rates = [0.1, 0.2, 3.0, 0.3, 2.0, 4.0, 0.4, 5.0]
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": rates})])
    try:
        serial = Simulator(n_workers=1).run(blowup, scan, on_error="flag")
    finally:
        _c_output(capfd)
    pooled = Simulator(n_workers=4).run(blowup, scan, on_error="flag")
    assert pooled["status"].values.tolist() == [0, 0, 1, 0, 1, 1, 0, 1]
    labels = [error.split(",")[0] for error in pooled.ds.attrs["errors"]]
    assert labels == ["rate=2", "rate=4", "rate=5", "rate=7"]
    xr.testing.assert_identical(pooled.ds, serial.ds)


def _worker_tolerances(spec: ModelSpec) -> list[float]:
    """Get the absolute tolerances of the model a worker keeps."""
    model = parallel.worker_cache(("model", spec.key), spec.load)
    vector = model.r_loaded.getIntegrator().getAbsoluteToleranceVector()
    return [float(v) for v in vector]


def test_the_tolerances_reach_the_workers() -> None:
    tolerance = AbsoluteTolerance(
        amount=1e-12, concentration=1e-9, other=1e-8, ids={"A": 1e-14}
    )
    simulator = Simulator(n_workers=2, absolute_tolerance=tolerance)
    model = simulator.load(_model())
    expected = [
        float(v) for v in model.r_loaded.getIntegrator().getAbsoluteToleranceVector()
    ]
    # the override by id is part of the vector which the worker gets
    assert expected[model.state_ids().index("A")] == pytest.approx(1e-14 * 2)
    spec = ModelSpec.of(model, simulator.integrator_settings)
    assert parallel.pool(2).submit(_worker_tolerances, spec).result() == expected


def test_every_model_of_a_dimension_gets_the_tolerances(tmp_path: Path) -> None:
    """Each model of a dimension of models has its own vector, the same in a worker."""
    slow = tmp_path / "slow.xml"
    slow.write_text(sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")))
    tolerance = AbsoluteTolerance(amount=1e-12, concentration=1e-9, ids={"A": 1e-14})
    simulator = Simulator(n_workers=2, absolute_tolerance=tolerance)
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("model", models={"fast": _model(), "slow": slow})],
    )
    models, _ = simulator._models(None, scan)
    for model in models:
        assert model.absolute_tolerance == tolerance
        ids = model.state_ids()
        vector = [
            float(v)
            for v in model.r_loaded.getIntegrator().getAbsoluteToleranceVector()
        ]
        spec = ModelSpec.of(model, simulator.integrator_settings)
        assert parallel.pool(2).submit(_worker_tolerances, spec).result() == vector
        assert len(ids) == len(vector)


def test_a_kept_pool_runs_other_settings_with_them() -> None:
    """The workers keep their models per settings, a second run does not reuse them."""
    results = []
    for settings in ({}, {"absolute_tolerance": 1e-4, "relative_tolerance": 1e-4}):
        serial = Simulator(n_workers=1, **settings).run(_model(), _scan())
        pooled = Simulator(n_workers=2, **settings).run(_model(), _scan())
        xr.testing.assert_identical(pooled.ds, serial.ds)
        results.append(pooled)
    assert not results[0]["[A]"].equals(results[1]["[A]"])


def test_an_unknown_setting_raises_before_a_pool_starts() -> None:
    with pytest.raises(ValueError, match="no settings"):
        Simulator(n_workers=2, nope=1.0).run(_model(), _scan())
    assert parallel._POOLS == {}


def test_a_late_change_agrees_in_the_pool_and_does_not_warn(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The serial run prints no "t + h = t", the workers print nothing at all."""
    scan = Scan(
        Simulation(end=2e5, changes=[Change(1.5e5, {"[A]": 2.0})], steps=20),
        [Dimension("d", values={"k2": [0.1, 0.2, 0.4, 0.8]})],
    )
    _c_output(capfd)
    serial = Simulator(n_workers=1).run(_model(), scan)
    assert "t + h = t" not in _c_output(capfd)
    pooled = Simulator(n_workers=4).run(_model(), scan)
    xr.testing.assert_identical(pooled.ds, serial.ds)


def test_an_interrupted_run_stops_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Any] = []

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        # Ctrl-C while the run waits for its chunks
        executor = parallel._POOLS[2]
        started.append((executor, list(executor._processes.values())))
        raise KeyboardInterrupt

    monkeypatch.setattr(simulator_module, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        Simulator(n_workers=2).run(_model(), _scan())
    # the run had a pool, Ctrl-C stopped its workers and dropped it
    executor, processes = started[0]
    assert processes
    assert not any(process.is_alive() for process in processes)
    assert parallel._POOLS == {}
    monkeypatch.undo()
    serial = Simulator(n_workers=1).run(_model(), _scan())
    xr.testing.assert_identical(
        Simulator(n_workers=2).run(_model(), _scan()).ds, serial.ds
    )
    # the next run started a new pool
    assert parallel._POOLS[2] is not executor


def test_none_runs_a_small_scan_serially(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_pool(n_workers: int) -> Any:
        raise AssertionError("a scan of 12 points started a pool")

    monkeypatch.setattr(parallel, "pool", no_pool)
    res = Simulator().run(_model(), _scan())
    assert res["[A]"].shape[:2] == (3, 4)
    assert np.isfinite(res["[A]"].values[..., 0]).all()


def test_none_runs_a_scan_from_the_threshold_in_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial = Simulator(n_workers=1).run(_model(), _scan())
    monkeypatch.setattr(parallel, "POOL_THRESHOLD", 12)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 2)
    pooled = Simulator().run(_model(), _scan())
    assert list(parallel._POOLS) == [2]
    xr.testing.assert_identical(pooled.ds, serial.ds)
