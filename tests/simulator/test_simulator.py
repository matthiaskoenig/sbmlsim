"""The simulator runs a scan and answers with a ScanResult, here serially."""

import ctypes
import gc
import os
import sys
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import roadrunner

import sbmlsim.simulator.simulator as simulator_module
import sbmlsim.simulator.worker as worker_module
from sbmlsim import Q
from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.result import ScanResult, TimecourseResult
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import ScanError, Simulator
from sbmlsim.simulator.worker import MAX_ERRORS
from tests.simulator.models import BLOWUP, PROBE, TOLERANCE_PROBE, sbml, sbml_minutes

SEL = ["time", "[A]", "[B]", "X", "k1"]


@pytest.fixture
def model() -> RoadrunnerSBMLModel:
    model = RoadrunnerSBMLModel(source=sbml())
    model.set_selections(SEL)
    return model


@pytest.fixture
def simulator() -> Simulator:
    return Simulator(n_workers=1)


def _c_output(capfd: pytest.CaptureFixture[str]) -> str:
    """Take the output of C of the points which failed, see test_worker.

    The streams of C are flushed with POSIX ctypes; on Windows the output of
    C which is still buffered is not taken.
    """
    if sys.platform != "win32":
        ctypes.CDLL(None).fflush(None)
    captured = capfd.readouterr()
    return captured.out + captured.err


def _assert_units(res: ScanResult) -> None:
    """Check that every variable and coordinate has a unit."""
    missing = sorted(str(name) for name in res.ds.variables if name not in res.units)
    assert not missing


def test_the_selections_of_a_model() -> None:
    model = RoadrunnerSBMLModel(source=sbml())
    assert model.selections is not None
    every = list(model.selections)
    model.set_selections(["time", "X"])
    assert model.selections == ["time", "X"]
    assert list(model.r_loaded.timeCourseSelections) == ["time", "X"]
    model.set_selections(None)
    assert model.selections == every


@pytest.mark.parametrize(
    "selections", [["[A]", "time", "X"], ["[A]", "X", "time"], ["time", "[A]", "X"]]
)
def test_the_time_is_selected_once(
    simulator: Simulator,
    model: RoadrunnerSBMLModel,
    selections: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The time is the first column of a run, wherever the model selects it.

    The selections of an experiment are sorted, which puts `time` last.
    """
    model.set_selections(selections)
    columns: list[tuple[str, ...]] = []
    run_chunk = simulator_module.run_chunk

    def recording(chunk: Any, loaded: RoadrunnerSBMLModel) -> Any:
        result = run_chunk(chunk, loaded)
        columns.append(chunk.selections)
        assert result.values.shape[2] == 3
        return result

    monkeypatch.setattr(simulator_module, "run_chunk", recording)
    res = simulator.run(model, Simulation(end=1, steps=2))
    assert columns == [("time", "[A]", "X")]
    assert res.variables == ["[A]", "X"]


def test_simulate_is_one_timecourse(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    result = simulator.simulate(model, Simulation(end=1, steps=2))
    assert isinstance(result, TimecourseResult)
    assert result.columns == tuple(SEL)


def test_a_simulation_is_a_scan_without_dimensions(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    res = simulator.run(model, Simulation(end=1, steps=10))
    assert isinstance(res, ScanResult)
    assert res.dims == ()
    assert res["[B]"].dims == ("time",)
    np.testing.assert_allclose(res["time"].values, np.linspace(0, 1, 11))
    assert res.variables == ["[A]", "[B]", "X", "k1"]
    assert (
        res.ds.attrs["integrator_settings"]["absolute_tolerance"]
        == AbsoluteTolerance().to_dict()
    )
    assert res.ds.attrs["integrator_settings"]["relative_tolerance"] == 1e-10
    assert res.ds.attrs["dims"] == []
    assert res.ds.attrs["scan"]["simulation"]["end"] == 1


def test_the_steps_of_the_integrator_are_ragged(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    res = simulator.run(model, Simulation(end=1))
    assert res.ragged
    assert res["time"].dims == ("_point",)
    expected = simulator.simulate(model, Simulation(end=1))
    np.testing.assert_allclose(res["[A]"].values, expected["[A]"], rtol=1e-12)


def test_a_dimension_of_values(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    scan = Scan(
        Simulation(end=1, steps=10), [Dimension("d", values={"b0": [0.0, 2.0]})]
    )
    res = simulator.run(model, scan)
    assert res["[B]"].dims == ("d", "time")
    np.testing.assert_allclose(res["[B]"].values[:, 0], [0.0, 2.0])
    # a changed target is a coordinate
    assert res["b0"].dims == ("d",)
    assert res["b0"].values.tolist() == [0.0, 2.0]
    assert res.units["b0"] == ""
    assert res["d"].values.tolist() == [0, 1]
    # labels carry no unit
    assert res.units["d"] == ""
    _assert_units(res)
    for k, b0 in enumerate([0.0, 2.0]):
        expected = simulator.simulate(
            model, Simulation(end=1, steps=10, preinit_changes={"b0": b0})
        )
        np.testing.assert_allclose(res["[B]"].values[k], expected["[B]"], rtol=1e-12)


def test_a_quantity_is_a_coordinate_in_its_unit() -> None:
    minutes = RoadrunnerSBMLModel(source=sbml_minutes())
    minutes.set_selections(["time", "X"])
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("dose", values={"f": Q([1.0, 2.0], "g")})],
    )
    res = Simulator(n_workers=1).run(minutes, scan)
    assert res["f"].values.tolist() == [1.0, 2.0]
    assert res.units["f"] == "gram"
    assert res.units["time"] == "min"
    # X = 3 * pinit = 6 * f, f in mg in the model
    np.testing.assert_allclose(res["X"].values[:, 0], [6000.0, 12000.0])


def test_a_change_at_a_time(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    scan = Scan(
        Simulation(end=2, steps=4), [Dimension("d", values={"k1": [0.1, 3.0]}, at=1.0)]
    )
    res = simulator.run(model, scan)
    for k, k1 in enumerate([0.1, 3.0]):
        expected = simulator.simulate(
            model, Simulation(end=2, steps=4, changes=[Change(1.0, {"k1": k1})])
        )
        np.testing.assert_allclose(res["[A]"].values[k], expected["[A]"], rtol=1e-12)


def test_a_change_at_a_time_wins_over_a_value(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    scan = Scan(
        Simulation(end=2, steps=4),
        [
            Dimension("a", values={"k2": [0.1, 0.2]}),
            Dimension("b", values={"k2": [3.0]}, at=1.0),
        ],
    )
    res = simulator.run(model, scan)
    for k, k2 in enumerate([0.1, 0.2]):
        expected = simulator.simulate(
            model,
            Simulation(
                end=2,
                steps=4,
                preinit_changes={"k2": k2},
                changes=[Change(1.0, {"k2": 3.0})],
            ),
        )
        np.testing.assert_allclose(res["[A]"].values[k, 0], expected["[A]"], rtol=1e-12)
    # the coordinate of a target is the one of the first dimension which changes it
    assert res["k2"].dims == ("a",)


def test_the_points_are_in_c_order(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    scan = Scan(
        Simulation(end=1, steps=2),
        [
            Dimension("a", values={"b0": [1.0, 2.0]}),
            Dimension("b", values={"k2": [0.1, 0.2, 0.3]}),
        ],
    )
    res = simulator.run(model, scan)
    assert res["[B]"].shape == (2, 3, 3)
    expected = simulator.simulate(
        model, Simulation(end=1, steps=2, preinit_changes={"b0": 2.0, "k2": 0.3})
    )
    np.testing.assert_allclose(res["[B]"].values[1, 2], expected["[B]"], rtol=1e-12)
    assert res["k2"].values.tolist() == [0.1, 0.2, 0.3]


def test_a_dimension_of_simulations(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    simulations = {
        "short": Simulation(end=1, steps=2),
        "long": Simulation(end=2, steps=4),
    }
    scan = Scan(Simulation(end=1), [Dimension("sim", simulations=simulations)])
    res = simulator.run(model, scan)
    assert res.ragged
    short = res["time"].sel(sim="short").values
    assert short[:3].tolist() == [0.0, 0.5, 1.0]
    assert np.isnan(short[3:]).all()
    assert res["time"].sel(sim="long").values.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert res.units["sim"] == ""
    _assert_units(res)


def test_simulations_with_one_output_share_a_grid(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    simulations = {
        "a": Simulation(end=1, steps=2),
        "b": Simulation(end=1, steps=2, preinit_changes={"b0": 2.0}),
    }
    res = simulator.run(
        model, Scan(Simulation(end=1), [Dimension("sim", simulations=simulations)])
    )
    assert not res.ragged
    assert res["[B]"].sel(sim="b").values[0] == pytest.approx(2.0)


def test_a_dimension_of_models(
    simulator: Simulator, model: RoadrunnerSBMLModel, tmp_path: Path
) -> None:
    slow = tmp_path / "slow.xml"
    slow.write_text(sbml(PROBE.replace("k1 = 0.8", "k1 = 0.1")))
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("model", models={"fast": model, "slow": slow})],
    )
    res = simulator.run(None, scan)
    assert res["[A]"].dims == ("model", "time")
    assert res["k1"].sel(model="slow").values[0] == pytest.approx(0.1)
    assert res["k1"].sel(model="fast").values[0] == pytest.approx(0.8)
    assert res.units["model"] == ""
    _assert_units(res)
    with pytest.raises(ValueError, match="replaces the model"):
        simulator.run(model, scan)
    with pytest.raises(ValueError, match="needs a model"):
        simulator.run(None, Simulation(end=1))


def test_every_model_of_a_dimension_has_its_own_tolerances(
    model: RoadrunnerSBMLModel,
) -> None:
    big = RoadrunnerSBMLModel(source=sbml(PROBE.replace("C = 2", "C = 4")))
    big.set_selections(SEL)
    tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-8, other=1e-7)
    simulator = Simulator(n_workers=1, absolute_tolerance=tolerance)
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("model", models={"probe": model, "big": big})],
    )
    simulator.run(None, scan)
    # A and B are concentration species in C, X an amount species
    for loaded, volume in ((model, 2.0), (big, 4.0)):
        expected = {"A": 1e-8 * volume, "B": 1e-8 * volume, "X": 1e-9}
        assert _vector(loaded) == pytest.approx(expected, rel=1e-12, abs=0)


def test_a_model_without_a_selection_is_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("model", models={"probe": model, "blowup": blowup})],
    )
    with pytest.raises(ValueError, match="'blowup'"):
        simulator.run(None, scan)


def test_models_with_other_units_are_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    minutes = RoadrunnerSBMLModel(source=sbml_minutes())
    scan = Scan(
        Simulation(end=1, steps=2),
        [Dimension("model", models={"seconds": model, "minutes": minutes})],
    )
    with pytest.raises(ValueError, match="units of 'time'"):
        simulator.run(None, scan)


def test_a_target_which_is_no_symbol_is_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_chunk(*args: object, **kwargs: object) -> object:
        raise AssertionError("a point ran")

    monkeypatch.setattr(simulator_module, "run_chunk", no_chunk)
    scan = Scan(
        Simulation(end=1, steps=2), [Dimension("d", values={"nope": [1.0, 2.0]})]
    )
    with pytest.raises(ValueError, match=r"'d' does not fit the model: .*'nope'"):
        simulator.run(model, scan)
    # every model of a dimension of models has the targets
    other = RoadrunnerSBMLModel(source=sbml(PROBE.replace("k2", "k3")))
    other.set_selections(SEL)
    scan = Scan(
        Simulation(end=1, steps=2),
        [
            Dimension("model", models={"probe": model, "other": other}),
            Dimension("d", values={"k2": [1.0, 2.0]}),
        ],
    )
    with pytest.raises(ValueError, match="'d' does not fit the model 'other'"):
        simulator.run(None, scan)


def test_a_grid_of_times(simulator: Simulator, model: RoadrunnerSBMLModel) -> None:
    simulation = Simulation(end=2, changes=[Change(1.0, {"[A]": 5.0})])
    res = simulator.run(model, simulation, time=[0.0, 0.5, 1.0, 3.0])
    assert res["[A]"].dims == ("time",)
    assert res["time"].values.tolist() == [0.0, 0.5, 1.0, 3.0]
    # the value after the change, none after the end
    assert res["[A]"].sel(time=1.0).item() == pytest.approx(5.0)
    assert np.isnan(res["[A]"].sel(time=3.0).item())


def test_a_grid_time_before_the_start_is_nan(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    res = simulator.run(model, Simulation(start=0.5, end=2), time=[0.0, 0.5, 1.0])
    values = res["[A]"].values
    assert np.isnan(values[0])
    assert np.isfinite(values[1:]).all()


def test_a_grid_of_times_with_a_unit() -> None:
    minutes = RoadrunnerSBMLModel(source=sbml_minutes())
    res = Simulator(n_workers=1).run(minutes, Simulation(end=2), time=Q([0, 60], "s"))
    assert res["time"].values.tolist() == [0.0, 1.0]


def test_an_empty_grid_of_times_is_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    with pytest.raises(ValueError, match="empty"):
        simulator.run(model, Simulation(end=1), time=[])


def test_a_failed_point_raises_with_its_labels_and_values(
    simulator: Simulator, capfd: pytest.CaptureFixture[str]
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(
        Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0]})]
    )
    try:
        with pytest.raises(
            ScanError, match=r"The point rate=1, k=2\.0 of the scan failed"
        ):
            simulator.run(blowup, scan)
    finally:
        _c_output(capfd)


def test_a_failed_simulation_raises(
    simulator: Simulator, capfd: pytest.CaptureFixture[str]
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    try:
        with pytest.raises(ScanError, match="The simulation failed: RuntimeError"):
            simulator.run(
                blowup, Simulation(end=1, steps=4, preinit_changes={"k": 2.0})
            )
    finally:
        _c_output(capfd)


def _failing_scan() -> Scan:
    """Get a scan whose first failing point in scan order is in its second plan.

    `S' = k S^2` with `S0 = 1` goes to infinity at the time `1 / k`: the
    simulation to 0.3 fails for `k = 5` and the one to 1 for `k = 2` and
    `k = 5`. The points in C order are (rate, sim), the plan of `a` has the
    points 0, 2, 4 and the plan of `b` the points 1, 3, 5, which fail at 4
    and at 3 and 5.
    """
    simulations = {"a": Simulation(end=0.3, steps=3), "b": Simulation(end=1, steps=4)}
    return Scan(
        Simulation(end=1),
        [
            Dimension("rate", values={"k": [0.1, 2.0, 5.0]}),
            Dimension("sim", simulations=simulations),
        ],
    )


def test_the_first_failed_point_in_scan_order_raises(
    simulator: Simulator, capfd: pytest.CaptureFixture[str]
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    try:
        with pytest.raises(ScanError, match=r"The point rate=1, k=2\.0, sim=b of"):
            simulator.run(blowup, _failing_scan())
    finally:
        _c_output(capfd)


def test_a_failed_point_is_flagged(
    simulator: Simulator,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(
        Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0, 0.2]})]
    )
    try:
        res = simulator.run(blowup, scan, on_error="flag")
    finally:
        output = _c_output(capfd)
    assert res["status"].values.tolist() == [0, 1, 0]
    assert res["status"].dims == ("rate",)
    assert res.units["status"] == ""
    _assert_units(res)
    assert np.isnan(res["S"].values[1]).all()
    assert np.isfinite(res["S"].values[[0, 2]]).all()
    assert res.ds.attrs["errors"][0].startswith("rate=1, k=2.0: RuntimeError")
    assert "1 of 3 points of the scan failed" in caplog.text
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    # roadrunner does not repeat the error of every point which failed
    assert "CVODE Error" not in output


def test_the_errors_are_in_scan_order(
    simulator: Simulator, capfd: pytest.CaptureFixture[str]
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    try:
        res = simulator.run(blowup, _failing_scan(), on_error="flag")
    finally:
        _c_output(capfd)
    assert res["status"].values.tolist() == [[0, 0], [0, 1], [1, 1]]
    assert [e.split(":")[0] for e in res.ds.attrs["errors"]] == [
        "rate=1, k=2.0, sim=b",
        "rate=2, k=5.0, sim=a",
        "rate=2, k=5.0, sim=b",
    ]


def test_a_scan_keeps_the_log_level_of_roadrunner(
    simulator: Simulator, capfd: pytest.CaptureFixture[str]
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": [2.0]})])
    level = roadrunner.Logger.getLevel()
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_WARNING)
    try:
        simulator.run(blowup, scan, on_error="flag")
        assert roadrunner.Logger.getLevel() == roadrunner.Logger.LOG_WARNING
        # also when the point raises
        with pytest.raises(ScanError):
            simulator.run(blowup, scan)
        assert roadrunner.Logger.getLevel() == roadrunner.Logger.LOG_WARNING
    finally:
        roadrunner.Logger.setLevel(level)
        _c_output(capfd)


def test_the_errors_are_capped(
    simulator: Simulator,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    k = [0.1] + [2.0] * (MAX_ERRORS + 2)
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": k})])
    try:
        res = simulator.run(blowup, scan, on_error="flag")
    finally:
        _c_output(capfd)
    assert int(res["status"].sum()) == MAX_ERRORS + 2
    assert [e.split(",")[0] for e in res.ds.attrs["errors"]] == [
        f"rate={i}" for i in range(1, MAX_ERRORS + 1)
    ]
    assert f"{MAX_ERRORS + 2} of {MAX_ERRORS + 3} points" in caplog.text


def test_a_selection_which_the_result_reserves_is_an_error(
    simulator: Simulator,
) -> None:
    model = RoadrunnerSBMLModel(
        source=sbml(PROBE.replace("f = 2", "f = 2; status = 1"))
    )
    with pytest.raises(ValueError, match=r"\['status'\]"):
        simulator.run(model, Simulation(end=1, steps=2), on_error="flag")
    model.set_selections(["time", "[A]", "status"])
    with pytest.raises(ValueError, match=r"\['status'\]"):
        simulator.run(model, Simulation(end=1, steps=2))


def test_a_worker_process_silences_sundials(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    names = worker_module.SUNDIALS_LOGS
    before = {name: os.environ.get(name) for name in names}
    with monkeypatch.context() as mp:
        for name in names:
            # recorded even if it is not set, so the undo removes what
            # quiet_sundials sets
            mp.setenv(name, "")
            mp.delenv(name)
        mp.setattr(worker_module.parallel, "in_worker", lambda: False)
        worker_module.quiet_sundials()
        assert not set(names) & set(os.environ)
        mp.setattr(worker_module.parallel, "in_worker", lambda: True)
        mp.setenv(names[0], "mine.log")
        worker_module.quiet_sundials()
        # a setting of the user stays
        assert os.environ[names[0]] == "mine.log"
        mp.delenv(names[0])
        worker_module.quiet_sundials()
        assert {os.environ[name] for name in names} == {os.devnull}
        # a model loaded after it fails without the messages of SUNDIALS
        blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
        scan = Scan(
            Simulation(end=1, steps=4), [Dimension("rate", values={"k": [2.0]})]
        )
        _c_output(capfd)
        res = Simulator(n_workers=1).run(blowup, scan, on_error="flag")
        assert res["status"].values.tolist() == [1]
        assert "cvodes" not in _c_output(capfd)
    # the process of the tests is not silenced
    assert {name: os.environ.get(name) for name in names} == before


def test_no_compile_per_point(
    simulator: Simulator, model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    compile_simulation = simulator_module.compile_simulation

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return compile_simulation(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(simulator_module, "compile_simulation", counting)
    values = Scan(
        Simulation(end=1, steps=2),
        [
            Dimension("a", values={"b0": [1.0, 2.0]}),
            Dimension("b", values={"k2": [0.1, 0.2, 0.3]}, at=0.5),
        ],
    )
    simulator.run(model, values)
    assert len(calls) == 1
    simulations = {"a": Simulation(end=1, steps=2), "b": Simulation(end=2, steps=2)}
    calls.clear()
    simulator.run(
        model,
        Scan(
            Simulation(end=1),
            [
                Dimension("sim", simulations=simulations),
                Dimension("b", values={"k2": [0.1, 0.2]}),
            ],
        ),
    )
    assert len(calls) == 2


def test_the_objects_of_the_user_are_not_changed(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    simulation = Simulation(
        end=1, steps=2, preinit_changes={"a0": 2.0}, changes=[Change(0.5, {"k1": 0.1})]
    )
    dimension = Dimension("d", values={"k1": [0.3, 0.4]}, at=0.5)
    assert model.selections is not None
    before = (simulation.to_dict(), dimension.to_dict(), list(model.selections))
    simulator.run(
        model, Scan(simulation, [dimension, Dimension("e", values={"a0": [1.0, 5.0]})])
    )
    assert (simulation.to_dict(), dimension.to_dict(), list(model.selections)) == before


def test_a_dimension_named_as_a_selection_is_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    with pytest.raises(ValueError, match=r"\['X'\]"):
        simulator.run(
            model, Scan(Simulation(end=1), [Dimension("X", values={"b0": [1.0]})])
        )


def test_a_scanned_selection_is_a_variable_and_no_coordinate(
    simulator: Simulator,
) -> None:
    model = RoadrunnerSBMLModel(source=sbml())  # every entity is selected
    scan = Scan(Simulation(end=1, steps=2), [Dimension("d", values={"k1": [0.5, 1.0]})])
    res = simulator.run(model, scan)
    assert res["k1"].dims == ("d", "time")
    np.testing.assert_allclose(res["k1"].values[:, 0], [0.5, 1.0])
    assert "k1" not in res.ds.coords


def test_ragged_chunks_are_padded_to_the_longest(
    simulator: Simulator, model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(simulator_module, "MAX_CHUNK", 1)
    # the first chunk is the short one
    scan = Scan(Simulation(end=2), [Dimension("d", values={"k1": [0.1, 30.0]})])
    res = simulator.run(model, scan)
    lengths = []
    for k, k1 in enumerate([0.1, 30.0]):
        expected = simulator.simulate(
            model, Simulation(end=2, preinit_changes={"k1": k1})
        )
        n = len(expected)
        lengths.append(n)
        np.testing.assert_allclose(
            res["[A]"].values[k, :n], expected["[A]"], rtol=1e-12
        )
        assert np.isnan(res["time"].values[k, n:]).all()
    assert lengths[0] < lengths[1] == res.ds.sizes["_point"]


def test_the_chunks_share_a_plan_and_are_bounded(
    simulator: Simulator, model: RoadrunnerSBMLModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[tuple[int, int]] = []
    run_chunk = simulator_module.run_chunk

    def recording(chunk: object, model: RoadrunnerSBMLModel) -> object:
        sizes.append((id(chunk.plan), len(chunk.indices)))  # ty: ignore[unresolved-attribute]
        return run_chunk(chunk, model)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(simulator_module, "run_chunk", recording)
    simulations = {"a": Simulation(end=1, steps=2), "b": Simulation(end=2, steps=2)}
    scan = Scan(
        Simulation(end=1),
        [
            Dimension("d", values={"k2": np.linspace(0.1, 1, 9)}),
            Dimension("sim", simulations=simulations),
        ],
    )
    simulator.run(model, scan)
    # ceil(18 / 4) = 5 points at most, 9 points per plan
    assert sorted(n for _, n in sizes) == [4, 4, 5, 5]
    assert len({plan for plan, _ in sizes}) == 2


def test_the_changes_of_a_model_are_defaults(
    simulator: Simulator, tmp_path: Path
) -> None:
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    abstract = AbstractModel(source=path, changes={"b0": 3.0})
    res = simulator.run(abstract, Simulation(end=1, steps=2))
    assert res["[B]"].values[0] == pytest.approx(3.0)
    scan = Scan(Simulation(end=1, steps=2), [Dimension("d", values={"b0": [1.0, 4.0]})])
    res = simulator.run(abstract, scan)
    np.testing.assert_allclose(res["[B]"].values[:, 0], [1.0, 4.0])


def test_a_result_is_written_and_read(
    simulator: Simulator, model: RoadrunnerSBMLModel, tmp_path: Path
) -> None:
    scan = Scan(
        Simulation(end=1, steps=2), [Dimension("dose", values={"b0": [1.0, 2.0]})]
    )
    res = simulator.run(model, scan, on_error="flag")
    res.to_netcdf(tmp_path / "res.nc")
    again = ScanResult.from_netcdf(tmp_path / "res.nc")
    np.testing.assert_array_equal(again["[B]"].values, res["[B]"].values)
    assert again.ds.attrs == res.ds.attrs


def test_the_integrator_settings_reach_the_model(model: RoadrunnerSBMLModel) -> None:
    tolerance = AbsoluteTolerance(ids={"A": 1e-14})
    simulator = Simulator(
        n_workers=1, absolute_tolerance=tolerance, relative_tolerance=1e-8
    )
    assert simulator.load(model) is model
    assert model.absolute_tolerance == tolerance
    assert model.r_loaded.getIntegrator().getValue("relative_tolerance") == 1e-8


def test_an_unknown_setting_is_an_error(model: RoadrunnerSBMLModel) -> None:
    with pytest.raises(ValueError, match="no settings"):
        Simulator(n_workers=1, nope=1.0).run(model, Simulation(end=1))


def test_the_arguments_are_checked(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    with pytest.raises(ValueError, match="at least 1, not 0"):
        Simulator(n_workers=0)
    with pytest.raises(ValueError, match="'raise' or 'flag'"):
        simulator.run(model, Simulation(end=1), on_error="ignore")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Unsupported model type"):
        simulator.load(1.0)  # ty: ignore[invalid-argument-type]


# the tolerance integration tests of 0.8.5, on Simulator.load and simulate


def test_the_roadrunner_instance_follows_a_derived_model(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    """A model which is derived for a pre-initialization change is the one run."""
    tolerances = _vector(model)
    result = simulator.simulate(
        model, Simulation(end=1, preinit_changes={"pinit": 7.0}, times=[0])
    )
    assert result["X"][0] == pytest.approx(21.0)
    # the change is no change of the model: X = 3 * pinit = 3 * 2 * f
    result = simulator.simulate(model, Simulation(end=1, times=[0]))
    assert result["X"][0] == pytest.approx(12.0)
    assert _vector(model) == tolerances


def _vector(model: RoadrunnerSBMLModel) -> dict[str, float]:
    """Get the absolute tolerances which CVODE uses by the id of its state."""
    integrator = model.r_loaded.getIntegrator()
    return dict(
        zip(
            model.state_ids(),
            (float(v) for v in integrator.getAbsoluteToleranceVector()),
            strict=True,
        )
    )


def test_integrator_settings_are_passed_on_and_kept(tmp_path: Path) -> None:
    """Every setting of the integrator reaches roadrunner, also for a later model."""
    path = tmp_path / "probe.xml"
    path.write_text(sbml())
    simulator = Simulator(n_workers=1, initial_time_step=1e-9)
    simulator.set_integrator_settings(maximum_num_steps=1234)
    # a new model, as for every task of an experiment; the model keeps the
    # instance of roadrunner which owns the integrator
    model = simulator.load(path)
    integrator = model.r_loaded.getIntegrator()
    assert integrator.getValue("initial_time_step") == pytest.approx(1e-9)
    assert integrator.getValue("maximum_num_steps") == 1234
    assert integrator.getValue("relative_tolerance") == 1e-10


def test_the_tolerance_of_every_state_reaches_cvode(tmp_path: Path) -> None:
    """The tolerances per state are the vector of CVODE, after a new model too."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-8, other=1e-7)
    simulator = Simulator(n_workers=1, absolute_tolerance=tolerance)
    # A and S are concentration species in C = 2 and U = 1e-12 (raised to
    # 2e-6), X an amount species, D a parameter with a rate rule, which
    # roadrunner integrates before the species
    expected = {"A": 1e-8 * 2, "S": 1e-8 * 2e-6, "X": 1e-9, "D": 1e-7}
    model = simulator.load(path)
    assert _vector(model) == pytest.approx(expected, rel=1e-12, abs=0)
    model = simulator.load(path)
    assert _vector(model) == pytest.approx(expected, rel=1e-12, abs=0)
    table = model.tolerances()
    assert list(table["sid"]) == model.state_ids()
    assert dict(zip(table["sid"], table["absolute_tolerance"], strict=True)) == (
        pytest.approx(expected, rel=1e-12, abs=0)
    )
    assert set(table["kind"]) == {"amount", "concentration", "other"}


def test_a_float_tolerance_is_the_same_for_every_kind(tmp_path: Path) -> None:
    """The scaling of roadrunner by the initial values is not used."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    model = Simulator(n_workers=1, absolute_tolerance=1e-10).load(path)
    expected = {"A": 1e-10 * 2, "S": 1e-10 * 2e-6, "X": 1e-10, "D": 1e-10}
    assert _vector(model) == pytest.approx(expected, rel=1e-12, abs=0)


#: a species which decays and a constant with a rate rule, which roadrunner
#: integrates before the species
DECAY = """
model decay
  compartment C = 1;
  species A in C;
  A = 1; P = 1
  P' = 0
  J: A -> ; A
end
"""


def test_the_tolerance_of_a_state_controls_that_state(tmp_path: Path) -> None:
    """A loose tolerance of one state does not loosen the error of another."""
    path = tmp_path / "decay.xml"
    path.write_text(sbml(DECAY))
    tolerance = AbsoluteTolerance(
        amount=1e-12, concentration=1e-12, other=1e-12, ids={"P": 1e-2}
    )
    simulator = Simulator(
        n_workers=1, absolute_tolerance=tolerance, relative_tolerance=1e-6
    )
    res = simulator.run(path, Simulation(end=30, steps=30))
    time = np.asarray(res["time"].values, dtype=float)
    error = np.abs(np.asarray(res["[A]"].values, dtype=float) - np.exp(-time))
    assert np.max(error) < 1e-5


def test_a_degenerate_compartment_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A compartment whose volume is raised to the floor is logged once per model."""
    path = tmp_path / "tolerances.xml"
    path.write_text(sbml(TOLERANCE_PROBE))
    with caplog.at_level("WARNING"):
        simulator = Simulator(n_workers=1)
        model = simulator.load(path)
        simulator.set_integrator_settings(absolute_tolerance=1e-9)
        simulator.load(model)
    messages = [r.getMessage() for r in caplog.records if "'U'" in r.getMessage()]
    assert len(messages) == 1


def test_the_tolerances_are_set_once_per_roadrunner_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A simulation does not set the tolerances again, a change of them does."""
    calls: list[AbsoluteTolerance] = []
    set_absolute_tolerance = RoadrunnerSBMLModel._set_absolute_tolerance

    def spy(self: RoadrunnerSBMLModel, tolerance: AbsoluteTolerance) -> None:
        calls.append(tolerance)
        set_absolute_tolerance(self, tolerance)

    monkeypatch.setattr(RoadrunnerSBMLModel, "_set_absolute_tolerance", spy)
    model = RoadrunnerSBMLModel(source=sbml())
    tolerance = AbsoluteTolerance(amount=1e-9, concentration=1e-8, other=1e-7)
    simulator = Simulator(n_workers=1, absolute_tolerance=tolerance)
    simulation = Simulation(end=1, steps=2)
    calls.clear()
    simulator.simulate(model, simulation)
    expected = _vector(model)
    simulator.simulate(model, simulation)
    simulator.load(model)
    assert calls == [tolerance]
    # a change of the setting
    simulator.set_integrator_settings(absolute_tolerance=1e-6)
    simulator.simulate(model, simulation)
    assert calls == [tolerance, AbsoluteTolerance.of(1e-6)]
    assert set(_vector(model).values()) == {1e-6 * 2, 1e-6}
    # the tolerances of the integrator changed elsewhere
    simulator.set_integrator_settings(absolute_tolerance=tolerance)
    simulator.simulate(model, simulation)
    integrator = model.r_loaded.getIntegrator()
    integrator.setValue("absolute_tolerance", 1e-3)
    integrator.setValue("relative_tolerance", 1e-3)
    calls.clear()
    simulator.simulate(model, simulation)
    assert calls == [tolerance]
    assert _vector(model) == expected
    assert integrator.getValue("relative_tolerance") == 1e-10
    # a new instance of roadrunner, the model does not keep the old one alive
    old = weakref.ref(model.r_loaded)
    model.r = roadrunner.RoadRunner(model.r_loaded.getSBML())
    gc.collect()
    assert old() is None
    calls.clear()
    simulator.simulate(model, simulation)
    assert calls == [tolerance]
    assert _vector(model) == expected
    # another simulator with the same settings
    calls.clear()
    Simulator(n_workers=1, absolute_tolerance=tolerance).simulate(model, simulation)
    assert calls == []
