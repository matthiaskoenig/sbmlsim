"""The simulator runs a scan and answers with a ScanResult, here serially."""

import ctypes
import os
import sys
from pathlib import Path

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
from tests.simulator.models import BLOWUP, PROBE, TOLERANCE_PROBE, sbml, sbml_minutes

posix = pytest.mark.skipif(
    sys.platform == "win32", reason="the streams of C are flushed with POSIX ctypes"
)

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
    """Take the output of C of the points which failed, see test_worker."""
    ctypes.CDLL(None).fflush(None)
    captured = capfd.readouterr()
    return captured.out + captured.err


def test_the_selections_of_a_model() -> None:
    model = RoadrunnerSBMLModel(source=sbml())
    assert model.selections is not None
    every = list(model.selections)
    model.set_selections(["time", "X"])
    assert model.selections == ["time", "X"]
    assert list(model.r_loaded.timeCourseSelections) == ["time", "X"]
    model.set_selections(None)
    assert model.selections == every


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
    with pytest.raises(ValueError, match="replaces the model"):
        simulator.run(model, scan)
    with pytest.raises(ValueError, match="needs a model"):
        simulator.run(None, Simulation(end=1))


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


def test_a_grid_of_times_with_a_unit() -> None:
    minutes = RoadrunnerSBMLModel(source=sbml_minutes())
    res = Simulator(n_workers=1).run(minutes, Simulation(end=2), time=Q([0, 60], "s"))
    assert res["time"].values.tolist() == [0.0, 1.0]


def test_an_empty_grid_of_times_is_an_error(
    simulator: Simulator, model: RoadrunnerSBMLModel
) -> None:
    with pytest.raises(ValueError, match="empty"):
        simulator.run(model, Simulation(end=1), time=[])


@posix
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


@posix
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


@posix
def test_the_first_failed_point_in_scan_order_raises(
    simulator: Simulator, capfd: pytest.CaptureFixture[str]
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    try:
        with pytest.raises(ScanError, match=r"The point rate=1, k=2\.0, sim=b of"):
            simulator.run(blowup, _failing_scan())
    finally:
        _c_output(capfd)


@posix
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
    assert np.isnan(res["S"].values[1]).all()
    assert np.isfinite(res["S"].values[[0, 2]]).all()
    assert res.ds.attrs["errors"][0].startswith("rate=1, k=2.0: RuntimeError")
    assert "1 of 3 points of the scan failed" in caplog.text
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    # roadrunner does not repeat the error of every point which failed
    assert "CVODE Error" not in output


@posix
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


@posix
def test_a_flagged_scan_keeps_the_log_level_of_roadrunner(
    simulator: Simulator, capfd: pytest.CaptureFixture[str]
) -> None:
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": [2.0]})])
    level = roadrunner.Logger.getLevel()
    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_WARNING)
    try:
        simulator.run(blowup, scan, on_error="flag")
        assert roadrunner.Logger.getLevel() == roadrunner.Logger.LOG_WARNING
    finally:
        roadrunner.Logger.setLevel(level)
        _c_output(capfd)


@posix
def test_a_worker_process_silences_sundials(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    for name in worker_module.SUNDIALS_LOGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(worker_module.parallel, "in_worker", lambda: False)
    worker_module.quiet_sundials()
    assert not set(worker_module.SUNDIALS_LOGS) & set(os.environ)
    monkeypatch.setattr(worker_module.parallel, "in_worker", lambda: True)
    monkeypatch.setenv(worker_module.SUNDIALS_LOGS[0], "mine.log")
    worker_module.quiet_sundials()
    # a setting of the user stays
    assert os.environ[worker_module.SUNDIALS_LOGS[0]] == "mine.log"
    monkeypatch.delenv(worker_module.SUNDIALS_LOGS[0])
    worker_module.quiet_sundials()
    assert {os.environ[name] for name in worker_module.SUNDIALS_LOGS} == {os.devnull}
    # a model loaded after it fails without the messages of SUNDIALS
    blowup = RoadrunnerSBMLModel(source=sbml(BLOWUP))
    scan = Scan(Simulation(end=1, steps=4), [Dimension("rate", values={"k": [2.0]})])
    _c_output(capfd)
    res = Simulator(n_workers=1).run(blowup, scan, on_error="flag")
    assert res["status"].values.tolist() == [1]
    assert "cvodes" not in _c_output(capfd)


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
    result = simulator.simulate(
        model, Simulation(end=1, preinit_changes={"pinit": 7.0}, times=[0])
    )
    assert result["X"][0] == pytest.approx(21.0)
    assert simulator.load(model).r_loaded is model.r_loaded


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
