"""A scan computes observables from every simulation."""

import json
from pathlib import Path

import numpy as np
import pkpdutils as pk
import pytest
import xarray as xr

import sbmlsim.simulator.simulator as simulator_module
from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.result import ScanResult
from sbmlsim.simulation import PK, Change, Custom, Dimension, Formula, Scan, Simulation
from sbmlsim.simulation.observables import ObservableKind
from sbmlsim.simulator import ScanError, Simulator
from sbmlsim.units import ureg
from tests.simulator.models import (
    BLOWUP,
    auc_of_c,
    doubled,
    fails_for_large_k1,
    last_value,
    sbml,
    sbml_pk,
)

KA, KE, V = 1.0, 0.2, 10.0


def concentration(t: np.ndarray | float, dose: float) -> np.ndarray:
    return dose * KA / (V * (KA - KE)) * (np.exp(-KE * t) - np.exp(-KA * t))


@pytest.fixture(scope="module")
def pk_sbml() -> str:
    return sbml_pk()


def dosed(steps: int | None = 480, times: tuple[float, ...] = (0.0,)) -> Simulation:
    return Simulation(
        end=48, steps=steps, changes=[Change(list(times), {"PODOSE": Q(100, "mg")})]
    )


def dose_scan(
    steps: int | None = 480, doses: tuple[float, ...] = (50.0, 100.0, 200.0)
) -> Scan:
    return Scan(
        dosed(steps), [Dimension("dose", values={"PODOSE": Q(list(doses), "mg")})]
    )


REDUCTIONS = [
    Formula("c", "[C]"),
    Formula("cmax", "max(c)"),
    Formula("cmin", "min(c)"),
    Formula("cmean", "mean(c)"),
    Formula("c10", "at(c, 10)"),
    Formula("rel", "c / cmax"),
]


def test_reductions_against_the_analytic_solution(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, dose_scan(), REDUCTIONS)
    grid = res["time"].values
    assert res["cmax"].dims == ("dose",) and res["c"].dims == ("dose", "time")
    for k, dose in enumerate((50.0, 100.0, 200.0)):
        exact = concentration(grid, dose)
        assert float(res["cmax"][k]) == pytest.approx(exact.max(), rel=1e-6)
        assert float(res["cmin"][k]) == pytest.approx(exact.min(), abs=1e-9)
        assert float(res["cmean"][k]) == pytest.approx(
            np.trapezoid(exact, grid) / 48.0, rel=1e-6
        )
        assert float(res["c10"][k]) == pytest.approx(
            concentration(10.0, dose), rel=1e-6
        )
    np.testing.assert_allclose(res["rel"].values.max(axis=1), 1.0)
    assert ureg.Quantity(1.0, res.units["cmax"]).to("mg/l").magnitude == pytest.approx(
        1.0
    )
    assert res.units["rel"] == "dimensionless"


def test_the_observables_are_computed_on_the_native_solution(pk_sbml: str) -> None:
    observables = [*REDUCTIONS, PK("p", "[C]", dose="PODOSE", route="oral")]
    scan = dose_scan(steps=None, doses=(50.0, 200.0))
    grid = np.linspace(0.0, 48.0, 7)
    native = Simulator().run(pk_sbml, scan, observables)
    gridded = Simulator().run(pk_sbml, scan, observables, time=grid)
    assert native.ragged and not gridded.ragged
    scalars = [str(n) for n in native.ds.data_vars if native.ds[n].dims == ("dose",)]
    assert {"cmax", "cmean", "c10", "p.cmax", "p.cl_f"} <= set(scalars)
    for name in scalars:
        xr.testing.assert_identical(gridded[name], native[name])
    # the maximum of the solution, not of the coarse grid
    assert (gridded["cmax"].values > gridded["c"].values.max(axis=1)).all()
    assert gridded["c"].dims == ("dose", "time")
    np.testing.assert_array_equal(gridded["time"].values, grid)
    on_grid = native.interpolate(grid)
    for name in ("c", "rel"):
        np.testing.assert_array_equal(gridded[name].values, on_grid[name].values)


def test_a_baseline_formula_broadcasts_over_the_time(pk_sbml: str) -> None:
    observables = [
        Formula("c", "[C]"),
        Formula("c1", "at(c, 1)"),
        Formula("rel", "c / c1"),
    ]
    res = Simulator().run(pk_sbml, dose_scan(steps=48), observables)
    np.testing.assert_allclose(res["rel"].sel(time=1.0).values, 1.0)


def test_reductions_on_the_steps_of_the_integrator(pk_sbml: str) -> None:
    res = Simulator().run(
        pk_sbml, dose_scan(steps=None, doses=(50.0, 200.0)), REDUCTIONS
    )
    assert res.ragged
    for k, dose in enumerate((50.0, 200.0)):
        time = res["time"].values[k]
        time = time[np.isfinite(time)]
        exact = concentration(time, dose)
        assert float(res["cmax"][k]) == pytest.approx(exact.max(), rel=1e-6)
        assert float(res["cmean"][k]) == pytest.approx(
            np.trapezoid(exact, time) / 48.0, rel=1e-6
        )


def test_pk_equals_pkpdutils_on_the_same_arrays(pk_sbml: str) -> None:
    observables = [Formula("conc", "[C]"), PK("c", "[C]", dose="PODOSE", route="oral")]
    res = Simulator().run(pk_sbml, dose_scan(doses=(50.0, 100.0)), observables)
    timecourses = pk.Timecourses.from_arrays(
        res["time"].values,
        res["conc"].values,
        time_unit="hr",
        unit="mg/l",
        dims=("_sim",),
        dose={
            "amount": np.array([[50.0], [100.0]]),
            "time": np.zeros((2, 1)),
            "unit": "mg",
        },
        route="oral",
    )
    direct = pk.nca(timecourses)
    for parameter in ("cmax", "tmax", "auc_inf_obs", "thalf", "cl_f"):
        np.testing.assert_allclose(
            res[f"c.{parameter}"].values, direct.ds[parameter].values, rtol=1e-9
        )
    assert float(res["c.auc_inf_obs"][1]) == pytest.approx(100.0 / (V * KE), rel=2e-3)
    assert float(res["c.thalf"][1]) == pytest.approx(np.log(2.0) / KE, rel=5e-3)
    assert float(res["c.tmax"][1]) == pytest.approx(
        np.log(KA / KE) / (KA - KE), abs=0.1
    )
    assert float(res["c.cl_f"][0]) == pytest.approx(V * KE, rel=2e-3)
    assert float(res["c.cl_f"][1]) == pytest.approx(V * KE, rel=2e-3)


def test_a_multiple_dosing(pk_sbml: str) -> None:
    observables = [
        PK("c", "[C]", dose="PODOSE", route="oral"),
        Formula("depot", "at(PODOSE, 24)"),
    ]
    res = Simulator().run(pk_sbml, dosed(times=(0.0, 24.0)), observables)
    assert float(res["c.tau"]) == pytest.approx(24.0)
    # at the time of a change, the value after it: the new dose in the depot
    assert float(res["depot"]) == pytest.approx(100.0)


def test_a_zero_dose_first_keeps_the_dose_parameters(pk_sbml: str) -> None:
    res = Simulator().run(
        pk_sbml,
        dose_scan(doses=(0.0, 50.0, 100.0)),
        [PK("c", "[C]", dose="PODOSE", route="oral")],
    )
    cl_f = res["c.cl_f"].values
    assert np.isnan(cl_f[0])
    np.testing.assert_allclose(cl_f[1:], V * KE, rtol=2e-3)
    assert np.isfinite(res["c.cmax"].values).all()


def test_a_second_dose_which_is_zero_first_keeps_the_interval(pk_sbml: str) -> None:
    scan = Scan(
        dosed(),
        [Dimension("second", values={"PODOSE": Q([0.0, 100.0], "mg")}, at=24.0)],
    )
    res = Simulator().run(pk_sbml, scan, [PK("c", "[C]", dose="PODOSE", route="oral")])
    tau = res["c.tau"].values
    assert np.isnan(tau[0])
    assert tau[1] == pytest.approx(24.0)
    np.testing.assert_allclose(res["c.cl_f"].values[0], V * KE, rtol=2e-3)


def test_pk_without_a_dose(pk_sbml: str) -> None:
    res = Simulator().run(pk_sbml, dosed(), [PK("c", "[C]")])
    assert "c.cmax" in res and "c.cl_f" not in res


def test_a_dose_only_a_dimension_gives(pk_sbml: str) -> None:
    scan = Scan(
        Simulation(end=48, steps=480),
        [Dimension("dose", values={"PODOSE": Q([50.0, 100.0], "mg")})],
    )
    res = Simulator().run(pk_sbml, scan, [PK("c", "[C]", dose="PODOSE", route="oral")])
    np.testing.assert_allclose(res["c.cl_f"].values, V * KE, rtol=2e-3)


def test_custom_observables(pk_sbml: str) -> None:
    observables = [
        Formula("c", "[C]"),
        Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
        Custom(
            "twice", doubled, "mg/l", symbols=["[C]"], kind=ObservableKind.TIMECOURSE
        ),
    ]
    res = Simulator().run(pk_sbml, dose_scan(), observables)
    np.testing.assert_allclose(
        res["auc"].values, np.trapezoid(res["c"].values, res["time"].values, axis=1)
    )
    np.testing.assert_allclose(res["twice"].values, 2.0 * res["c"].values)


def test_keep_drops_intermediates_but_evaluates_them(pk_sbml: str) -> None:
    full = Simulator().run(pk_sbml, dose_scan(), REDUCTIONS)
    kept = Simulator().run(pk_sbml, dose_scan(), REDUCTIONS, keep=["rel", "c10"])
    assert set(kept.ds.data_vars) == {"rel", "c10"}
    xr.testing.assert_equal(kept["rel"], full["rel"])
    xr.testing.assert_equal(kept["c10"], full["c10"])


def test_a_result_of_values_per_simulation_has_no_time(
    pk_sbml: str, tmp_path: Path
) -> None:
    observables = [
        PK("c", "[C]", dose="PODOSE", route="oral"),
        Formula("cmax", "max([C])"),
    ]
    res = Simulator().run(pk_sbml, dose_scan(), observables, keep=["c", "cmax"])
    assert "time" not in res.ds.dims and "time" not in res.ds.variables
    assert res["cmax"].dims == ("dose",)
    again = ScanResult.from_netcdf(_write(res, tmp_path))
    xr.testing.assert_equal(again.ds, res.ds)
    summary = res.summary("dose")
    assert "statistic" in summary.ds.dims
    stored = json.loads(json.dumps(res.ds.attrs["observables"]))
    assert [o["id"] for o in stored] == ["c", "cmax"]


def _write(res: ScanResult, tmp_path: Path) -> Path:
    path = tmp_path / "result.nc"
    res.to_netcdf(path)
    return path


def test_observables_read_what_the_model_does_not_select(pk_sbml: str) -> None:
    model = RoadrunnerSBMLModel(source=pk_sbml)
    model.set_selections(["time"])
    res = Simulator().run(model, dosed(), [Formula("c", "[C]")])
    assert float(res["c"].max()) == pytest.approx(
        concentration(np.linspace(0, 48, 481), 100.0).max(), rel=1e-6
    )


def test_the_result_does_not_depend_on_the_workers_or_the_chunks(
    pk_sbml: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observables = [
        *REDUCTIONS,
        PK("p", "[C]", dose="PODOSE", route="oral"),
        Custom("auc", auc_of_c, "mg*hr/l", symbols=["[C]"]),
    ]
    scan = dose_scan(doses=tuple(np.linspace(10.0, 200.0, 12)))
    reference = Simulator(n_workers=1).run(pk_sbml, scan, observables)
    for workers in (2, 4):
        res = Simulator(n_workers=workers).run(pk_sbml, scan, observables)
        xr.testing.assert_equal(res.ds, reference.ds)
    monkeypatch.setattr(simulator_module, "_chunk_size", lambda n, w: 1)
    xr.testing.assert_equal(
        Simulator(n_workers=2).run(pk_sbml, scan, observables).ds, reference.ds
    )
    monkeypatch.setattr(simulator_module, "_chunk_size", lambda n, w: 1000)
    xr.testing.assert_equal(
        Simulator(n_workers=1).run(pk_sbml, scan, observables).ds, reference.ds
    )


def test_a_failing_point_is_nan_for_every_observable() -> None:
    scan = Scan(
        Simulation(end=1, steps=4), [Dimension("rate", values={"k": [0.1, 2.0]})]
    )
    observables = [
        Formula("smax", "max(S)"),
        Formula("s", "S"),
        Custom("last", last_value, "dimensionless", symbols=["S"]),
    ]
    res = Simulator().run(sbml(BLOWUP), scan, observables, on_error="flag")
    assert res["status"].values.tolist() == [0, 1]
    assert np.isfinite(res["smax"].values[0]) and np.isnan(res["smax"].values[1])
    assert np.isnan(res["last"].values[1]) and np.isnan(res["s"].values[1]).all()
    with pytest.raises(ScanError, match=r"k=2\.0"):
        Simulator().run(sbml(BLOWUP), scan, observables)


def test_an_observable_which_fails_fails_its_point() -> None:
    scan = Scan(Simulation(end=1, steps=4), [Dimension("d", values={"k1": [0.5, 2.0]})])
    observables = [Custom("check", fails_for_large_k1, "dimensionless", symbols=["k1"])]
    res = Simulator().run(sbml(), scan, observables, on_error="flag")
    assert res["status"].values.tolist() == [0, 1]
    assert "k1 is too large" in res.ds.attrs["errors"][0]
    with pytest.raises(ScanError, match="k1 is too large"):
        Simulator().run(sbml(), scan, observables)


def test_the_observables_are_checked_when_the_scan_is_compiled(pk_sbml: str) -> None:
    with pytest.raises(ValueError, match="'nope'"):
        Simulator().run(pk_sbml, dose_scan(), [Formula("x", "nope")])
    scan = Scan(dosed(), [Dimension("cmax", values={"PODOSE": Q([1.0, 2.0], "mg")})])
    with pytest.raises(ValueError, match="cmax"):
        Simulator().run(pk_sbml, scan, [Formula("cmax", "max([C])")])


def test_a_dimension_of_models_with_observables() -> None:
    models = {"slow": sbml_pk(ke=0.2), "fast": sbml_pk(ke=0.3)}
    scan = Scan(dosed(), [Dimension("model", models=models)])
    res = Simulator().run(None, scan, [PK("c", "[C]", dose="PODOSE", route="oral")])
    np.testing.assert_allclose(
        res["c.thalf"].values, np.log(2.0) / np.array([0.2, 0.3]), rtol=5e-3
    )


def test_a_pk_parameter_of_a_ragged_scan_does_not_depend_on_the_chunks(
    pk_sbml: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = dose_scan(steps=None, doses=(50.0, 100.0, 200.0, 20.0))
    observables = [PK("p", "[C]", dose="PODOSE", route="oral")]
    monkeypatch.setattr(simulator_module, "_chunk_size", lambda n, w: 1000)
    one = Simulator(n_workers=1).run(pk_sbml, scan, observables)
    monkeypatch.setattr(simulator_module, "_chunk_size", lambda n, w: 1)
    many = Simulator(n_workers=1).run(pk_sbml, scan, observables)
    xr.testing.assert_equal(many.ds, one.ds)
