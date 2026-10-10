"""Fit mappings of a value per simulation and of values over a dimension."""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import sbmlsim.fit.optimization as optimization_module
from sbmlsim import Q
from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitMappingCollection, FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, ParameterScaleType, ResidualType
from sbmlsim.fit.runner import run_optimization
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import Change, Custom, Dimension, Observable, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.worker import SUNDIALS_LOGS
from tests.fit.hooks import Scaling
from tests.fit.scalar_experiment import (
    CMAX,
    CMAX_UNIT,
    DOSES,
    OBSERVABLES,
    TRUE_V,
    DoseStudy,
    MixedStudy,
    OutsideStudy,
    RowsStudy,
    ScalarStudy,
    dose_scan,
    dosed,
)
from tests.simulator.models import sbml_pk


def _problem(
    experiment: type[SimulationExperiment],
    mappings: list[str],
    tmp_path: Path,
    parameter: FitParameter | None = None,
    **settings: Any,
) -> OptimizationProblem:
    problem = OptimizationProblem(
        "scalar",
        [FitMappingCollection(experiment=experiment, mappings=mappings)],
        [
            parameter
            or FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(
        FitSettings(parameter_scale=ParameterScaleType.LINEAR, **settings)
    )
    return problem


class TwoDimStudy(DoseStudy):
    """The scan over the doses has a second dimension over the elimination."""

    def simulations(self) -> dict[str, Simulation | Scan]:
        scan = Scan(
            dosed(),
            [
                Dimension("dose", values={"PODOSE": Q(DOSES, "mg")}),
                Dimension("rate", values={"ke": Q([0.1, 0.2], "1/hr")}),
            ],
        )
        return {"single": dosed(), "doses": scan}


class SelectedPointStudy(MixedStudy):
    """The timecourse and the cmax of points of the scan over the doses.

    The mappings of the point of 100 mg compare with the data of the plain
    simulation of 100 mg; the timecourse of the point of 50 mg is another
    selection of the same task.
    """

    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = super().fit_mappings()
        for key, label in (("fm_tc_scan", 1), ("fm_tc_scan_low", 0)):
            mappings[key] = FitMapping(
                self,
                reference=FitData(self, dataset="tc", xid="time", yid="C"),
                observable=FitData(
                    self, task="task_doses", xid="time", yid="[C]", sel={"dose": label}
                ),
            )
        mappings["fm_cmax_scan"] = FitMapping(
            self,
            reference=FitData(
                self, dataset="tab", xid=None, yid="cmax", yid_sd="cmax_sd"
            ),
            observable=FitData(
                self, task="task_doses", xid=None, yid="pk.cmax", sel={"dose": 1}
            ),
        )
        return mappings


class EmptyScanStudy(MixedStudy):
    """The simulation of 100 mg as a scan without dimensions."""

    def simulations(self) -> dict[str, Simulation | Scan]:
        return {"single": Scan(dosed(), []), "doses": dose_scan()}


class EmptyRowsStudy(ScalarStudy):
    """A selection of rows which no row of the study table has."""

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_empty": FitMapping(
                self,
                reference=FitData(
                    self, dataset="tab", xid=None, yid="cmax", sel={"cmax": -1.0}
                ),
                observable=FitData(self, task="task_single", xid=None, yid="pk.cmax"),
            )
        }


class LabelsStudy(DoseStudy):
    """The cmax of 50 and 200 mg against points of the scan selected by labels."""

    REF_DOSES: ClassVar[list[float]] = [50.0, 200.0]

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            key: FitMapping(
                self,
                reference=FitData(self, dataset="tab_doses", xid="dose", yid="cmax"),
                observable=FitData(
                    self,
                    task="task_doses",
                    xid="dose.PODOSE",
                    yid="pk.cmax",
                    sel={"dose": labels},
                ),
            )
            for key, labels in (
                ("fm_labels", [0, 2]),
                ("fm_reversed", [2, 0]),
                ("fm_unordered", [1, 0, 2]),
            )
        }


class LateStudy(MixedStudy):
    """A concentration at 60 hr, after the end of the simulation at 48 hr."""

    def datasets(self) -> dict[str, DataSet]:
        sets = super().datasets()
        df = pd.DataFrame({"time": [24.0, 60.0], "C": [0.1, 0.01]})
        sets["tc"] = DataSet.from_df(
            df, udict={"time": "hr", "C": "mg/l"}, ureg=self.ureg
        )
        return sets


class SimulationsStudy(ScalarStudy):
    """The cmax of a point of a scan over simulations, which a fit refuses."""

    def simulations(self) -> dict[str, Simulation | Scan]:
        scan = Scan(
            dosed(), [Dimension("dose", simulations={"low": dosed(), "high": dosed()})]
        )
        return {"single": dosed(), "doses": scan}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm": FitMapping(
                self,
                reference=FitData(self, dataset="tab", xid=None, yid="cmax"),
                observable=FitData(
                    self,
                    task="task_doses",
                    xid=None,
                    yid="pk.cmax",
                    sel={"dose": "low"},
                ),
            )
        }


class ThalfStudy(ScalarStudy):
    """The half-life of a simulation of 12 hr.

    A slow absorption has its peak after the end, so the analysis finds no
    terminal phase and the half-life has no value.
    """

    def simulations(self) -> dict[str, Simulation | Scan]:
        short = Simulation(
            end=12, steps=120, changes=[Change(0, {"PODOSE": Q(100, "mg")})]
        )
        return {"single": short, "doses": dose_scan()}

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame({"thalf": [3.5]})
        return {"tab": DataSet.from_df(df, udict={"thalf": "hr"}, ureg=self.ureg)}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_thalf": FitMapping(
                self,
                reference=FitData(self, dataset="tab", xid=None, yid="thalf"),
                observable=FitData(self, task="task_single", xid=None, yid="pk.thalf"),
            )
        }


def peak_below_20_litre(time: np.ndarray, values: dict[str, Any]) -> float:
    """The peak of [C], which fails for a volume above 20 l."""
    if values["V"][0] > 20.0:
        raise ValueError("the volume is too large")
    return float(np.max(values["[C]"]))


class CustomStudy(ScalarStudy):
    """The cmax of 100 mg against an observable which fails for large volumes."""

    def observables(self) -> dict[str, Observable]:
        peak = Custom("peak", peak_below_20_litre, "mg/l", symbols=["[C]", "V"])
        return {**OBSERVABLES, "peak": peak}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_peak": FitMapping(
                self,
                reference=FitData(
                    self, dataset="tab", xid=None, yid="cmax", yid_sd="cmax_sd"
                ),
                observable=FitData(self, task="task_single", xid=None, yid="peak"),
            )
        }


class RateStudy(DoseStudy):
    """The cmax over a scan of the elimination, whose rate a hook reads."""

    RATES: ClassVar[list[float]] = [0.1, 0.2, 0.4]

    def simulations(self) -> dict[str, Simulation | Scan]:
        rates = Dimension("rate", values={"ke": Q(self.RATES, "1/hr")})
        return {"single": dosed(), "doses": Scan(dosed(), [rates])}

    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame({"ke": self.RATES, "cmax": np.full(3, CMAX[1])})
        units = {"ke": "1/hr", "cmax": CMAX_UNIT}
        return {"tab_rates": DataSet.from_df(df, udict=units, ureg=self.ureg)}

    def fit_mappings(self) -> dict[str, FitMapping]:
        return {
            "fm_rate": FitMapping(
                self,
                reference=FitData(self, dataset="tab_rates", xid="ke", yid="cmax"),
                observable=FitData(
                    self, task="task_doses", xid="rate.ke", yid="pk.cmax"
                ),
            )
        }


class GramStudy(DoseStudy):
    """The doses of the data in g, which round below the doses of the scan in mg."""

    def simulations(self) -> dict[str, Simulation | Scan]:
        doses = Dimension("dose", values={"PODOSE": Q([1001.0, 2002.0, 4004.0], "mg")})
        return {"single": dosed(), "doses": Scan(dosed(), [doses])}

    def datasets(self) -> dict[str, DataSet]:
        grams = np.array([1.001, 2.002, 4.004])
        cmax = CMAX[1] * grams * 10.0
        df = pd.DataFrame({"dose": grams, "cmax": cmax, "cmax_sd": 0.05 * cmax})
        units = {"dose": "g", "cmax": CMAX_UNIT}
        return {"tab_doses": DataSet.from_df(df, udict=units, ureg=self.ureg)}


class BlowupStudy(ScalarStudy):
    """The cmax of a model whose integration fails for `kb = 1`."""

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"m": AbstractModel(source=sbml_pk(blowup=True))}


def test_a_value_per_simulation_is_fitted(tmp_path: Path) -> None:
    problem = _problem(ScalarStudy, ["fm_cmax"], tmp_path)
    assert problem.observation_kinds == ["scalar"]
    assert problem.observation_dims == [None]
    assert problem.xid_observable == [None]
    assert np.all(np.isnan(problem.x_references[0]))
    assert problem.cost_least_square(np.array([TRUE_V])) == pytest.approx(
        0.0, abs=1e-12
    )
    assert problem.cost_least_square(np.array([2 * TRUE_V])) > 0.1


def test_values_over_a_dimension_are_fitted(tmp_path: Path) -> None:
    problem = _problem(DoseStudy, ["fm_dose"], tmp_path)
    assert problem.observation_kinds == ["dimension"]
    assert problem.observation_dims == ["dose"]
    assert problem.xid_observable == ["dose.PODOSE"]
    assert problem.cost_least_square(np.array([TRUE_V])) == pytest.approx(
        0.0, abs=1e-10
    )
    fits, _ = problem.optimize(size=1, seed=1, max_nfev=30)
    assert abs(float(fits[0].x[0]) - TRUE_V) / TRUE_V < 1e-3


def test_the_complete_data_of_a_dimension_is_the_curve_over_its_values(
    tmp_path: Path,
) -> None:
    problem = _problem(DoseStudy, ["fm_dose"], tmp_path)
    data = problem.residuals(np.array([TRUE_V]), complete_data=True)
    assert isinstance(data, dict)
    np.testing.assert_allclose(data["x_obs"][0], [50.0, 100.0, 200.0])
    np.testing.assert_allclose(data["y_obs"][0], CMAX, rtol=1e-8)
    np.testing.assert_allclose(data["y_obsip"][0], CMAX, rtol=1e-8)


def test_several_rows_compare_with_one_value(tmp_path: Path) -> None:
    problem = _problem(RowsStudy, ["fm_rows"], tmp_path)
    data = problem.residuals(np.array([TRUE_V]), complete_data=True)
    assert isinstance(data, dict)
    np.testing.assert_allclose(data["y_obsip"][0], np.full(3, CMAX[1]))
    np.testing.assert_allclose(
        data["res_abs"][0], CMAX[1] * (1 - np.array([0.9, 1.0, 1.2])), atol=1e-12
    )
    assert np.isnan(data["x_obs"][0]).all() and data["x_obs"][0].shape == (1,)
    np.testing.assert_allclose(data["y_obs"][0], [CMAX[1]])


def _count_simulations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the simulations of the fit, which the groups run through `execute`."""
    calls: list[int] = []
    execute = optimization_module.execute

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return execute(*args, **kwargs)

    monkeypatch.setattr(optimization_module, "execute", counted)
    return calls


def test_a_timecourse_and_a_scalar_of_one_task_share_a_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = _problem(MixedStudy, ["fm_tc", "fm_cmax"], tmp_path)
    assert sorted(problem.observation_kinds) == ["scalar", "timecourse"]
    assert len(problem.mapping_groups) == 1
    calls = _count_simulations(monkeypatch)
    assert problem.cost_least_square(np.array([TRUE_V])) == pytest.approx(0.0, abs=1e-8)
    assert len(calls) == 1


def test_values_over_a_dimension_cost_one_simulation_per_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = _problem(DoseStudy, ["fm_dose"], tmp_path)
    calls = _count_simulations(monkeypatch)
    problem.residuals(np.array([TRUE_V]))
    assert len(calls) == 3


def test_baseline_residuals_raise_for_a_scalar(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="baseline"):
        _problem(
            ScalarStudy,
            ["fm_cmax"],
            tmp_path,
            residual=ResidualType.ABSOLUTE_TO_BASELINE,
        )


def test_a_dose_outside_the_scan_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"400"):
        _problem(OutsideStudy, ["fm_dose"], tmp_path)


def test_a_failed_observable_gives_the_failure_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = _problem(ScalarStudy, ["fm_cmax"], tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CVODE failed")

    monkeypatch.setattr(optimization_module, "execute", fail)
    residuals = problem.residuals(np.array([TRUE_V]))
    assert isinstance(residuals, np.ndarray)
    np.testing.assert_allclose(residuals, 5.0 * problem.y_references[0])


def test_a_failed_integration_gives_the_failure_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the integrator of the model, created when it loads, does not print the
    # messages of SUNDIALS about the blowup, see `worker.quiet_sundials`
    for name in SUNDIALS_LOGS:
        monkeypatch.setenv(name, os.devnull)
    kb = FitParameter(pid="kb", lower_bound=0.0, upper_bound=2.0, start_value=0.5)
    problem = _problem(BlowupStudy, ["fm_cmax"], tmp_path, parameter=kb)
    working = problem.residuals(np.array([0.0]))
    failed = problem.residuals(np.array([1.0]))
    assert isinstance(working, np.ndarray) and isinstance(failed, np.ndarray)
    np.testing.assert_allclose(working, 0.0, atol=1e-4)
    np.testing.assert_allclose(failed, 5.0 * problem.y_references[0])


def test_a_parameter_without_value_gives_the_failure_residual(tmp_path: Path) -> None:
    ka = FitParameter(
        pid="ka", lower_bound=1e-4, upper_bound=10.0, start_value=1.0, unit="1/hr"
    )
    problem = _problem(ThalfStudy, ["fm_thalf"], tmp_path, parameter=ka)
    working = problem.residuals(np.array([1.0]))
    failed = problem.residuals(np.array([1e-3]))
    assert isinstance(working, np.ndarray) and isinstance(failed, np.ndarray)
    assert np.isfinite(working).all() and abs(working[0]) < 0.1
    np.testing.assert_allclose(failed, 5.0 * problem.y_references[0])
    data = problem.residuals(np.array([1e-3]), complete_data=True)
    assert isinstance(data, dict)
    np.testing.assert_allclose(data["res_abs"][0], 5.0 * problem.y_references[0])
    assert np.isnan(data["y_obsip"][0]).all()
    # the optimizer starts where the half-life has no value and continues
    fit, _ = problem.optimize_run(x0=np.array([1e-3]), max_nfev=5)
    assert fit.status != -1, fit.message


def test_an_observable_which_fails_gives_the_failure_residual(tmp_path: Path) -> None:
    problem = _problem(CustomStudy, ["fm_peak"], tmp_path)
    working = problem.residuals(np.array([TRUE_V]))
    failed = problem.residuals(np.array([30.0]))
    assert isinstance(working, np.ndarray) and isinstance(failed, np.ndarray)
    np.testing.assert_allclose(working, 0.0, atol=1e-9)
    np.testing.assert_allclose(failed, 5.0 * problem.y_references[0])


def test_the_derived_changes_read_the_values_of_a_point(tmp_path: Path) -> None:
    problem = OptimizationProblem(
        "scalar",
        [FitMappingCollection(experiment=RateStudy, mappings=["fm_rate"])],
        [FitParameter(pid="V", lower_bound=1.0, upper_bound=100.0, unit="l")],
        base_path=tmp_path,
        data_path=tmp_path,
        hybridizations=[Scaling(model="m", target="ka", factor="ke")],
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    for point, ke in enumerate(RateStudy.RATES):
        plan = problem.evaluated_plan(0, np.array([TRUE_V]), point=point)
        preinit = {a.target: a.value for a in plan.preinit}
        # the hook sets `ka = ke * ka` from the elimination of the point
        assert preinit["ke"] == pytest.approx(ke)
        assert preinit["ka"] == pytest.approx(ke * 1.0)
    # the simulation of the points is the scan with the derived absorption
    data = problem.residuals(np.array([TRUE_V]), complete_data=True)
    assert isinstance(data, dict)
    rates = Q(RateStudy.RATES, "1/hr")
    scan = Scan(dosed(), [Dimension("rate", values={"ke": rates, "ka": rates})])
    expected = Simulator().run(sbml_pk(), scan, list(OBSERVABLES.values()))
    np.testing.assert_allclose(data["y_obs"][0], expected["pk.cmax"].values, rtol=1e-5)


def test_a_parameter_whose_target_a_dimension_sets_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The values of the dimension win, the fitted value has no effect."""
    ke = FitParameter(
        pid="ke", lower_bound=0.01, upper_bound=1.0, start_value=0.3, unit="1/hr"
    )
    problem = _problem(RateStudy, ["fm_rate"], tmp_path, parameter=ke)
    [record] = [r for r in caplog.records if "no effect" in r.getMessage()]
    assert record.levelname == "WARNING"
    message = record.getMessage()
    assert "'ke'" in message and "'rate'" in message
    assert "RateStudy" in message
    # the cost does not depend on the parameter
    assert problem.cost_least_square(np.array([0.05])) == pytest.approx(
        problem.cost_least_square(np.array([0.9]))
    )


def test_a_parameter_of_another_target_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _problem(RateStudy, ["fm_rate"], tmp_path)
    assert not [r for r in caplog.records if "no effect" in r.getMessage()]


def test_a_reference_at_an_end_of_a_dimension_matches_after_rounding(
    tmp_path: Path,
) -> None:
    problem = _problem(GramStudy, ["fm_dose"], tmp_path)
    assert problem.x_references[0][0] == 1001.0
    np.testing.assert_allclose(problem.x_references[0], [1001.0, 2002.0, 4004.0])
    residuals = problem.residuals(np.array([TRUE_V]))
    assert isinstance(residuals, np.ndarray)
    np.testing.assert_allclose(residuals / problem.y_references[0], 0.0, atol=1e-4)


def test_a_mapping_over_two_dimensions_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sel=") as err:
        _problem(TwoDimStudy, ["fm_dose"], tmp_path)
    assert "'dose'" in str(err.value) and "'rate'" in str(err.value)


def test_a_timecourse_of_a_selected_point_fits_like_the_simulation(
    tmp_path: Path,
) -> None:
    plain = _problem(SelectedPointStudy, ["fm_tc"], tmp_path)
    scan = _problem(SelectedPointStudy, ["fm_tc_scan"], tmp_path)
    assert scan.observation_kinds == ["timecourse"]
    for v in (5.0, TRUE_V, 30.0):
        assert scan.cost_least_square(np.array([v])) == pytest.approx(
            plain.cost_least_square(np.array([v])), rel=1e-10, abs=1e-14
        )


def test_a_value_per_simulation_of_a_selected_point_fits_like_the_simulation(
    tmp_path: Path,
) -> None:
    plain = _problem(SelectedPointStudy, ["fm_cmax"], tmp_path)
    scan = _problem(SelectedPointStudy, ["fm_cmax_scan"], tmp_path)
    assert scan.observation_kinds == ["scalar"]
    for v in (5.0, TRUE_V, 30.0):
        assert scan.cost_least_square(np.array([v])) == pytest.approx(
            plain.cost_least_square(np.array([v])), rel=1e-10, abs=1e-14
        )


def test_a_scan_without_dimensions_fits_like_its_simulation(tmp_path: Path) -> None:
    plain = _problem(MixedStudy, ["fm_cmax", "fm_tc"], tmp_path)
    scan = _problem(EmptyScanStudy, ["fm_cmax", "fm_tc"], tmp_path)
    assert scan.observation_kinds == ["scalar", "timecourse"]
    assert scan.scans == [None, None]
    for v in (5.0, TRUE_V, 30.0):
        assert scan.cost_least_square(np.array([v])) == pytest.approx(
            plain.cost_least_square(np.array([v])), rel=1e-10, abs=1e-14
        )


def test_an_empty_selection_of_rows_names_the_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no row") as err:
        _problem(EmptyRowsStudy, ["fm_empty"], tmp_path)
    assert str(err.value).startswith("EmptyRowsStudy.fm_empty: ")


def test_labels_select_the_points_of_a_dimension(tmp_path: Path) -> None:
    for key, doses in (("fm_labels", [50.0, 200.0]), ("fm_reversed", [200.0, 50.0])):
        problem = _problem(LabelsStudy, [key], tmp_path)
        data = problem.residuals(np.array([TRUE_V]), complete_data=True)
        assert isinstance(data, dict)
        np.testing.assert_allclose(data["x_obs"][0], doses)
        np.testing.assert_allclose(data["res_abs"][0], 0.0, atol=1e-12)


def test_values_of_a_dimension_which_are_not_monotonic_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="monotonic"):
        _problem(LabelsStudy, ["fm_unordered"], tmp_path)


def test_a_time_outside_the_output_of_a_group_with_observables_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"\[60\.0\]"):
        _problem(LateStudy, ["fm_tc", "fm_cmax"], tmp_path)


def test_mappings_of_one_point_of_a_scan_share_a_group(tmp_path: Path) -> None:
    problem = _problem(SelectedPointStudy, ["fm_tc_scan", "fm_cmax_scan"], tmp_path)
    assert problem.mapping_groups == [[0, 1]]
    assert problem.cost_least_square(np.array([TRUE_V])) == pytest.approx(0.0, abs=1e-8)


def test_mappings_of_one_task_with_other_selections_are_other_groups(
    tmp_path: Path,
) -> None:
    problem = _problem(SelectedPointStudy, ["fm_tc_scan", "fm_tc_scan_low"], tmp_path)
    assert problem.mapping_groups == [[0], [1]]
    plain = _problem(SelectedPointStudy, ["fm_tc", "fm_tc_scan"], tmp_path)
    assert plain.mapping_groups == [[0], [1]]


def test_a_scan_over_simulations_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"SimulationsStudy\.fm: .*all set values"):
        _problem(SimulationsStudy, ["fm"], tmp_path)


def test_an_observable_of_the_wrong_kind_raises(tmp_path: Path) -> None:
    class TimecourseOfCmax(ScalarStudy):
        def fit_mappings(self) -> dict[str, FitMapping]:
            return {
                "fm": FitMapping(
                    self,
                    reference=FitData(self, dataset="tab", xid="cmax", yid="cmax"),
                    observable=FitData(
                        self, task="task_single", xid="time", yid="pk.cmax"
                    ),
                )
            }

    with pytest.raises(ValueError, match="value per simulation"):
        _problem(TimecourseOfCmax, ["fm"], tmp_path)


def test_a_parallel_scalar_fit_equals_the_serial_fit(tmp_path: Path) -> None:
    """The workers build the groups, observables and points in `initialize`."""

    def problem() -> OptimizationProblem:
        return OptimizationProblem(
            "scalar",
            [FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"])],
            [
                FitParameter(
                    pid="V",
                    lower_bound=1.0,
                    upper_bound=100.0,
                    start_value=30.0,
                    unit="l",
                )
            ],
            base_path=tmp_path,
            data_path=tmp_path,
        )

    arguments: dict[str, Any] = {
        "settings": FitSettings(parameter_scale=ParameterScaleType.LINEAR),
        "size": 2,
        "seed": 1234,
        "show_progress": False,
        "max_nfev": 20,
    }
    serial = run_optimization(problem(), serial=True, **arguments)
    parallel = run_optimization(problem(), n_cores=2, **arguments)
    np.testing.assert_allclose(parallel.xopt, serial.xopt, rtol=1e-6)
    np.testing.assert_allclose(serial.xopt, [TRUE_V], rtol=1e-3)
