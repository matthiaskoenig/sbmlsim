"""The fit of a problem whose simulation groups are scans in the engine (#624).

A group of a scan is a condition of the request per point of the scan, the
plan of the point with the values of the parameters applied first, and a
mapping over a dimension of the scan is a curve per point its data is at. The
residuals of the request are the ones of the problem point for point, as for
the other kinds of a request in `tests/fit/test_fit_core.py`.
"""

import dataclasses
from pathlib import Path
from typing import ClassVar, override

import numpy as np
import pandas as pd
import pytest

from sbmlsim import Q
from sbmlsim.data import DataSet
from sbmlsim.engine import Engine
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import (
    FitData,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    fit_core,
)
from sbmlsim.fit.fit_core import fit_request, gaps_of
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import RUN_IN_PYTHON, RUN_IN_THE_ENGINE, ObjectiveType
from sbmlsim.fit.runner import run_optimization
from sbmlsim.simulation import Dimension, Formula, Observable, Scan, Simulation
from sbmlsim.task import Task
from tests.fit.scalar_experiment import (
    CONC,
    DOSES,
    TIME,
    TRUE_V,
    DoseStudy,
    OneDoseStudy,
    dosed,
)
from tests.fit.test_fit_core import core_problem, linear_settings, on_core
from tests.fit.test_scalar_mappings import (
    GramStudy,
    LabelsStudy,
    RateStudy,
    SelectedPointStudy,
)

#: the doses of a dimension with a time, a second dose at `LATE`
LATE = 24.0


class ScanStudy(DoseStudy):
    """The scan over the doses with reductions and a dimension with a time.

    `c_at` and `c_max` are values per simulation of a formula of one
    reduction over the dimension of the doses; `second` scans a second dose
    at 24 hr, a change of every point, whose timecourse reads the state
    before and after it at the time of the change.
    """

    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        second = Dimension("second", values={"PODOSE": Q([50.0, 100.0], "mg")}, at=LATE)
        return {**super().simulations(), "second": Scan(dosed(), [second])}

    @override
    def observables(self) -> dict[str, Observable]:
        return {
            **super().observables(),
            "c_at": Formula("c_at", "at([C], 6)", unit="mg/l"),
            "c_max": Formula("c_max", "max([C])", unit="mg/l"),
        }

    @override
    def datasets(self) -> dict[str, DataSet]:
        datasets = super().datasets()
        doses = np.asarray(DOSES)
        datasets["reduced"] = DataSet.from_df(
            pd.DataFrame({"dose": doses, "at": 0.004 * doses, "max": 0.006 * doses}),
            udict={"dose": "mg", "at": "mg/l", "max": "mg/l"},
            ureg=self.ureg,
        )
        datasets["tc"] = DataSet.from_df(
            pd.DataFrame({"time": TIME[::40], "C": CONC[::40]}),
            udict={"time": "hr", "C": "mg/l"},
            ureg=self.ureg,
        )
        # from the second dose on, whose first point reads the state after it
        datasets["tc_late"] = DataSet.from_df(
            pd.DataFrame({"time": TIME[240::40], "C": CONC[240::40]}),
            udict={"time": "hr", "C": "mg/l"},
            ureg=self.ureg,
        )
        return datasets

    @override
    def tasks(self) -> dict[str, Task]:
        return {**super().tasks(), "task_second": Task(model="m", simulation="second")}

    @override
    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings = super().fit_mappings()
        for key, yid, observable in (
            ("fm_dose_at", "at", "c_at"),
            ("fm_dose_max", "max", "c_max"),
        ):
            mappings[key] = FitMapping(
                self,
                reference=FitData(self, dataset="reduced", xid="dose", yid=yid),
                observable=FitData(
                    self, task="task_doses", xid="dose.PODOSE", yid=observable
                ),
            )
        for key, dataset in (("fm_second_tc", "tc"), ("fm_second_late", "tc_late")):
            mappings[key] = FitMapping(
                self,
                reference=FitData(self, dataset=dataset, xid="time", yid="C"),
                observable=FitData(
                    self, task="task_second", xid="time", yid="[C]", sel={"second": 1}
                ),
            )
        return mappings


class BetweenStudy(DoseStudy):
    """A dose of 75 mg, between the points of 50 and 100 mg of the scan."""

    REF_DOSES: ClassVar[list[float]] = [50.0, 75.0, 200.0]


class RateAtStudy(RateStudy):
    """The scan of the elimination from 12 hr on, a change of every point."""

    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        rates = Dimension("rate", values={"ke": Q(self.RATES, "1/hr")}, at=12.0)
        return {"single": dosed(), "doses": Scan(dosed(), [rates])}


#: the elimination rate, a parameter of the fit whose target a scan sets
KE = FitParameter(
    pid="ke", lower_bound=0.01, upper_bound=1.0, start_value=0.3, unit="1/hr"
)


def scan_problem(
    path: Path,
    experiment: type[SimulationExperiment],
    mappings: list[str],
    parameter: FitParameter | None = None,
) -> OptimizationProblem:
    """Get the problem of mappings of a study of the scans over the doses."""
    problem = OptimizationProblem(
        "scans",
        [FitMappingCollection(experiment=experiment, mappings=mappings)],
        [
            parameter
            or FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=path,
        data_path=path,
        engine=Engine.CORE,
    )
    return on_core(problem, linear_settings())


#: the study, the mappings and the parameter of a problem of scan groups, the
#: number of the conditions of its request and the point of the condition of
#: every curve, in the order of the curves
SCANS: dict[
    str,
    tuple[type[SimulationExperiment], list[str], FitParameter | None, int, list[int]],
] = {
    # the cmax of a PK observable over the doses, the dose of every point in
    # the batch of its analysis
    "dose": (DoseStudy, ["fm_dose"], None, 3, [0, 1, 2]),
    # one point of the scan, selected by its label
    "one": (OneDoseStudy, ["fm_one"], None, 1, [0]),
    # points selected by labels, in their order and reversed, which are other
    # groups of the same task
    "labels": (LabelsStudy, ["fm_labels", "fm_reversed"], None, 4, [0, 1, 1, 0]),
    # a timecourse of a point and a value per simulation of the same point,
    # and a timecourse of another point, which outputs the times of its data
    "points": (
        SelectedPointStudy,
        ["fm_tc_scan", "fm_tc_scan_low", "fm_cmax_scan"],
        None,
        2,
        [0, 0, 0],
    ),
    # a dimension of another target than the parameter of the fit
    "rate": (RateStudy, ["fm_rate"], None, 3, [0, 1, 2]),
    # formulas of one reduction over the doses, which share the conditions of
    # their points
    "reductions": (ScanStudy, ["fm_dose_at", "fm_dose_max"], None, 3, [0, 1, 2] * 2),
    # timecourses of a point whose dimension is a change at 24 hr, one of
    # which starts at the change and reads the state after it
    "change": (ScanStudy, ["fm_second_tc", "fm_second_late"], None, 1, [0, 0]),
    # a dimension with a time which sets the target of the parameter of the
    # fit: the change of the point wins from its time on, as in python
    "rate_at": (RateAtStudy, ["fm_rate"], KE, 3, [0, 1, 2]),
}


@pytest.mark.parametrize("case", list(SCANS))
def test_the_residuals_of_a_scan_are_the_ones_of_the_problem(
    tmp_path: Path, case: str
) -> None:
    """A scan group is a condition per point and its residuals are the ones of python.

    Bitwise at three points: the condition of a point is the plan python
    simulates the point with, the values of the point applied after the
    values of the parameters, and the curve of a measurement over a
    dimension reads the point it is at.
    """
    experiment, mappings, parameter, conditions, points = SCANS[case]
    problem = scan_problem(tmp_path, experiment, mappings, parameter)

    assert gaps_of(problem) == []

    request = fit_request(problem)
    assert len(request["conditions"]) == conditions
    assert [
        int(curve["condition"].rsplit("|", 1)[1]) for curve in request["curves"]
    ] == points
    core = core_problem(problem)
    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        np.testing.assert_array_equal(residuals, expected)


def test_a_point_after_the_change_of_a_dimension_reads_the_state_after_it(
    tmp_path: Path,
) -> None:
    """The first point of a curve at the change of a dimension, see `after_change_of`.

    The change is one of the plan of the point, which the plan of the group
    does not have.
    """
    problem = scan_problem(tmp_path, ScanStudy, ["fm_second_late"])

    [curve] = fit_request(problem)["curves"]

    assert curve["times"][0] == LATE
    assert curve["after_change"] == [True] + [False] * (len(curve["times"]) - 1)


def test_data_at_a_point_within_the_rounding_of_its_unit_reads_the_point(
    tmp_path: Path,
) -> None:
    """1.001 g is 1000.9999999999999 mg, the point of 1001 mg of the scan.

    Python interpolates between the points at the x of the data, which is the
    value of the point to the slope over the rounding of the unit.
    """
    problem = scan_problem(tmp_path, GramStudy, ["fm_dose"])

    assert gaps_of(problem) == []

    core = core_problem(problem)
    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        np.testing.assert_allclose(
            np.asarray(core.residuals(x), dtype=float),
            np.asarray(problem.residuals(x), dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )


def test_the_curves_over_a_dimension_are_one_per_point(tmp_path: Path) -> None:
    """A curve per measurement at a point, numbered by its place in the data."""
    problem = scan_problem(tmp_path, DoseStudy, ["fm_dose"])

    request = fit_request(problem)

    assert [curve["id"] for curve in request["curves"]] == [
        "fm_dose[0:1]",
        "fm_dose[1:2]",
        "fm_dose[2:3]",
    ]
    assert [condition["id"] for condition in request["conditions"]] == [
        f"DoseStudy|doses|0|{point}" for point in range(3)
    ]
    for point, curve in enumerate(request["curves"]):
        assert curve["reduction"]["kind"] == "pk"
        # the dose of the point is the dose of the batch of its analysis
        assert curve["reduction"]["batch"]["dose_amount"] == [DOSES[point]]
    np.testing.assert_array_equal(
        fit_core._residual_rows(problem), np.arange(len(DOSES))
    )


def test_a_dimension_which_sets_the_target_of_a_parameter_is_a_gap(
    tmp_path: Path,
) -> None:
    """The dimension wins in python and the parameter of a condition in the engine."""
    problem = scan_problem(tmp_path, RateStudy, ["fm_rate"], KE)

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == ["scan"]
    assert "'ke'" in gaps[0].message


def test_data_between_the_points_of_a_scan_is_a_gap(tmp_path: Path) -> None:
    """75 mg is interpolated between the points of 50 and 100 mg."""
    problem = scan_problem(tmp_path, BetweenStudy, ["fm_dose"])

    gaps = gaps_of(problem)

    assert [gap.code for gap in gaps] == ["scan"]
    assert "'fm_dose'" in gaps[0].message


def test_the_jacobian_of_a_scan_is_the_derivative_of_its_residuals(
    tmp_path: Path,
) -> None:
    """The forward sensitivities of the conditions of the points of a scan."""
    from sbmlsim.fit.fisher import jacobian

    problem = scan_problem(tmp_path, ScanStudy, ["fm_dose_at", "fm_second_tc"])
    x = problem.to_scale(problem.xmodel)

    sensitivities = fit_core.residual_jacobian(problem, x, threads=2)

    differences = jacobian(problem, x, step=1e-4)
    assert sensitivities.shape == differences.shape
    np.testing.assert_allclose(sensitivities, differences, rtol=1e-3, atol=1e-6)
    assert np.linalg.norm(sensitivities) > 0.0


def test_a_fit_of_a_scan_in_the_engine_finds_the_optimum_of_python(
    tmp_path: Path,
) -> None:
    """The engine and python fit the cmax over the doses to the same optimum."""
    results = {}
    for engine_fit in (True, False):
        (tmp_path / str(engine_fit)).mkdir()
        problem = scan_problem(tmp_path / str(engine_fit), DoseStudy, ["fm_dose"])
        results[engine_fit] = run_optimization(
            problem,
            problem.settings_initialized,
            size=1,
            seed=1234,
            show_progress=False,
            engine_fit=engine_fit,
        )

    assert results[True].optimizer_of == RUN_IN_THE_ENGINE
    assert results[False].optimizer_of == RUN_IN_PYTHON
    engine, python = results[True].fits[0], results[False].fits[0]
    np.testing.assert_allclose(engine.x, python.x, rtol=1e-6)
    assert engine.cost == pytest.approx(python.cost, rel=1e-8, abs=1e-12)
    np.testing.assert_allclose(engine.x, [TRUE_V], rtol=1e-3)


def test_the_residuals_of_a_likelihood_over_a_dimension_are_the_ones_of_python(
    tmp_path: Path,
) -> None:
    """A curve per point carries the data and the scale rows of its measurements.

    `_residual_rows` places the two rows of every measurement of a curve
    among the residuals of the problem, which has the data of the mapping
    first and the scales after it.
    """
    problem = OptimizationProblem(
        "scans",
        [FitMappingCollection(experiment=DoseStudy, mappings=["fm_dose"])],
        [
            FitParameter(
                pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
        engine=Engine.CORE,
    )
    problem = on_core(
        problem,
        dataclasses.replace(linear_settings(), objective=ObjectiveType.LIKELIHOOD),
    )

    assert gaps_of(problem) == []

    core = core_problem(problem)
    rows = fit_core._residual_rows(problem)
    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        np.testing.assert_allclose(residuals, expected[rows], rtol=1e-14, atol=1e-14)
