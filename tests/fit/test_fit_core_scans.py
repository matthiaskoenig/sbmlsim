"""The fit of a problem whose simulation groups are scans in the engine (#624).

A group of a scan is a condition of the request per point of the scan, the
plan of the point with the values of the parameters applied first, and a
mapping over a dimension of the scan is a curve per point its data is at, or
one curve over the points of the dimension for data between them. A point
whose dimension sets the target of a parameter of the fit pins it. The
residuals of the request are the ones of the problem point for point, as for
the other kinds of a request in `tests/fit/test_fit_core.py`.
"""

import dataclasses
import functools
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
    FitSettings,
    fit_core,
)
from sbmlsim.fit.fit_core import fit_request, gaps_of
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import RUN_IN_PYTHON, RUN_IN_THE_ENGINE, ObjectiveType
from sbmlsim.fit.runner import run_optimization
from sbmlsim.simulation import Dimension, Formula, Observable, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.fit.scalar_experiment import (
    CMAX_UNIT,
    CONC,
    DOSES,
    OBSERVABLES,
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
from tests.simulator.models import sbml_pk

#: the doses of a dimension with a time, a second dose at `LATE`
LATE = 24.0

#: the relative tolerance of the residuals of the engine against python: a
#: few ulps, since the engine and numpy compute the same operations, but the
#: compiler or the libm of another platform may round them differently, e.g.
#: a multiplication and an addition fused into one (1 ulp on macos-latest)
ULPS = 1e-14


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


class BetweenAtStudy(ScanStudy):
    """`at` and `max` formulas over the doses, measured between the points.

    The reduction of a formula is the one of every point, where the batch of a
    `PK` analysis carries the dose of its point.
    """

    @override
    def datasets(self) -> dict[str, DataSet]:
        datasets = super().datasets()
        doses = np.array([60.0, 100.0, 150.0, 199.0])
        datasets["reduced"] = DataSet.from_df(
            pd.DataFrame({"dose": doses, "at": 0.004 * doses, "max": 0.006 * doses}),
            udict={"dose": "mg", "at": "mg/l", "max": "mg/l"},
            ureg=self.ureg,
        )
        return datasets


class RateBetweenStudy(RateStudy):
    """The cmax at elimination rates between the points of the scan of the rate."""

    @override
    def datasets(self) -> dict[str, DataSet]:
        rates = [0.15, 0.2, 0.3]
        df = pd.DataFrame({"ke": rates, "cmax": [3.0, 2.5, 2.0]})
        units = {"ke": "1/hr", "cmax": "mg/l"}
        return {"tab_rates": DataSet.from_df(df, udict=units, ureg=self.ureg)}


class RateAtStudy(RateStudy):
    """The scan of the elimination from 12 hr on, a change of every point."""

    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        rates = Dimension("rate", values={"ke": Q(self.RATES, "1/hr")}, at=12.0)
        return {"single": dosed(), "doses": Scan(dosed(), [rates])}


def preinit_scan(volume: str) -> Scan:
    """Get the scan of the elimination rate whose volume a formula sets first."""
    rates = Dimension("rate", values={"ke": Q(RateStudy.RATES, "1/hr")})
    preinit = Simulation(
        end=48, steps=480, changes=dosed().changes, preinit_changes={"V": volume}
    )
    return Scan(preinit, [rates])


class RatePreinitStudy(RateStudy):
    """The scan of the elimination with a volume which a formula of the rate sets.

    The formula is evaluated before the simulation, after the rate of the
    point, which it reads.
    """

    #: the formula of the volume
    VOLUME: ClassVar[str] = "100 * ke"

    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        return {"single": dosed(), "doses": preinit_scan(self.VOLUME)}


class RateSumPreinitStudy(RatePreinitStudy):
    """A volume of the rate of the point and of the absorption, a parameter of the fit."""

    VOLUME: ClassVar[str] = "50 * ke + 5 * ka"


class RateProductPreinitStudy(RatePreinitStudy):
    """A volume of the product of the rate and the absorption.

    No linear form over the parameters of the fit, but one over the
    absorption at every point, where the rate is the number of the point.
    """

    VOLUME: ClassVar[str] = "100 * ke * ka"


@functools.cache
def _product_cmax() -> list[float]:
    """Get the cmax of `RateProductPreinitStudy` at every rate, at the absorption 1/hr.

    The integrator settings are the ones of a fit with the default settings,
    so a fit of the absorption to these values has the residuals 0 at 1/hr.
    """
    settings = FitSettings()
    simulator = Simulator(
        absolute_tolerance=settings.absolute_tolerance,
        relative_tolerance=settings.relative_tolerance,
        variable_step_size=settings.variable_step_size,
        initial_time_step=settings.initial_time_step,
    )
    res = simulator.run(
        sbml_pk(),
        preinit_scan(RateProductPreinitStudy.VOLUME),
        list(OBSERVABLES.values()),
        keep=["pk.cmax"],
    )
    return [float(value) for value in res["pk.cmax"].values]


class RateProductFitStudy(RateProductPreinitStudy):
    """The cmax of the product of the rate and the absorption at 1/hr.

    The data the model has at the absorption 1/hr, so a fit of the absorption
    ends at 1/hr with the residuals 0: the optimum of a least squares fit
    does not depend on the precision of the jacobian an optimizer reads.
    """

    @override
    def datasets(self) -> dict[str, DataSet]:
        df = pd.DataFrame({"ke": self.RATES, "cmax": _product_cmax()})
        units = {"ke": "1/hr", "cmax": CMAX_UNIT}
        return {"tab_rates": DataSet.from_df(df, udict=units, ureg=self.ureg)}


class TwiceStudy(BetweenStudy):
    """The scan over the doses with 50 mg twice."""

    @override
    def simulations(self) -> dict[str, Simulation | Scan]:
        doses = Dimension("dose", values={"PODOSE": Q([50.0, 50.0, 100.0], "mg")})
        return {"single": dosed(), "doses": Scan(dosed(), [doses])}


#: the elimination rate, a parameter of the fit whose target a scan sets
KE = FitParameter(
    pid="ke", lower_bound=0.01, upper_bound=1.0, start_value=0.3, unit="1/hr"
)

#: the absorption rate, a parameter of the fit no scan sets
KA = FitParameter(
    pid="ka", lower_bound=0.1, upper_bound=10.0, start_value=1.5, unit="1/hr"
)


def scan_problem(
    path: Path,
    experiment: type[SimulationExperiment],
    mappings: list[str],
    parameter: FitParameter | list[FitParameter] | None = None,
) -> OptimizationProblem:
    """Get the problem of mappings of a study of the scans over the doses."""
    if parameter is None:
        parameter = FitParameter(
            pid="V", lower_bound=1.0, upper_bound=100.0, start_value=30.0, unit="l"
        )
    problem = OptimizationProblem(
        "scans",
        [FitMappingCollection(experiment=experiment, mappings=mappings)],
        parameter if isinstance(parameter, list) else [parameter],
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

    To a few ulps at three points: the condition of a point is the plan python
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

        np.testing.assert_allclose(residuals, expected, rtol=ULPS, atol=0.0)


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


#: the studies of a formula before the simulation which reads a pinned target,
#: and the constant and the factor of the absorption of the derived volume of
#: the condition of every rate of the scan, none for a formula of the rate alone
PREINIT_CASES: dict[
    str,
    tuple[type[RatePreinitStudy], list[FitParameter], list[tuple[float, float]] | None],
] = {
    # the volume of the rate alone is the number of the point
    "pin": (RatePreinitStudy, [KE], None),
    "pin_and_absorption": (RatePreinitStudy, [KE, KA], None),
    # a linear form over the absorption whose constant is the term of the rate
    "sum": (
        RateSumPreinitStudy,
        [KE, KA],
        [(50.0 * rate, 5.0) for rate in RateStudy.RATES],
    ),
    # the product is a linear form over the absorption at every point, whose
    # factor is the rate of the point
    "product": (
        RateProductPreinitStudy,
        [KE, KA],
        [(0.0, 100.0 * rate) for rate in RateStudy.RATES],
    ),
}


@pytest.mark.parametrize("case", list(PREINIT_CASES))
def test_a_formula_before_the_simulation_reads_the_pinned_target_as_python(
    tmp_path: Path, case: str
) -> None:
    """The formula of the volume reads the rate of the point, as in python.

    Python evaluates the formula after the values of the point, so it reads
    the value of the point and not the parameter of the fit: the condition of
    a point carries the formula with the pin substituted, a number of the
    request where it reads nothing else and a derived value of the other
    parameters where it does.
    """
    experiment, parameters, derived = PREINIT_CASES[case]
    problem = scan_problem(tmp_path, experiment, ["fm_rate"], parameters)

    assert gaps_of(problem) == []

    conditions = fit_request(problem)["conditions"]
    assert [c["pinned"] for c in conditions] == [
        [{"parameter": 0, "value": rate}] for rate in RateStudy.RATES
    ]
    if derived is None:
        assert all("derived" not in c for c in conditions)
    else:
        assert [c["derived"] for c in conditions] == [
            [
                {
                    "target": "V",
                    "constant": pytest.approx(constant, rel=1e-15),
                    "terms": [{"parameter": 1, "factor": pytest.approx(factor)}],
                }
            ]
            for constant, factor in derived
        ]
    core = core_problem(problem)
    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        np.testing.assert_allclose(residuals, expected, rtol=ULPS, atol=0.0)


@pytest.mark.parametrize("case", ["sum", "product"])
def test_the_jacobian_of_a_formula_of_a_pinned_target_is_its_derivative(
    tmp_path: Path, case: str
) -> None:
    """The volume follows the absorption and not the rate, whose column is zero.

    A condition with a derived value has no forward sensitivities in the
    engine, which takes the forward differences of its residuals over its
    `DIFF_STEP` of 1e-3 instead, so the column of the absorption is accurate
    to about 1e-3 of its largest entry (the volume and the absorption itself
    cancel in the cmax of the middle rate of `sum`) and is compared at its
    scale. The rate moves no residual of a condition which pins it, so its
    difference is exactly zero.
    """
    from sbmlsim.fit.fisher import jacobian

    experiment, parameters, _ = PREINIT_CASES[case]
    problem = scan_problem(tmp_path, experiment, ["fm_rate"], parameters)
    x = problem.to_scale(problem.xmodel)

    sensitivities = fit_core.residual_jacobian(problem, x, threads=2)

    differences = jacobian(problem, x, step=1e-4)
    assert sensitivities.shape == differences.shape == (len(RateStudy.RATES), 2)
    np.testing.assert_allclose(
        sensitivities, differences, rtol=0.0, atol=1e-3 * np.abs(differences).max()
    )
    assert np.all(sensitivities[:, 0] == 0.0)
    assert np.all(sensitivities[:, 1] != 0.0)


def test_a_fit_of_a_formula_of_a_pinned_target_finds_the_optimum_of_python(
    tmp_path: Path,
) -> None:
    """The engine and python fit the absorption of the product to one optimum."""
    results = {}
    for engine_fit in (True, False):
        (tmp_path / str(engine_fit)).mkdir()
        problem = scan_problem(
            tmp_path / str(engine_fit), RateProductFitStudy, ["fm_rate"], [KE, KA]
        )
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
    # the absorption of the data; the rate moves no residual and stays at the
    # start of the run, which is the same in both
    assert engine.x[1] == pytest.approx(1.0, rel=1e-4)


def test_a_dimension_with_a_value_twice_is_refused(tmp_path: Path) -> None:
    """50 mg twice is no order `numpy.interp` and a curve over a dimension share.

    The problem refuses a mapping over such a dimension, so data between the
    points of a scan is always a curve over its dimension and never a gap.
    """
    with pytest.raises(ValueError, match="not strictly monotonic"):
        scan_problem(tmp_path, TwiceStudy, ["fm_dose"])


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


#: the cases of the fields of `fit/1` a scan needs beyond a curve of a point:
#: the study, the mappings, the parameter and the points every curve over a
#: dimension reads, in the order of the curves
FIELD_CASES: dict[
    str,
    tuple[type[SimulationExperiment], list[str], FitParameter | None, list],
] = {
    # the cmax of a PK observable at a dose between two points, the batch of
    # the analysis of every point with its own dose
    "between_pk": (BetweenStudy, ["fm_dose"], None, [[0, 1, 2]]),
    # formulas of one reduction over the doses, the same at every point
    "between_reductions": (
        BetweenAtStudy,
        ["fm_dose_at", "fm_dose_max"],
        None,
        [[0, 1, 2]] * 2,
    ),
    # a dimension which sets the target of the parameter of the fit before
    # the simulation, which every point pins
    "pinned": (RateStudy, ["fm_rate"], KE, [None] * 3),
    # both: the rates of the data between the points, which pin the rate
    "pinned_between": (
        RateBetweenStudy,
        ["fm_rate"],
        KE,
        [[0, 1, 2]],
    ),
}


@pytest.mark.parametrize("case", list(FIELD_CASES))
def test_the_residuals_of_the_fields_of_a_scan_are_the_ones_of_the_problem(
    tmp_path: Path, case: str
) -> None:
    """A curve over a dimension and a pinned parameter, the residuals of python.

    The curve over a dimension is `numpy.interp` of the values of the points
    operation for operation, so the residuals agree to a few ulps (`ULPS`),
    and a pinned target holds the value of the point,
    which wins over the parameter in python.
    """
    experiment, mappings, parameter, over = FIELD_CASES[case]
    problem = scan_problem(tmp_path, experiment, mappings, parameter)

    assert gaps_of(problem) == []

    request = fit_request(problem)
    assert [
        [int(c.rsplit("|", 1)[1]) for c in curve["dimension"]["conditions"]]
        if "dimension" in curve
        else None
        for curve in request["curves"]
    ] == over
    pinned = [c for c in request["conditions"] if "pinned" in c]
    assert len(pinned) == (len(request["conditions"]) if parameter else 0)
    core = core_problem(problem)
    for factor in (1.0, 1.3, 0.7):
        x = problem.to_scale(problem.xmodel * factor)
        expected = np.asarray(problem.residuals(x), dtype=float)

        residuals = np.asarray(core.residuals(x), dtype=float)

        np.testing.assert_allclose(residuals, expected, rtol=ULPS, atol=0.0)


def test_a_curve_over_a_dimension_carries_the_reduction_of_every_point(
    tmp_path: Path,
) -> None:
    """The dose of every point is the dose of the batch of its analysis."""
    problem = scan_problem(tmp_path, BetweenStudy, ["fm_dose"])

    [curve] = fit_request(problem)["curves"]

    assert "condition" not in curve
    assert "reduction" not in curve
    dimension = curve["dimension"]
    assert dimension["values"] == DOSES
    assert dimension["x"] == BetweenStudy.REF_DOSES
    assert [r["batch"]["dose_amount"] for r in dimension["reductions"]] == [
        [dose] for dose in DOSES
    ]


def test_a_pin_is_the_value_of_the_point_in_the_units_of_the_model(
    tmp_path: Path,
) -> None:
    """The rate of every point holds the target of `ke`, the parameter of the fit."""
    problem = scan_problem(tmp_path, RateStudy, ["fm_rate"], KE)

    request = fit_request(problem)

    assert [c["pinned"] for c in request["conditions"]] == [
        [{"parameter": 0, "value": rate}] for rate in RateStudy.RATES
    ]


def test_the_jacobian_of_a_curve_over_a_dimension_is_the_derivative_of_its_residuals(
    tmp_path: Path,
) -> None:
    """The interpolation of the forward sensitivities of the points."""
    from sbmlsim.fit.fisher import jacobian

    problem = scan_problem(tmp_path, BetweenAtStudy, ["fm_dose_at", "fm_dose_max"])
    x = problem.to_scale(problem.xmodel)

    sensitivities = fit_core.residual_jacobian(problem, x, threads=2)

    differences = jacobian(problem, x, step=1e-4)
    assert sensitivities.shape == differences.shape
    np.testing.assert_allclose(sensitivities, differences, rtol=1e-3, atol=1e-6)
    assert np.linalg.norm(sensitivities) > 0.0


def test_a_fit_between_the_points_in_the_engine_finds_the_optimum_of_python(
    tmp_path: Path,
) -> None:
    """The engine and python fit the cmax between the doses to the same optimum."""
    results = {}
    for engine_fit in (True, False):
        (tmp_path / str(engine_fit)).mkdir()
        problem = scan_problem(tmp_path / str(engine_fit), BetweenStudy, ["fm_dose"])
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


def test_the_column_of_a_pinned_parameter_is_zero(tmp_path: Path) -> None:
    """Every point holds the rate at its own value, so `ke` moves no residual."""
    problem = scan_problem(tmp_path, RateStudy, ["fm_rate"], KE)
    x = problem.to_scale(problem.xmodel)

    sensitivities = fit_core.residual_jacobian(problem, x, threads=2)

    assert sensitivities.shape == (len(RateStudy.RATES), 1)
    assert np.all(sensitivities == 0.0)
