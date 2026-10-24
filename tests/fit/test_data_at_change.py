"""A fit compares data at the time of a change with the state before the change.

The amount `Aurine` of the PK model with `urine=True` is collected from the
dose at 0 and emptied by a change at 24 h, the end of a collection interval,
so the measured amount at 24 h is the amount collected until then, which a
fit compares with the state before the change (`data_at_change="before"`),
while the results of a run keep the state after it, `0`.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.figure import Figure

from sbmlsim import Q
from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import (
    FitData,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    NoiseModel,
)
from sbmlsim.fit.objects import ObservableModel
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import DataAtChangeType, FitSettings, ParameterScaleType
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.petab_v2 import EXTENSION_VERSION, to_petab
from sbmlsim.fit.petab_v2.gaps import gaps_of_problem, measurements_at_changes
from sbmlsim.fit.petab_v2.reader import PetabReader, from_petab
from sbmlsim.fit.report import FitReport
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import PK, Change, Formula, Observable, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.simulator.models import sbml, sbml_pk
from tests.simulator.test_executor import SAWTOOTH

#: the elimination rate of the model, the value of the fitted parameter
TRUE_KE = 0.2
DOSE = Change(0, {"PODOSE": Q(100, "mg")})
#: the urine is emptied at the end of the first collection interval
RESET = Change(24, {"Aurine": Q(0, "mg")})
TIMES = [12.0, 24.0, 36.0, 48.0]
#: the grid of output times of the simulation `coarse`, without the reset
COARSE = np.linspace(0.0, 48.0, 8)
#: the times of data just before the reset, after the last step of the
#: integrator before it
LATE = [12.0, 23.95, 36.0, 48.0]
#: the relative tolerance of a state of the steps of the integrator, e.g. at
#: the end of a segment, against the state at exact output times
STEPS_RTOL = 1e-6
#: the relative tolerance of data between two steps of the integrator, which
#: a group which keeps its output interpolates linearly
INTERPOLATED_RTOL = 5e-4
#: the amounts eliminated at which the concentration is compared
ELIMINATED = [20.0, 50.0, 90.0, 99.0]
#: the file of the model in the base path of a problem
MODEL_FILE = "urine.xml"
#: below the resolution of the time of the simulation, `1e-9 * 48` hours
NOISE = 1e-8
OBSERVABLES: dict[str, Observable] = {
    "pk": PK("pk", "[C]", dose="PODOSE", route="oral"),
    # values per simulation and a timecourse of the urine, which a fit must
    # compute on the timecourse of a run
    "urine_max": Formula("urine_max", "max(Aurine)"),
    "urine_twice": Formula("urine_twice", "2 * Aurine"),
    # a timecourse which reads a value per simulation, and one with a reduction
    "urine_ratio": Formula("urine_ratio", "Aurine / urine_max"),
    "urine_share": Formula("urine_share", "Aurine / max(Aurine)"),
}


def _simulator() -> Simulator:
    """Get a simulator with the integrator of a fit with the default settings."""
    settings = FitSettings()
    return Simulator(
        absolute_tolerance=settings.absolute_tolerance,
        relative_tolerance=settings.relative_tolerance,
        variable_step_size=settings.variable_step_size,
        initial_time_step=settings.initial_time_step,
    )


def _truth() -> dict[str, Any]:
    """The collected urine at the times of the data and the values of a run.

    The amount at 12 h and 24 h is the one of the first collection, i.e. of a
    simulation without the reset, the one at 36 h and 48 h the one of the
    second collection.
    """
    simulator = _simulator()
    model = simulator.load(sbml_pk(urine=True))
    first = simulator.simulate(
        model, Simulation(end=48, times=TIMES[:2], changes=[DOSE])
    )
    second = simulator.simulate(
        model, Simulation(end=48, times=TIMES[2:], changes=[DOSE, RESET])
    )
    late = simulator.simulate(
        model, Simulation(end=48, times=LATE, changes=[DOSE, RESET])
    )
    run = simulator.run(
        model,
        Simulation(end=48, changes=[DOSE, RESET]),
        list(OBSERVABLES.values()),
        keep=["pk.cmax", "pk.auc_last", "urine_max", "Aurine", "Aelim", "[C]"],
    )
    coarse = simulator.run(
        model,
        Simulation(end=48, steps=7, changes=[DOSE, RESET]),
        list(OBSERVABLES.values()),
        keep=["urine_max"],
    )
    return {
        "urine": np.concatenate([first["Aurine"], second["Aurine"]]),
        "coarse_max": float(coarse["urine_max"].values),
        "late": np.asarray(late["Aurine"], dtype=float),
        "cmax": float(run["pk.cmax"].values),
        "auc": float(run["pk.auc_last"].values),
        "urine_max": float(run["urine_max"].values),
        "after": run,
        "units": {k: run.units[k] for k in ("pk.cmax", "pk.auc_last", "urine_max")},
    }


TRUTH = _truth()


class UrineStudy(SimulationExperiment):
    """The urine of one oral dose, collected in two intervals of 24 h."""

    def datasets(self) -> dict[str, DataSet]:
        urine = pd.DataFrame({"time": TIMES, "Aurine": TRUTH["urine"]})
        units = TRUTH["units"]
        values = pd.DataFrame(
            {
                "cmax": [TRUTH["cmax"]],
                "auc": [TRUTH["auc"]],
                "urine_max": [TRUTH["urine_max"]],
            }
        )
        return {
            "urine": DataSet.from_df(
                urine, udict={"time": "hr", "Aurine": "mg"}, ureg=self.ureg
            ),
            "late": DataSet.from_df(
                pd.DataFrame({"time": LATE, "Aurine": TRUTH["late"]}),
                udict={"time": "hr", "Aurine": "mg"},
                ureg=self.ureg,
            ),
            "twice": DataSet.from_df(
                urine.assign(Aurine=2 * urine["Aurine"]),
                udict={"time": "hr", "Aurine": "mg"},
                ureg=self.ureg,
            ),
            # the concentration over the amount eliminated
            "over": DataSet.from_df(
                pd.DataFrame({"Aelim": ELIMINATED, "C": [1.0] * len(ELIMINATED)}),
                udict={"Aelim": "mg", "C": "mg/l"},
                ureg=self.ureg,
            ),
            # a time which differs from the reset by less than the resolution
            # of the time is at the reset
            "noisy": DataSet.from_df(
                urine.iloc[:2].assign(time=[12.0, 24.0 + NOISE]),
                udict={"time": "hr", "Aurine": "mg"},
                ureg=self.ureg,
            ),
            # the first collection, which ends at the reset
            "first": DataSet.from_df(
                urine.iloc[:2], udict={"time": "hr", "Aurine": "mg"}, ureg=self.ureg
            ),
            # the second collection with two replicates at its start
            "replicates": DataSet.from_df(
                pd.DataFrame(
                    {
                        "time": [24.0, *TIMES[1:]],
                        "Aurine": [0.0, 0.0, TRUTH["urine"][2], TRUTH["urine"][3]],
                    }
                ),
                udict={"time": "hr", "Aurine": "mg"},
                ureg=self.ureg,
            ),
            # the second collection, which starts at the reset
            "second": DataSet.from_df(
                pd.DataFrame(
                    {
                        "time": TIMES[1:],
                        "Aurine": [0.0, TRUTH["urine"][2], TRUTH["urine"][3]],
                    }
                ),
                udict={"time": "hr", "Aurine": "mg"},
                ureg=self.ureg,
            ),
            # the second collection, its start just before the reset, closer
            # than the resolution of the time
            "second_early": DataSet.from_df(
                pd.DataFrame(
                    {
                        "time": [24.0 - NOISE, *TIMES[2:]],
                        "Aurine": [0.0, TRUTH["urine"][2], TRUTH["urine"][3]],
                    }
                ),
                udict={"time": "hr", "Aurine": "mg"},
                ureg=self.ureg,
            ),
            "ratio": DataSet.from_df(
                pd.DataFrame({"time": TIMES, "r": [1.0] * len(TIMES)}),
                udict={"time": "hr", "r": "dimensionless"},
                ureg=self.ureg,
            ),
            # the dose at the start
            "depot": DataSet.from_df(
                pd.DataFrame({"time": [0.0], "PODOSE": [100.0]}),
                udict={"time": "hr", "PODOSE": "mg"},
                ureg=self.ureg,
            ),
            "values": DataSet.from_df(
                values,
                udict={
                    "cmax": units["pk.cmax"],
                    "auc": units["pk.auc_last"],
                    "urine_max": units["urine_max"],
                },
                ureg=self.ureg,
            ),
        }

    def models(self) -> dict[str, AbstractModel | Path]:
        # a file, which the PEtab export writes
        return {"m": AbstractModel(source=Path(self.base_path or ".") / MODEL_FILE)}

    def simulations(self) -> dict[str, Simulation]:
        return {
            "collected": Simulation(end=48, changes=[DOSE, RESET]),
            # the urine is emptied at the end of the simulation
            "end": Simulation(end=24, changes=[DOSE, RESET]),
            # a grid of output times without the time of the reset
            "coarse": Simulation(end=48, steps=7, changes=[DOSE, RESET]),
            # the protocol from a negative start, shifted to the times of the
            # data
            "shifted": Simulation(
                start=-24,
                end=24,
                changes=[
                    Change(-24, {"PODOSE": Q(100, "mg")}),
                    Change(0, {"Aurine": Q(0, "mg")}),
                ],
                time_shift=24,
            ),
        }

    def observables(self) -> dict[str, Observable]:
        return dict(OBSERVABLES)

    def tasks(self) -> dict[str, Task]:
        return {
            "task_collected": Task(model="m", simulation="collected"),
            "task_end": Task(model="m", simulation="end"),
            "task_coarse": Task(model="m", simulation="coarse"),
            "task_shifted": Task(model="m", simulation="shifted"),
        }

    def fit_mappings(self) -> dict[str, FitMapping]:
        def scalar(
            yid: str, observable: str, task: str = "task_collected"
        ) -> FitMapping:
            return FitMapping(
                self,
                reference=FitData(self, dataset="values", xid=None, yid=yid),
                observable=FitData(self, task=task, xid=None, yid=observable),
            )

        return {
            "fm_urine": FitMapping(
                self,
                reference=FitData(self, dataset="urine", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="Aurine"
                ),
            ),
            "fm_late": FitMapping(
                self,
                reference=FitData(self, dataset="late", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="Aurine"
                ),
            ),
            "fm_twice": FitMapping(
                self,
                reference=FitData(self, dataset="twice", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="urine_twice"
                ),
            ),
            "fm_end": FitMapping(
                self,
                reference=FitData(self, dataset="first", xid="time", yid="Aurine"),
                observable=FitData(self, task="task_end", xid="time", yid="Aurine"),
            ),
            # a curve of one time, a total amount at the end of the collection
            "fm_end_single": FitMapping(
                self,
                reference=FitData(
                    self, dataset="urine", xid="time", yid="Aurine", sel={"time": 24.0}
                ),
                observable=FitData(self, task="task_end", xid="time", yid="Aurine"),
            ),
            "fm_first": FitMapping(
                self,
                reference=FitData(self, dataset="first", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="Aurine"
                ),
            ),
            "fm_second": FitMapping(
                self,
                reference=FitData(self, dataset="second", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="Aurine"
                ),
            ),
            "fm_second_early": FitMapping(
                self,
                reference=FitData(
                    self, dataset="second_early", xid="time", yid="Aurine"
                ),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="Aurine"
                ),
            ),
            "fm_replicates": FitMapping(
                self,
                reference=FitData(self, dataset="replicates", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="Aurine"
                ),
            ),
            "fm_model": FitMapping(
                self,
                reference=FitData(self, dataset="twice", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="urine_model"
                ),
                observable_model=ObservableModel("2 * Aurine", unit="mg"),
            ),
            "fm_noise": FitMapping(
                self,
                reference=FitData(self, dataset="urine", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="Aurine"
                ),
                noise=NoiseModel(formula="0.1 * Aurine + 0.01 * PODOSE"),
            ),
            "fm_depot_noise": FitMapping(
                self,
                reference=FitData(self, dataset="depot", xid="time", yid="PODOSE"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="PODOSE"
                ),
                noise=NoiseModel(formula="0.1 * PODOSE"),
            ),
            "fm_ratio": FitMapping(
                self,
                reference=FitData(self, dataset="ratio", xid="time", yid="r"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="urine_ratio"
                ),
            ),
            "fm_share": FitMapping(
                self,
                reference=FitData(self, dataset="ratio", xid="time", yid="r"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="urine_share"
                ),
            ),
            "fm_coarse": FitMapping(
                self,
                reference=FitData(self, dataset="urine", xid="time", yid="Aurine"),
                observable=FitData(self, task="task_coarse", xid="time", yid="Aurine"),
            ),
            "fm_coarse_ratio": FitMapping(
                self,
                reference=FitData(self, dataset="ratio", xid="time", yid="r"),
                observable=FitData(
                    self, task="task_coarse", xid="time", yid="urine_ratio"
                ),
            ),
            "fm_coarse_share": FitMapping(
                self,
                reference=FitData(self, dataset="ratio", xid="time", yid="r"),
                observable=FitData(
                    self, task="task_coarse", xid="time", yid="urine_share"
                ),
            ),
            "fm_coarse_late": FitMapping(
                self,
                reference=FitData(self, dataset="late", xid="time", yid="Aurine"),
                observable=FitData(self, task="task_coarse", xid="time", yid="Aurine"),
            ),
            "fm_noisy": FitMapping(
                self,
                reference=FitData(self, dataset="noisy", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="Aurine"
                ),
            ),
            "fm_depot": FitMapping(
                self,
                reference=FitData(self, dataset="depot", xid="time", yid="PODOSE"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="PODOSE"
                ),
            ),
            # a timecourse over another selection, which keeps the output
            "fm_over": FitMapping(
                self,
                reference=FitData(self, dataset="over", xid="Aelim", yid="C"),
                observable=FitData(self, task="task_collected", xid="Aelim", yid="[C]"),
            ),
            "fm_cmax": scalar("cmax", "pk.cmax"),
            "fm_auc": scalar("auc", "pk.auc_last"),
            "fm_urine_max": scalar("urine_max", "urine_max"),
            # the cmax of the simulation which ends with the reset
            "fm_end_cmax": scalar("cmax", "pk.cmax", task="task_end"),
            "fm_coarse_max": scalar("urine_max", "urine_max", task="task_coarse"),
            "fm_shifted": FitMapping(
                self,
                reference=FitData(self, dataset="urine", xid="time", yid="Aurine"),
                observable=FitData(self, task="task_shifted", xid="time", yid="Aurine"),
            ),
            "fm_shifted_max": scalar("urine_max", "urine_max", task="task_shifted"),
        }


def _problem(
    mappings: list[str],
    tmp_path: Path,
    data_at_change: DataAtChangeType = DataAtChangeType.BEFORE,
) -> OptimizationProblem:
    (tmp_path / MODEL_FILE).write_text(sbml_pk(urine=True))
    problem = OptimizationProblem(
        "urine",
        [FitMappingCollection(experiment=UrineStudy, mappings=mappings)],
        [
            FitParameter(
                pid="ke",
                lower_bound=0.01,
                upper_bound=1.0,
                start_value=0.5,
                unit="1/hr",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(
        FitSettings(
            parameter_scale=ParameterScaleType.LINEAR, data_at_change=data_at_change
        )
    )
    return problem


def _predictions(problem: OptimizationProblem) -> dict[str, np.ndarray]:
    """Get the prediction of every mapping at the true parameters by its key."""
    x = np.array([TRUE_KE])
    return {
        problem.mapping_keys[k]: prediction
        for k, prediction in problem.predictions(x, problem.indices()).items()
    }


def test_the_urine_at_the_reset_is_the_collected_amount(tmp_path: Path) -> None:
    """A timecourse which outputs the times of the data sees the state before."""
    problem = _problem(["fm_urine"], tmp_path)
    assert problem._group_graphs == [None]
    prediction = _predictions(problem)["fm_urine"]
    np.testing.assert_allclose(prediction, TRUTH["urine"], rtol=1e-5)
    assert problem.cost_least_square(np.array([TRUE_KE])) == pytest.approx(
        0.0, abs=1e-6
    )


def test_after_compares_with_the_emptied_urine(tmp_path: Path) -> None:
    """`after` is the semantics of PEtab: the state after the change."""
    problem = _problem(["fm_urine"], tmp_path, DataAtChangeType.AFTER)
    prediction = _predictions(problem)["fm_urine"]
    assert prediction[1] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(
        prediction[[0, 2, 3]], TRUTH["urine"][[0, 2, 3]], rtol=1e-5
    )


def test_the_results_of_a_run_have_the_states_before_and_after_the_change() -> None:
    """The output of a run at the time of the reset: the urine before it, then `0`."""
    run = TRUTH["after"]
    time = run["time"].values
    assert np.count_nonzero(time == 24.0) == 2
    np.testing.assert_allclose(
        run["Aurine"].values[time == 24.0], [TRUTH["urine"][1], 0.0], rtol=STEPS_RTOL
    )


DENSE = ["fm_urine", "fm_twice", "fm_cmax", "fm_auc", "fm_urine_max"]


@pytest.mark.parametrize("dense", [False, True])
def test_the_simulation_continues_from_the_state_at_the_reset(
    tmp_path: Path, dense: bool
) -> None:
    """The urine after the reset is the amount collected from 24 h on.

    The steps of the integrator of a group with observables end the segment
    before the reset in its state at 24 h, so the second collection starts
    there.
    """
    mappings = ["fm_urine", "fm_cmax"] if dense else ["fm_urine"]
    prediction = _predictions(_problem(mappings, tmp_path))["fm_urine"]
    np.testing.assert_allclose(
        prediction, TRUTH["urine"], rtol=INTERPOLATED_RTOL if dense else 1e-5
    )


@pytest.mark.parametrize("data_at_change", list(DataAtChangeType))
def test_data_just_before_a_change_is_interpolated_up_to_the_state_before(
    tmp_path: Path, data_at_change: DataAtChangeType
) -> None:
    """Data between the last step before the reset and the reset sees the urine before it.

    The steps of the integrator of a group with observables reach the reset at
    24 h from a step before 23.95 h; the data there is interpolated between
    that step and the state before the reset, not the emptied urine after it.
    """
    time = TRUTH["after"]["time"].values
    assert time[time < 24.0].max() < LATE[1]
    problem = _problem(["fm_late", "fm_cmax"], tmp_path, data_at_change)
    prediction = _predictions(problem)["fm_late"]
    np.testing.assert_allclose(prediction, TRUTH["late"], rtol=INTERPOLATED_RTOL)


@pytest.mark.parametrize("dense", [False, True])
def test_a_first_point_just_before_a_change_takes_the_state_after(
    tmp_path: Path, dense: bool
) -> None:
    """The start of a curve closer to the reset than the resolution is at the reset.

    It is the time of the reset, whose state after it the first point of a
    curve takes, also where the output of a group with observables jumps.
    """
    mappings = ["fm_second_early", "fm_cmax"] if dense else ["fm_second_early"]
    prediction = _predictions(_problem(mappings, tmp_path))["fm_second_early"]
    assert prediction[0] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(
        prediction[1:],
        TRUTH["urine"][2:],
        rtol=INTERPOLATED_RTOL if dense else 1e-5,
    )


@pytest.mark.parametrize("dense", [False, True])
def test_after_takes_a_time_just_before_a_change_as_it_is(
    tmp_path: Path, dense: bool
) -> None:
    """`after` compares a time closer to the reset than the resolution as PEtab does.

    The time is before the reset, its state is the urine collected until the
    reset, not the emptied urine after it.
    """
    mappings = ["fm_second_early", "fm_cmax"] if dense else ["fm_second_early"]
    problem = _problem(mappings, tmp_path, DataAtChangeType.AFTER)
    prediction = _predictions(problem)["fm_second_early"]
    assert prediction[0] == pytest.approx(
        TRUTH["urine"][1], rel=INTERPOLATED_RTOL if dense else 1e-5
    )


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("data_at_change", list(DataAtChangeType))
def test_the_curve_of_a_mapping_jumps_at_a_change(
    tmp_path: Path, dense: bool, data_at_change: DataAtChangeType
) -> None:
    """The curve of the figure of a mapping has the states before and after the reset.

    The prediction at the reset lies on the curve: the state before it with
    `before`, the state after it with `after`.
    """
    mappings = ["fm_urine", "fm_cmax"] if dense else ["fm_urine"]
    problem = _problem(mappings, tmp_path, data_at_change)
    k = problem.mapping_keys.index("fm_urine")
    data = problem.residuals(np.array([TRUE_KE]), complete_data=True)
    assert isinstance(data, dict)
    x, y = np.asarray(data["x_obs"][k]), np.asarray(data["y_obs"][k])
    prediction = np.asarray(data["y_obsip"][k])
    assert np.all(np.diff(x) >= 0)
    at = np.flatnonzero(x == 24.0)
    assert at.size == 2
    assert y[at[0]] == pytest.approx(TRUTH["urine"][1], rel=1e-5)
    assert y[at[1]] == pytest.approx(0.0, abs=1e-8)
    expected = 0 if data_at_change is DataAtChangeType.BEFORE else 1
    # the curve is the output of the simulation, the prediction the output at
    # the times of the data: both within the tolerances of the integrator
    assert prediction[1] == pytest.approx(y[at[expected]], rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("dense", [False, True])
def test_the_log_panel_of_a_curve_has_a_gap_at_zero(
    tmp_path: Path, dense: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The emptied urine after the reset is no vertical line to the bottom.

    A logarithmic axis clips the `0` of the curve after the reset (and at the
    start of a dense output), the curve has a gap there; the linear panel
    keeps the jump to `0`.
    """
    figures: dict[str, Figure] = {}
    monkeypatch.setattr(
        FitReport,
        "_save_mpl_figure",
        lambda self, fig, path: figures.__setitem__(path.stem, fig),
    )
    mappings = ["fm_urine", "fm_cmax"] if dense else ["fm_urine"]
    problem = _problem(mappings, tmp_path)
    report = FitReport(
        problem,
        problem.settings_initialized,
        ParameterSet(sid="true", values={"ke": TRUE_KE}),
    )
    report.plot_mapping(tmp_path)
    try:
        for name in ("UrineStudy_fm_urine",):
            linear, log = figures[name].axes[:2]
            assert log.get_yscale() == "log"
            [curve] = [
                line for line in linear.get_lines() if line.get_label() == "true"
            ]
            [masked] = [line for line in log.get_lines() if line.get_label() == "true"]
            y = np.asarray(curve.get_ydata(), dtype=float)
            y_log = np.asarray(masked.get_ydata(), dtype=float)
            zero = y <= 0.0
            assert zero.any()
            assert np.isnan(y_log[zero]).all()
            np.testing.assert_array_equal(y_log[~zero], y[~zero])
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_a_group_with_observables_sees_the_state_before(tmp_path: Path) -> None:
    """The dense output of a group with a PK observable, the timecourse before.

    The data between the steps of the integrator is interpolated as before,
    the data at the reset is the state before it, also for an observable
    which is a timecourse.
    """
    problem = _problem(DENSE, tmp_path)
    assert len(problem.mapping_groups) == 1
    assert problem._group_graphs[0] is not None
    before = _predictions(problem)
    after = _predictions(_problem(DENSE, tmp_path, DataAtChangeType.AFTER))
    for key, factor in (("fm_urine", 1.0), ("fm_twice", 2.0)):
        assert before[key][1] == pytest.approx(
            factor * TRUTH["urine"][1], rel=STEPS_RTOL
        )
        assert after[key][1] == pytest.approx(0.0, abs=1e-8)
        np.testing.assert_array_equal(before[key][[0, 2, 3]], after[key][[0, 2, 3]])


def test_a_grid_without_the_time_of_the_change(tmp_path: Path) -> None:
    """The output of the simulation does not have the time of the reset.

    The fit adds it for the timecourse, the value per simulation is the one
    of a run of the grid, which misses the urine before the reset.
    """
    problem = _problem(["fm_coarse", "fm_coarse_max"], tmp_path)
    assert 24.0 not in COARSE
    assert 24.0 in problem.plans[0].times
    before = _predictions(problem)
    after = _predictions(
        _problem(["fm_coarse", "fm_coarse_max"], tmp_path, DataAtChangeType.AFTER)
    )
    assert before["fm_coarse"][1] == pytest.approx(TRUTH["urine"][1], rel=1e-5)
    np.testing.assert_array_equal(
        before["fm_coarse"][[0, 2, 3]], after["fm_coarse"][[0, 2, 3]]
    )
    assert before["fm_coarse_max"] == pytest.approx(after["fm_coarse_max"], rel=1e-12)
    assert before["fm_coarse_max"][0] == pytest.approx(TRUTH["coarse_max"], rel=1e-12)
    assert TRUTH["coarse_max"] < TRUTH["urine"][1]
    # the state after the reset at its time, which the output does not have
    assert after["fm_coarse"][1] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("data_at_change", list(DataAtChangeType))
def test_a_grid_without_the_time_of_the_change_jumps(
    tmp_path: Path, data_at_change: DataAtChangeType
) -> None:
    """The curve of a grid without the time of the reset jumps at the reset.

    The data at 23.95 h between the output at 20.6 h and the reset is
    interpolated toward the urine before the reset.
    """
    mappings = ["fm_coarse", "fm_coarse_late", "fm_coarse_max"]
    problem = _problem(mappings, tmp_path, data_at_change)
    assert 24.0 not in COARSE
    k = problem.mapping_keys.index("fm_coarse")
    data = problem.residuals(np.array([TRUE_KE]), complete_data=True)
    assert isinstance(data, dict)
    x, y = np.asarray(data["x_obs"][k]), np.asarray(data["y_obs"][k])
    assert np.all(np.diff(x) >= 0)
    at = np.flatnonzero(x == 24.0)
    assert at.size == 2
    assert y[at[0]] == pytest.approx(TRUTH["urine"][1], rel=1e-5)
    assert y[at[1]] == pytest.approx(0.0, abs=1e-8)
    # the output before the reset and the state before it
    previous = int(np.flatnonzero(x < 24.0)[-1])
    expected = np.interp(LATE[1], [x[previous], 24.0], [y[previous], y[at[0]]])
    late = _predictions(problem)["fm_coarse_late"]
    assert late[1] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("dense", [False, True])
def test_a_time_within_the_resolution_is_at_the_change(
    tmp_path: Path, dense: bool
) -> None:
    """A time just after the reset, closer than the resolution, sees the state before."""
    mappings = ["fm_noisy", "fm_cmax"] if dense else ["fm_noisy"]
    before = _predictions(_problem(mappings, tmp_path))["fm_noisy"]
    assert before[1] == pytest.approx(TRUTH["urine"][1], rel=STEPS_RTOL)
    after = _predictions(_problem(mappings, tmp_path, DataAtChangeType.AFTER))
    assert after["fm_noisy"][1] == pytest.approx(0.0, abs=1e-6)


def test_a_timecourse_over_another_selection(tmp_path: Path) -> None:
    """A group with a timecourse over another selection keeps its output.

    The timecourse over the time of the same group sees the state before the
    reset, the timecourse over the amount eliminated is interpolated on the
    output as a run has it.
    """
    problem = _problem(["fm_urine", "fm_over"], tmp_path)
    assert len(problem.mapping_groups) == 1
    assert problem._group_graphs[0] is not None
    predictions = _predictions(problem)
    assert predictions["fm_urine"][1] == pytest.approx(
        TRUTH["urine"][1], rel=STEPS_RTOL
    )
    run = TRUTH["after"]
    np.testing.assert_allclose(
        predictions["fm_over"],
        np.interp(ELIMINATED, run["Aelim"].values, run["[C]"].values),
        rtol=1e-9,
    )
    alone = _predictions(_problem(["fm_over"], tmp_path))["fm_over"]
    np.testing.assert_array_equal(alone, predictions["fm_over"])


@pytest.mark.parametrize("dense", [False, True])
def test_two_mappings_of_one_time_near_a_change(tmp_path: Path, dense: bool) -> None:
    """The reset and a time closer to it than the resolution, in two mappings."""
    mappings = ["fm_urine", "fm_noisy"] + (["fm_cmax"] if dense else [])
    problem = _problem(mappings, tmp_path)
    if not dense:
        # both times are one output time, next to the start
        assert len(problem.plans[0].times) == len(TIMES) + 1
    predictions = _predictions(problem)
    assert predictions["fm_noisy"][1] == predictions["fm_urine"][1]
    assert predictions["fm_noisy"][1] == pytest.approx(
        TRUTH["urine"][1], rel=STEPS_RTOL
    )


@pytest.mark.parametrize("dense", [False, True])
def test_a_negative_start_and_a_time_shift(tmp_path: Path, dense: bool) -> None:
    """The change at 0 of a simulation from -24 h is at 24 h in the time of the data."""
    mappings = ["fm_shifted", "fm_shifted_max"] if dense else ["fm_shifted"]
    problem = _problem(mappings, tmp_path)
    assert problem.times_before_changes(0).tolist() == [24.0]
    before = _predictions(problem)["fm_shifted"]
    after = _predictions(_problem(mappings, tmp_path, DataAtChangeType.AFTER))
    assert before[1] == pytest.approx(TRUTH["urine"][1], rel=STEPS_RTOL)
    assert after["fm_shifted"][1] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_array_equal(before[[0, 2, 3]], after["fm_shifted"][[0, 2, 3]])
    if not dense:
        np.testing.assert_allclose(
            before[[0, 2, 3]], TRUTH["urine"][[0, 2, 3]], rtol=1e-5
        )


@pytest.mark.parametrize("dense", [False, True])
def test_the_data_at_the_start_sees_the_changes_at_the_start(
    tmp_path: Path, dense: bool
) -> None:
    """A change at the start is the initial condition, the dose is in the depot."""
    mappings = ["fm_depot", "fm_cmax"] if dense else ["fm_depot"]
    prediction = _predictions(_problem(mappings, tmp_path))["fm_depot"]
    assert prediction[0] == pytest.approx(100.0)


def test_the_observables_are_computed_on_the_timecourse_of_a_run(
    tmp_path: Path,
) -> None:
    """The PK parameters and the reductions do not see the state before."""
    mappings = ["fm_urine", "fm_twice", "fm_cmax", "fm_auc", "fm_urine_max"]
    before = _predictions(_problem(mappings, tmp_path))
    after = _predictions(_problem(mappings, tmp_path, DataAtChangeType.AFTER))
    for key, name in (
        ("fm_cmax", "cmax"),
        ("fm_auc", "auc"),
        ("fm_urine_max", "urine_max"),
    ):
        assert before[key] == pytest.approx(after[key], rel=1e-12)
        assert before[key][0] == pytest.approx(TRUTH[name], rel=1e-6)
    # the maximum of the output of a run is the urine before the reset
    assert TRUTH["urine_max"] == pytest.approx(TRUTH["urine"][1], rel=STEPS_RTOL)
    assert after["fm_urine"][1] == pytest.approx(0.0, abs=1e-8)
    assert after["fm_twice"][1] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("dense", [False, True])
def test_a_change_at_the_end_is_compared_before_it(tmp_path: Path, dense: bool) -> None:
    """A change at the end of the simulation: the data at the end sees the state before."""
    mappings = ["fm_end", "fm_end_cmax"] if dense else ["fm_end"]
    before = _predictions(_problem(mappings, tmp_path))["fm_end"]
    after = _predictions(_problem(mappings, tmp_path, DataAtChangeType.AFTER))["fm_end"]
    assert before[1] == pytest.approx(TRUTH["urine"][1], rel=STEPS_RTOL)
    assert after[1] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("dense", [False, True])
def test_a_curve_of_one_time_takes_the_state_before(
    tmp_path: Path, dense: bool
) -> None:
    """A total amount at the reset is no start of a collection: the amount before."""
    mappings = ["fm_end_single", "fm_end_cmax"] if dense else ["fm_end_single"]
    problem = _problem(mappings, tmp_path)
    k = problem.mapping_keys.index("fm_end_single")
    assert problem.times_before_changes(k).tolist() == [24.0]
    assert _predictions(problem)["fm_end_single"][0] == pytest.approx(
        TRUTH["urine"][1], rel=STEPS_RTOL
    )
    after = _problem(mappings, tmp_path, DataAtChangeType.AFTER)
    assert _predictions(after)["fm_end_single"][0] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("dense", [False, True])
def test_replicates_at_the_start_of_a_curve_take_the_state_after(
    tmp_path: Path, dense: bool
) -> None:
    """Two points at the first time of a longer curve both start the collection."""
    mappings = ["fm_replicates", "fm_first"] + (["fm_cmax"] if dense else [])
    problem = _problem(mappings, tmp_path)
    k = problem.mapping_keys.index("fm_replicates")
    assert problem.times_before_changes(k).size == 0
    predictions = _predictions(problem)
    np.testing.assert_allclose(predictions["fm_replicates"][:2], 0.0, atol=1e-8)
    assert predictions["fm_first"][1] == pytest.approx(
        TRUTH["urine"][1], rel=STEPS_RTOL
    )


@pytest.mark.parametrize("dense", [False, True])
def test_a_curve_which_starts_and_one_which_ends_at_a_reset(
    tmp_path: Path, dense: bool
) -> None:
    """The first collection ends at the reset, the second starts at it.

    In one simulation the data at the reset is the collected amount for the
    curve which ends there and `0` for the first point of the curve which
    starts there (#262, "First point after").
    """
    mappings = ["fm_first", "fm_second"] + (["fm_cmax"] if dense else [])
    problem = _problem(mappings, tmp_path)
    assert len(problem.mapping_groups) == 1
    predictions = _predictions(problem)
    assert predictions["fm_first"][1] == pytest.approx(
        TRUTH["urine"][1], rel=STEPS_RTOL
    )
    assert predictions["fm_second"][0] == pytest.approx(0.0, abs=1e-8)
    after = _predictions(_problem(mappings, tmp_path, DataAtChangeType.AFTER))
    np.testing.assert_array_equal(predictions["fm_second"], after["fm_second"])
    assert after["fm_first"][1] == pytest.approx(0.0, abs=1e-8)
    if not dense:
        np.testing.assert_allclose(
            predictions["fm_second"][1:], TRUTH["urine"][2:], rtol=1e-5
        )
        assert problem.cost_least_square(np.array([TRUE_KE])) == pytest.approx(
            0.0, abs=1e-6
        )


@pytest.mark.parametrize("dense", [False, True])
def test_a_group_with_data_at_a_change_and_without(tmp_path: Path, dense: bool) -> None:
    """A mapping without data at the change shares the group with one which has it."""
    mappings = ["fm_urine", "fm_depot"] + (["fm_cmax"] if dense else [])
    problem = _problem(mappings, tmp_path)
    predictions = _predictions(problem)
    assert predictions["fm_depot"] == pytest.approx([100.0])
    assert predictions["fm_urine"][1] == pytest.approx(
        TRUTH["urine"][1], rel=STEPS_RTOL
    )
    assert np.isfinite(problem.cost_least_square(np.array([TRUE_KE])))


@pytest.mark.parametrize("dense", [False, True])
def test_an_observable_model_and_a_noise_model_at_a_change(
    tmp_path: Path, dense: bool
) -> None:
    """The symbols of an observable model and of a noise model take the state before."""
    mappings = ["fm_model", "fm_noise", "fm_depot_noise"]
    problem = _problem(mappings + (["fm_cmax"] if dense else []), tmp_path)
    assert len(problem.mapping_groups) == 1
    x = np.array([TRUE_KE])
    evaluations = problem.evaluations(x, problem.indices())
    by_key = {problem.mapping_keys[k]: e for k, e in evaluations.items()}
    assert by_key["fm_model"].prediction[1] == pytest.approx(
        2 * TRUTH["urine"][1], rel=STEPS_RTOL
    )
    noise = by_key["fm_noise"].selections
    assert noise["Aurine"][1] == pytest.approx(TRUTH["urine"][1], rel=STEPS_RTOL)
    assert set(noise) == {"Aurine", "PODOSE"}
    assert by_key["fm_depot_noise"].selections["PODOSE"] == pytest.approx([100.0])


def test_a_timecourse_reads_the_values_per_simulation_of_the_run(
    tmp_path: Path,
) -> None:
    """A value per simulation which a timecourse reads is the one of the run.

    The output of the steps of the integrator has the state before the
    reset, the maximum of the urine: `Aurine / urine_max` at the reset is
    one. The grid without the reset misses it, its maximum is below the
    collected amount, so the ratio is above one on the output with the time
    of the reset, which the fit adds; a reduction inside the formula of the
    timecourse, `Aurine / max(Aurine)`, is evaluated on that output, whose
    maximum is the state before the reset: the ratio is one.
    """
    problem = _problem(["fm_ratio", "fm_share", "fm_cmax"], tmp_path)
    predictions = _predictions(problem)
    assert predictions["fm_ratio"][1] == pytest.approx(1.0, rel=STEPS_RTOL)
    assert predictions["fm_share"][1] == pytest.approx(1.0, rel=1e-12)
    coarse = _predictions(_problem(["fm_coarse_ratio", "fm_coarse_share"], tmp_path))
    ratio = coarse["fm_coarse_ratio"][1]
    assert ratio == pytest.approx(
        TRUTH["urine"][1] / TRUTH["coarse_max"], rel=STEPS_RTOL
    )
    assert ratio > 1.0
    assert coarse["fm_coarse_share"][1] == pytest.approx(1.0, rel=1e-12)


def test_the_settings_compare_the_data_before_a_change_by_default() -> None:
    """`before` is the default, stored settings without the key are `after`."""
    settings = FitSettings()
    assert settings.data_at_change is DataAtChangeType.BEFORE
    assert settings.to_dict()["data_at_change"] == "before"
    after = FitSettings(data_at_change=DataAtChangeType.AFTER)
    assert FitSettings.from_dict(after.to_dict()) == after
    # the settings stored by 0.8.5 compared the data with the state after
    stored = settings.to_dict()
    del stored["data_at_change"]
    assert FitSettings.from_dict(stored).data_at_change is DataAtChangeType.AFTER
    # a string is converted, the settings stay hashable and compare by value
    converted = FitSettings(data_at_change="after")  # ty: ignore[invalid-argument-type]
    assert converted == after
    assert hash(converted) == hash(after)
    with pytest.raises(ValueError, match="data_at_change"):
        FitSettings(data_at_change="later")  # ty: ignore[invalid-argument-type]


def test_the_times_of_the_data_at_a_change(tmp_path: Path) -> None:
    """The data at the reset and the noisy time are at the change, not the start."""
    problem = _problem(
        ["fm_urine", "fm_noisy", "fm_depot", "fm_second", "fm_cmax"], tmp_path
    )
    times = {
        problem.mapping_keys[k]: problem.times_before_changes(k).tolist()
        for k in problem.indices()
    }
    assert times == {
        "fm_urine": [24.0],
        "fm_noisy": [24.0 + NOISE],
        "fm_depot": [],
        # the first point of a curve takes the state after the change
        "fm_second": [],
        "fm_cmax": [],
    }
    after = _problem(["fm_urine"], tmp_path, DataAtChangeType.AFTER)
    assert after.times_before_changes(0).size == 0


def test_the_export_reports_the_measurements_at_a_change(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """PEtab compares a measurement at a change with the state after it."""
    problem = _problem(["fm_urine"], tmp_path)
    with caplog.at_level(logging.WARNING):
        ids = {gap.id for gap in gaps_of_problem(problem)}
    assert "measurement-at-change" in ids
    assert measurements_at_changes(problem) == {"UrineStudy.fm_urine": [24.0]}
    assert "UrineStudy.fm_urine at [24.0]" in caplog.text

    after = _problem(["fm_urine"], tmp_path, DataAtChangeType.AFTER)
    assert "measurement-at-change" not in {gap.id for gap in gaps_of_problem(after)}
    assert measurements_at_changes(after) == {}
    for start in (["fm_depot"], ["fm_second"]):
        problem = _problem(start, tmp_path)
        assert "measurement-at-change" not in {
            gap.id for gap in gaps_of_problem(problem)
        }


@pytest.mark.parametrize(
    "data_at_change", [DataAtChangeType.BEFORE, DataAtChangeType.AFTER]
)
def test_the_round_trip_keeps_the_data_at_a_change(
    tmp_path: Path, data_at_change: DataAtChangeType
) -> None:
    """The extension carries the setting, the read problem fits the same data."""
    problem = _problem(["fm_urine"], tmp_path, data_at_change)
    to_petab(problem, tmp_path / "petab", settings=problem.settings_initialized)
    read, settings = from_petab(tmp_path / "petab" / "problem.yaml", opid="read")
    assert settings == problem.settings_initialized
    read.initialize(settings)
    x = np.array([TRUE_KE])
    np.testing.assert_allclose(
        read.predictions(x)[0], problem.predictions(x)[0], rtol=1e-6
    )


def test_an_extension_before_0_4_0_compares_the_data_after_a_change(
    tmp_path: Path,
) -> None:
    """Version 0.4.0 carries `data_at_change`, the settings before it are `after`."""
    problem = _problem(["fm_urine"], tmp_path)
    yaml_file = to_petab(
        problem, tmp_path / "petab", settings=problem.settings_initialized
    )
    config = yaml.safe_load(yaml_file.read_text())
    block = config["extensions"]["sbmlsim"]
    assert block["version"] == EXTENSION_VERSION == "0.4.0"
    assert block["settings"]["data_at_change"] == DataAtChangeType.BEFORE.value
    assert PetabReader.from_yaml(yaml_file).settings.data_at_change is (
        DataAtChangeType.BEFORE
    )

    block["version"] = "0.3.0"
    del block["settings"]["data_at_change"]
    yaml_file.write_text(yaml.safe_dump(config, sort_keys=False))
    assert PetabReader.from_yaml(yaml_file).settings.data_at_change is (
        DataAtChangeType.AFTER
    )


def test_a_problem_without_the_extension_has_the_semantics_of_petab() -> None:
    """A PEtab problem of another tool compares the data after the change."""
    reader = PetabReader.__new__(PetabReader)
    reader.extension = None
    assert reader.settings.data_at_change is DataAtChangeType.AFTER


#: the file of the sawtooth model in the base path of a problem
SAWTOOTH_FILE = "sawtooth.xml"


class SawtoothStudy(SimulationExperiment):
    """A sawtooth `X` which an event of the model resets at 1, 2, ..."""

    def datasets(self) -> dict[str, DataSet]:
        return {
            # the data at the reset at 1 is after it
            "x": DataSet.from_df(
                pd.DataFrame({"time": [0.5, 1.0, 1.5], "X": [0.5, 0.0, 0.5]}),
                udict={"time": "s", "X": "dimensionless"},
                ureg=self.ureg,
            ),
            "max": DataSet.from_df(
                pd.DataFrame({"xmax": [1.0]}),
                udict={"xmax": "dimensionless"},
                ureg=self.ureg,
            ),
        }

    def models(self) -> dict[str, AbstractModel | Path]:
        return {"m": AbstractModel(source=Path(self.base_path or ".") / SAWTOOTH_FILE)}

    def simulations(self) -> dict[str, Simulation]:
        return {
            "steps": Simulation(end=2.5),
            # the steps of roadrunner end after the reset at 2 before the
            # change at 1.998: the segment with the reset at 1 inside is
            # integrated again with exact output
            "past": Simulation(end=2.5, changes=[Change(1.998, {"k": 1.0})]),
        }

    def observables(self) -> dict[str, Observable]:
        return {"xmax": Formula("xmax", "max(X)")}

    def tasks(self) -> dict[str, Task]:
        return {
            "task_steps": Task(model="m", simulation="steps"),
            "task_past": Task(model="m", simulation="past"),
        }

    def fit_mappings(self) -> dict[str, FitMapping]:
        mappings: dict[str, FitMapping] = {}
        for task in ("task_steps", "task_past"):
            mappings[f"fm_x_{task}"] = FitMapping(
                self,
                reference=FitData(self, dataset="x", xid="time", yid="X"),
                observable=FitData(self, task=task, xid="time", yid="X"),
            )
            # a value per simulation, so the group keeps its output
            mappings[f"fm_max_{task}"] = FitMapping(
                self,
                reference=FitData(self, dataset="max", xid=None, yid="xmax"),
                observable=FitData(self, task=task, xid=None, yid="xmax"),
            )
        return mappings


#: the reset of the sawtooth at 1, closer to it than the resolution of the time
#: of a simulation to 2.5 but after it
LATE_RESET = "at (time >= 1.0000000001)"


@pytest.mark.parametrize("trigger", ["at (X >= 1)", LATE_RESET])
@pytest.mark.parametrize("task", ["task_steps", "task_past"])
def test_data_at_an_event_of_the_model_sees_the_state_after_it(
    tmp_path: Path, task: str, trigger: str
) -> None:
    """Data at the time of an event of the model takes the state after the event.

    The steps of the integrator have the states before and after the reset
    at 1 at its time, a segment integrated again with exact output a
    resolution before and after it; data at 1 is not their midpoint. CVODE
    locates the reset a little after its time, its states are a little after
    the data at 1, as they are for the late reset, which data at 1 sees as
    well.
    """
    prediction = sawtooth_prediction(tmp_path, task, trigger)
    np.testing.assert_allclose(prediction, [0.5, 0.0, 0.5], atol=1e-6)


def sawtooth_prediction(tmp_path: Path, task: str, trigger: str) -> np.ndarray:
    """Get the prediction of the fit of the sawtooth at the data `[0.5, 1, 1.5]`.

    Args:
        tmp_path: the directory of the model.
        task: the task of `SawtoothStudy`, `task_steps` or `task_past`.
        trigger: the trigger of the reset of the sawtooth.

    Returns:
        The prediction of `X` at the times of the data.
    """
    model = SAWTOOTH.replace("k = 1", "k = 1\n  p = 1").replace("at (X >= 1)", trigger)
    (tmp_path / SAWTOOTH_FILE).write_text(sbml(model))
    problem = OptimizationProblem(
        "sawtooth",
        [
            FitMappingCollection(
                experiment=SawtoothStudy, mappings=[f"fm_x_{task}", f"fm_max_{task}"]
            )
        ],
        [
            FitParameter(
                pid="p",
                lower_bound=0.1,
                upper_bound=10.0,
                start_value=1.0,
                unit="dimensionless",
            )
        ],
        base_path=tmp_path,
        data_path=tmp_path,
    )
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    assert problem._group_graphs[0] is not None
    k = problem.mapping_keys.index(f"fm_x_{task}")
    return problem.predictions(np.array([1.0]), problem.indices())[k]


def test_a_timecourse_observable_is_simulated_at_the_data(tmp_path: Path) -> None:
    """An observable without reductions is evaluated at the start and the data times."""
    problem = _problem(["fm_twice", "fm_urine"], tmp_path)
    assert problem._group_graphs == [None]
    assert problem._group_pointwise[0] is not None
    assert problem.plans[0].times == (0.0, *TIMES)
    predictions = _predictions(problem)
    np.testing.assert_allclose(predictions["fm_twice"], 2 * TRUTH["urine"], rtol=1e-5)
    np.testing.assert_array_equal(predictions["fm_twice"], 2 * predictions["fm_urine"])


@pytest.mark.parametrize("mapping", ["fm_ratio", "fm_share"])
def test_an_observable_of_the_whole_timecourse_keeps_the_output(
    tmp_path: Path, mapping: str
) -> None:
    """A value per simulation or a reduction needs the output of the simulation."""
    problem = _problem([mapping], tmp_path)
    assert problem._group_graphs[0] is not None
    assert problem._group_pointwise == [None]
    assert problem._curve_plans == [None]


def test_a_group_simulates_the_selections_of_its_mappings(tmp_path: Path) -> None:
    """The selections of a mapping are its own, not the ones of the other mappings."""
    problem = _problem(["fm_urine", "fm_model", "fm_over"], tmp_path)
    selections = dict(zip(problem.mapping_keys, problem.selections, strict=True))
    assert sorted(selections["fm_urine"]) == ["Aurine", "time"]
    assert sorted(selections["fm_model"]) == ["Aurine", "time"]
    assert sorted(selections["fm_over"]) == ["Aelim", "[C]"]


@pytest.mark.parametrize("mappings", [["fm_urine"], ["fm_twice"]])
def test_the_curve_of_the_evaluation_is_the_output_of_the_simulation(
    tmp_path: Path, mappings: list[str]
) -> None:
    """The fit outputs the times of the data, the evaluation the whole curve."""
    problem = _problem(mappings, tmp_path)
    data = problem.residuals(np.array([TRUE_KE]), complete_data=True)
    assert isinstance(data, dict)
    x = np.asarray(data["x_obs"][0])
    # the steps of the integrator of the simulation, with the reset twice
    assert x.size > 10 * (len(TIMES) + 1)
    assert (x[0], x[-1]) == (0.0, 48.0)
    assert np.flatnonzero(x == 24.0).size == 2
