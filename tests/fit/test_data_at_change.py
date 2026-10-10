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

import numpy as np
import pandas as pd
import pytest

from sbmlsim import Q
from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitData, FitMapping, FitMappingCollection, FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import DataAtChangeType, FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2 import to_petab
from sbmlsim.fit.petab_v2.gaps import gaps_of_problem, measurements_at_changes
from sbmlsim.fit.petab_v2.reader import PetabReader, from_petab
from sbmlsim.model import AbstractModel
from sbmlsim.simulation import PK, Change, Formula, Observable, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task
from tests.simulator.models import sbml_pk

#: the elimination rate of the model, the value of the fitted parameter
TRUE_KE = 0.2
DOSE = Change(0, {"PODOSE": Q(100, "mg")})
#: the urine is emptied at the end of the first collection interval
RESET = Change(24, {"Aurine": Q(0, "mg")})
TIMES = [12.0, 24.0, 36.0, 48.0]
#: the relative tolerance of a state of the steps of the integrator: roadrunner
#: interpolates the last step of a segment linearly to its end
STEPS_RTOL = 1e-4
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
    run = simulator.run(
        model,
        Simulation(end=48, changes=[DOSE, RESET]),
        list(OBSERVABLES.values()),
        keep=["pk.cmax", "pk.auc_last", "urine_max", "Aurine", "Aelim", "[C]"],
    )
    return {
        "urine": np.concatenate([first["Aurine"], second["Aurine"]]),
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
                pd.DataFrame({"time": [24.0 + NOISE], "Aurine": [TRUTH["urine"][1]]}),
                udict={"time": "hr", "Aurine": "mg"},
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
            "fm_twice": FitMapping(
                self,
                reference=FitData(self, dataset="twice", xid="time", yid="Aurine"),
                observable=FitData(
                    self, task="task_collected", xid="time", yid="urine_twice"
                ),
            ),
            "fm_end": FitMapping(
                self,
                reference=FitData(
                    self, dataset="urine", xid="time", yid="Aurine", sel={"time": 24.0}
                ),
                observable=FitData(self, task="task_end", xid="time", yid="Aurine"),
            ),
            "fm_coarse": FitMapping(
                self,
                reference=FitData(self, dataset="urine", xid="time", yid="Aurine"),
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


def test_the_results_of_a_run_keep_the_state_after_the_change() -> None:
    """A run is not a fit: its output at the time of the reset is `0`."""
    run = TRUTH["after"]
    time = run["time"].values
    assert np.count_nonzero(time == 24.0) == 1
    assert run["Aurine"].values[time == 24.0][0] == pytest.approx(0.0, abs=1e-8)


DENSE = ["fm_urine", "fm_twice", "fm_cmax", "fm_auc", "fm_urine_max"]


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
    """The output of the simulation does not have the time of the reset."""
    problem = _problem(["fm_coarse", "fm_coarse_max"], tmp_path)
    assert 24.0 not in problem.plans[0].times
    before = _predictions(problem)
    after = _predictions(
        _problem(["fm_coarse", "fm_coarse_max"], tmp_path, DataAtChangeType.AFTER)
    )
    assert before["fm_coarse"][1] == pytest.approx(TRUTH["urine"][1], rel=1e-5)
    np.testing.assert_array_equal(
        before["fm_coarse"][[0, 2, 3]], after["fm_coarse"][[0, 2, 3]]
    )
    assert before["fm_coarse_max"] == pytest.approx(after["fm_coarse_max"], rel=1e-12)


@pytest.mark.parametrize("dense", [False, True])
def test_a_time_within_the_resolution_is_at_the_change(
    tmp_path: Path, dense: bool
) -> None:
    """A time just after the reset, closer than the resolution, sees the state before."""
    mappings = ["fm_noisy", "fm_cmax"] if dense else ["fm_noisy"]
    before = _predictions(_problem(mappings, tmp_path))["fm_noisy"]
    assert before[0] == pytest.approx(TRUTH["urine"][1], rel=STEPS_RTOL)
    after = _predictions(_problem(mappings, tmp_path, DataAtChangeType.AFTER))
    assert after["fm_noisy"][0] == pytest.approx(0.0, abs=1e-6)


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
        # both times are one output time
        assert len(problem.plans[0].times) == len(TIMES)
    predictions = _predictions(problem)
    assert predictions["fm_noisy"][0] == predictions["fm_urine"][1]
    assert predictions["fm_noisy"][0] == pytest.approx(
        TRUTH["urine"][1], rel=STEPS_RTOL
    )


@pytest.mark.parametrize("dense", [False, True])
def test_a_negative_start_and_a_time_shift(tmp_path: Path, dense: bool) -> None:
    """The change at 0 of a simulation from -24 h is at 24 h in the time of the data."""
    mappings = ["fm_shifted", "fm_shifted_max"] if dense else ["fm_shifted"]
    problem = _problem(mappings, tmp_path)
    assert problem.times_at_changes(0).tolist() == [24.0]
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
    # the maximum of the dense output misses the state before the reset
    assert TRUTH["urine_max"] < TRUTH["urine"][1]
    assert after["fm_urine"][1] == pytest.approx(0.0, abs=1e-8)
    assert after["fm_twice"][1] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("dense", [False, True])
def test_a_change_at_the_end_is_compared_before_it(tmp_path: Path, dense: bool) -> None:
    """A change at the end of the simulation: the data at the end sees the state before."""
    mappings = ["fm_end", "fm_end_cmax"] if dense else ["fm_end"]
    before = _predictions(_problem(mappings, tmp_path))["fm_end"]
    after = _predictions(_problem(mappings, tmp_path, DataAtChangeType.AFTER))["fm_end"]
    assert before[0] == pytest.approx(TRUTH["urine"][1], rel=STEPS_RTOL)
    assert after[0] == pytest.approx(0.0, abs=1e-8)


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
    problem = _problem(["fm_urine", "fm_noisy", "fm_depot", "fm_cmax"], tmp_path)
    times = {
        problem.mapping_keys[k]: problem.times_at_changes(k).tolist()
        for k in problem.indices()
    }
    assert times == {
        "fm_urine": [24.0],
        "fm_noisy": [24.0 + NOISE],
        "fm_depot": [],
        "fm_cmax": [],
    }


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
    depot = _problem(["fm_depot"], tmp_path)
    assert "measurement-at-change" not in {gap.id for gap in gaps_of_problem(depot)}


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


def test_a_problem_without_the_extension_has_the_semantics_of_petab() -> None:
    """A PEtab problem of another tool compares the data after the change."""
    reader = PetabReader.__new__(PetabReader)
    reader.extension = None
    assert reader.settings.data_at_change is DataAtChangeType.AFTER
