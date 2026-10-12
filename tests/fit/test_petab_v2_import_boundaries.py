"""Imported definitions serialize independently and report conflicting tables."""

import logging
import pickle
from pathlib import Path

import numpy as np
import petab.v2 as petab_v2
import pytest
from petab.v2.math import sympify_petab

from sbmlsim import Q
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit.petab_v2.export import _table, to_petab
from sbmlsim.fit.petab_v2.extension import SbmlsimExtension
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.runner import run_optimization
from sbmlsim.simulation import Change, Simulation, SteadyState
from tests.fit.test_petab_v2_reader import SETTINGS, write_problem


@pytest.mark.parametrize("initialized", [False, True])
def test_imported_definitions_with_same_class_name_pickle_independently(
    tmp_path: Path, initialized: bool
) -> None:
    """Equal generated names never share or overwrite the bound data."""
    problems = [
        PetabReader.from_yaml(
            write_problem(
                tmp_path / str(k),
                {"prey_o": "prey"},
                {"e1": "c1"},
                conditions=[("c1", "delta", str(value))],
            )
        ).to_optimization_problem()
        for k, value in enumerate([1.8, 3.6])
    ]
    if initialized:
        for problem in problems:
            problem.initialize(SETTINGS)
    restored = pickle.loads(pickle.dumps(problems))
    classes = [p.mapping_collections[0].experiment_class for p in restored]
    assert classes[0].__name__ == classes[1].__name__
    assert classes[0] is not classes[1]
    assert all(issubclass(cls, SimulationExperiment) for cls in classes)
    predictions = []
    for original, copy in zip(problems, restored, strict=True):
        original.initialize(SETTINGS)
        copy.initialize(SETTINGS)
        prediction = copy.predictions(np.asarray(copy.x0, dtype=float))[0]
        np.testing.assert_allclose(
            prediction, original.predictions(np.asarray(original.x0, dtype=float))[0]
        )
        predictions.append(prediction)
    assert not np.allclose(*predictions)
    # Serializing an unused restored class must not require runtime state.
    uninitialized_class = pickle.loads(pickle.dumps(classes[0]))
    assert (
        pickle.loads(pickle.dumps(uninitialized_class)).__name__ == classes[0].__name__
    )


def test_parallel_fit_of_imported_problem(tmp_path: Path) -> None:
    """A worker rebuilds its imported definition without a generated global."""
    reader = PetabReader.from_yaml(
        write_problem(tmp_path, {"prey_o": "prey"}, {"e1": None})
    )
    result = run_optimization(
        reader.to_optimization_problem(),
        settings=SETTINGS,
        size=2,
        n_cores=2,
        seed=1,
        show_progress=False,
        max_nfev=2,
    )
    assert len(result.fits) == 2
    assert all(np.isfinite(fit.cost) for fit in result.fits)
    assert all("Error" not in fit.message for fit in result.fits)


@pytest.mark.parametrize("edit", ["condition", "time", "remove_condition", "extension"])
def test_conflicting_standard_simulation_warns_with_ids(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, edit: str
) -> None:
    """The extension stays authoritative, but conflicting edits are visible."""
    reader = PetabReader.from_yaml(
        write_problem(
            tmp_path / "input",
            {"prey_o": "prey"},
            {"e1": "c1"},
            conditions=[("c1", "delta", "1.8")],
        )
    )
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    restored = PetabReader.from_yaml(to_petab(problem, tmp_path / "export"))
    experiment = restored.petab_problem.experiments[0]
    condition_id = experiment.periods[0].condition_ids[0]
    if edit == "condition":
        restored.petab_problem.conditions[0].changes[0].target_value = sympify_petab(
            "9.0"
        )
    elif edit == "time":
        experiment.periods[0].time = 0.5
    elif edit == "remove_condition":
        restored.petab_problem.conditions[0].changes.clear()
    else:
        assert restored.extension is not None
        restored.extension.experiments[experiment.id]["simulation"]["preinit_changes"][
            "delta"
        ] = 9.0
    with caplog.at_level(logging.WARNING):
        simulation = restored.simulations()[experiment.id]
        restored.simulations()
    messages = [
        record.message for record in caplog.records if "disagree with" in record.message
    ]
    assert len(messages) == 1
    assert experiment.id in messages[0]
    assert condition_id in messages[0]
    assert simulation.preinit_changes["delta"] == (9.0 if edit == "extension" else 1.8)


def test_equivalent_units_output_and_formulas_do_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Comparison uses model units and ignores extension-only output metadata."""
    reader = PetabReader.from_yaml(
        write_problem(tmp_path, {"prey_o": "prey"}, {"e1": None})
    )
    reader.extension = SbmlsimExtension(
        experiments={
            "e1": {
                "simulation": Simulation(
                    start=Q(0, "min"),
                    end=Q(1 / 6, "min"),
                    times=[Q(0, "min"), Q(1 / 6, "min")],
                    presimulation=SteadyState(preinit_changes={"delta": 1.8}),
                    changes=[Change(Q(1 / 12, "min"), {"delta": "alpha + beta"})],
                ).to_dict()
            }
        }
    )
    reader.petab_problem.experiments[0].periods = [
        petab_v2.ExperimentPeriod(time=float("-inf"), condition_ids=["pre"]),
        petab_v2.ExperimentPeriod(time=0.0),
        petab_v2.ExperimentPeriod(time=5.0, condition_ids=["later"]),
    ]
    _table(reader.petab_problem, "condition_tables").elements.extend(
        [
            petab_v2.Condition(
                id="pre",
                changes=[petab_v2.Change(target_id="delta", target_value="1.8")],
            ),
            petab_v2.Condition(
                id="later",
                changes=[
                    petab_v2.Change(target_id="delta", target_value="beta + alpha")
                ],
            ),
        ]
    )
    with caplog.at_level(logging.WARNING):
        simulation = reader.simulations()["e1"]
    assert simulation.times is not None
    assert not [r for r in caplog.records if "standard" in r.message]


def test_a_comparison_which_fails_only_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check of the standard tables never stops a problem from being read."""
    reader = PetabReader.from_yaml(
        write_problem(tmp_path, {"prey_o": "prey"}, {"e1": None})
    )
    reader.extension = SbmlsimExtension(
        experiments={
            "e1": {
                "simulation": Simulation(
                    start=0,
                    end=10,
                    changes=[Change(5, {"delta": "alpha + beta"})],
                ).to_dict()
            }
        }
    )
    reader.petab_problem.experiments[0].periods = [
        petab_v2.ExperimentPeriod(time=0.0),
        petab_v2.ExperimentPeriod(time=5.0, condition_ids=["later"]),
    ]
    _table(reader.petab_problem, "condition_tables").elements.append(
        petab_v2.Condition(
            id="later",
            changes=[petab_v2.Change(target_id="delta", target_value="beta + alpha")],
        )
    )

    def unreadable(formula: str) -> None:
        raise ValueError(f"cannot read '{formula}'")

    monkeypatch.setattr(
        "sbmlsim.fit.petab_v2.reader.sympify_petab", unreadable, raising=True
    )
    with caplog.at_level(logging.WARNING):
        simulation = reader.simulations()["e1"]
    assert simulation.changes[0].values == {"delta": "alpha + beta"}
    messages = [r.message for r in caplog.records if "cannot be compared" in r.message]
    assert len(messages) == 1
    assert "e1" in messages[0]
    assert "cannot read" in messages[0]
