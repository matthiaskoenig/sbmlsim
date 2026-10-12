"""An estimated parameter which a condition assigns to an entity of the model.

A problem of another tool, i.e. one without the `sbmlsim` extension, means
the change of its period: the value is the one of the estimated parameter at
the time of the period, after a pre-equilibration, and the identifier means
what the model means, a concentration based species is its concentration.
"""

from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import pandas as pd
import pytest
import yaml

from sbmlsim.fit import FitSettings
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2.reader import PetabReader
from tests.simulator.models import sbml

SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-10,
)

#: `S` relaxes to `k` with the rate 1, a concentration based species in `C`
RELAX = """
model relax
  compartment C = 2;
  species S in C;
  S = 0
  k = 0
  J: -> S; C * (k - S)
end
"""

TIMES = (0.0, 1.0, 4.0, 10.0)


def write_problem(
    directory: Path,
    periods: list[tuple[float, str]],
    conditions: list[tuple[str, str, str]],
    estimated: dict[str, float],
    with_extension: bool = False,
) -> Path:
    """Write a problem of the relaxation with the periods of one experiment.

    Args:
        directory: directory of the problem.
        periods: time and condition of every period of the experiment.
        conditions: id, target and value of every change.
        estimated: the estimated parameters with their nominal value.

    Returns:
        The YAML file of the problem.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.xml").write_text(sbml(RELAX))

    def table(name: str, rows: list[dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_csv(directory / f"{name}.tsv", sep="\t", index=False)

    table(
        "observables",
        [
            {
                "observableId": "s_o",
                "observableFormula": "S",
                "noiseFormula": 1.0,
                "noiseDistribution": "normal",
            }
        ],
    )
    table(
        "measurements",
        [
            {"observableId": "s_o", "experimentId": "e1", "measurement": 0.0, "time": t}
            for t in TIMES
        ],
    )
    table(
        "experiments",
        [
            {"experimentId": "e1", "time": time, "conditionId": condition}
            for time, condition in periods
        ],
    )
    table(
        "conditions",
        [
            {"conditionId": cid, "targetId": target, "targetValue": value}
            for cid, target, value in conditions
        ],
    )
    table(
        "parameters",
        [
            {
                "parameterId": pid,
                "lowerBound": 0.0,
                "upperBound": 10.0,
                "nominalValue": value,
                "estimate": True,
            }
            for pid, value in estimated.items()
        ],
    )
    path = directory / "problem.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "format_version": "2.0.0",
                "model_files": {"relax": {"location": "model.xml", "language": "sbml"}},
                "measurement_files": ["measurements.tsv"],
                "observable_files": ["observables.tsv"],
                "experiment_files": ["experiments.tsv"],
                "condition_files": ["conditions.tsv"],
                "parameter_files": ["parameters.tsv"],
            },
            sort_keys=False,
        )
    )
    if with_extension:
        config = yaml.safe_load(path.read_text())
        config["extensions"] = {"sbmlsim": {"version": "0.4.0", "required": True}}
        path.write_text(yaml.safe_dump(config))
    return path


def predictions(path: Path) -> np.ndarray:
    """Get the observable at the nominal values of the problem."""
    reader = PetabReader.from_yaml(path)
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    (prediction,) = problem.predictions(
        reader.nominal_parameters(problem).x(problem.pids)
    ).values()
    return prediction


@pytest.mark.parametrize("with_extension", [False, True])
def test_an_estimated_value_after_a_pre_equilibration(
    tmp_path: Path, with_extension: bool
) -> None:
    """The steady state of `k = 0` is `S = 0`, then `S` relaxes to `k_est`."""
    path = write_problem(
        tmp_path,
        periods=[(-np.inf, "pre"), (0.0, "stimulus")],
        conditions=[("pre", "k", "0"), ("stimulus", "k", "k_est")],
        estimated={"k_est": 2.0},
        with_extension=with_extension,
    )
    t = np.asarray(TIMES)
    np.testing.assert_allclose(
        predictions(path), 2.0 * (1 - np.exp(-t)), rtol=1e-6, atol=1e-9
    )


def test_an_estimated_value_of_a_later_period(tmp_path: Path) -> None:
    """`k_est` is the value from the start, `0` from the time 5 on."""
    path = write_problem(
        tmp_path,
        periods=[(0.0, "stimulus"), (5.0, "off")],
        conditions=[("stimulus", "k", "k_est"), ("off", "k", "0")],
        estimated={"k_est": 2.0},
    )
    t = np.asarray(TIMES)
    s5 = 2.0 * (1 - np.exp(-5.0))
    expected = np.where(t <= 5.0, 2.0 * (1 - np.exp(-t)), s5 * np.exp(-(t - 5.0)))
    np.testing.assert_allclose(predictions(path), expected, rtol=1e-6, atol=1e-9)


def test_an_estimated_value_of_a_concentration(tmp_path: Path) -> None:
    """`S` of a condition is the concentration of a concentration based species."""
    path = write_problem(
        tmp_path,
        periods=[(0.0, "start")],
        conditions=[("start", "S", "S0"), ("start", "k", "S0")],
        estimated={"S0": 5.0},
    )
    # the concentration stays at `k = S0 = 5`
    np.testing.assert_allclose(predictions(path), 5.0, rtol=1e-6)


@pytest.mark.parametrize("with_extension", [False, True])
def test_an_estimated_value_at_the_time_of_its_period(
    tmp_path: Path, with_extension: bool
) -> None:
    """The value of a later period applies from its time on, not before."""
    path = write_problem(
        tmp_path,
        periods=[(0.0, "off"), (3.0, "stimulus")],
        conditions=[("off", "k", "0"), ("stimulus", "k", "k_est")],
        estimated={"k_est": 2.0},
        with_extension=with_extension,
    )
    t = np.asarray(TIMES)
    expected = np.where(t <= 3.0, 0.0, 2.0 * (1 - np.exp(-(t - 3.0))))
    np.testing.assert_allclose(predictions(path), expected, rtol=1e-6, atol=1e-9)


def _write_two_model_problem(tmp_path: Path, with_extension: bool) -> Path:
    """Write a shared condition for concentration and amount based species."""
    path = write_problem(
        tmp_path,
        periods=[(0.0, "start")],
        conditions=[("start", "S", "5"), ("start", "k", "5")],
        estimated={"k_est": 2.0},
        with_extension=with_extension,
    )
    document = libsbml.readSBMLFromFile(str(tmp_path / "model.xml"))
    document.getModel().getSpecies("S").setHasOnlySubstanceUnits(True)
    libsbml.writeSBMLToFile(document, str(tmp_path / "amount.xml"))
    config = yaml.safe_load(path.read_text())
    config["model_files"]["amount"] = {"location": "amount.xml", "language": "sbml"}
    path.write_text(yaml.safe_dump(config))
    measurements = pd.read_csv(tmp_path / "measurements.tsv", sep="\t")
    measurements["modelId"] = "relax"
    second = measurements.assign(experimentId="e2", modelId="amount")
    pd.concat([measurements, second]).to_csv(
        tmp_path / "measurements.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        [
            {"experimentId": experiment, "time": 0.0, "conditionId": "start"}
            for experiment in ("e1", "e2")
        ]
    ).to_csv(tmp_path / "experiments.tsv", sep="\t", index=False)
    return path


@pytest.mark.parametrize("with_extension", [False, True])
@pytest.mark.parametrize("formula, expected", [("S", 5.0), ("2*S", 10.0)])
def test_shared_conditions_resolve_species_per_model(
    tmp_path: Path, with_extension: bool, formula: str, expected: float
) -> None:
    """A shared condition, observable and noise retain each model's semantics."""
    path = _write_two_model_problem(tmp_path, with_extension)
    observables = pd.read_csv(tmp_path / "observables.tsv", sep="\t")
    observables["observableFormula"] = formula
    observables["noiseFormula"] = "S"
    observables.to_csv(tmp_path / "observables.tsv", sep="\t", index=False)
    reader = PetabReader.from_yaml(path)
    assert reader.noise_model("s_o_e1").formula == "[S]"
    assert reader.noise_model("s_o_e2").formula == "S"
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    for prediction in problem.predictions(
        reader.nominal_parameters(problem).x(problem.pids)
    ).values():
        np.testing.assert_allclose(prediction, expected, rtol=1e-6)


def test_versioned_species_shared_across_models_keep_their_semantics(
    tmp_path: Path,
) -> None:
    """A shared estimated initial value uses each model's species selection."""
    path = _write_two_model_problem(tmp_path, with_extension=True)
    pd.DataFrame(
        [
            {"conditionId": "start", "targetId": "S", "targetValue": "k_est"},
            {"conditionId": "start", "targetId": "k", "targetValue": "2"},
        ]
    ).to_csv(tmp_path / "conditions.tsv", sep="\t", index=False)
    reader = PetabReader.from_yaml(path)
    assert reader._versions() == {}
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    for prediction in problem.predictions(
        reader.nominal_parameters(problem).x(problem.pids)
    ).values():
        np.testing.assert_allclose(prediction, 2.0, rtol=1e-6)


def test_conflicting_models_in_one_experiment_are_rejected(tmp_path: Path) -> None:
    """The reader does not silently choose one of conflicting measurement models."""
    path = _write_two_model_problem(tmp_path, with_extension=False)
    measurements = pd.read_csv(tmp_path / "measurements.tsv", sep="\t")
    measurements.loc[0, "modelId"] = "amount"
    measurements.to_csv(tmp_path / "measurements.tsv", sep="\t", index=False)
    reader = PetabReader.from_yaml(path)
    with pytest.raises(ValueError, match="experiment 'e1' uses multiple models"):
        reader.to_optimization_problem().initialize(SETTINGS)
