"""An estimated parameter which a condition assigns to an entity of the model.

A problem of another tool, i.e. one without the `sbmlsim` extension, means
the change of its period: the value is the one of the estimated parameter at
the time of the period, after a pre-equilibration, and the identifier means
what the model means, a concentration based species is its concentration.
"""

from pathlib import Path
from typing import Any

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


def test_an_estimated_value_after_a_pre_equilibration(tmp_path: Path) -> None:
    """The steady state of `k = 0` is `S = 0`, then `S` relaxes to `k_est`."""
    path = write_problem(
        tmp_path,
        periods=[(-np.inf, "pre"), (0.0, "stimulus")],
        conditions=[("pre", "k", "0"), ("stimulus", "k", "k_est")],
        estimated={"k_est": 2.0},
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


@pytest.mark.parametrize("time", [3.0])
def test_an_estimated_value_at_the_time_of_its_period(
    tmp_path: Path, time: float
) -> None:
    """The value of a later period applies from its time on, not before."""
    path = write_problem(
        tmp_path,
        periods=[(0.0, "off"), (time, "stimulus")],
        conditions=[("off", "k", "0"), ("stimulus", "k", "k_est")],
        estimated={"k_est": 2.0},
    )
    t = np.asarray(TIMES)
    expected = np.where(t <= time, 0.0, 2.0 * (1 - np.exp(-(t - time))))
    np.testing.assert_allclose(predictions(path), expected, rtol=1e-6, atol=1e-9)
