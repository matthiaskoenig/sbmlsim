"""Tests of the reader on a small problem: its observables and experiments.

The problem is the model of Lotka and Volterra of `tests/data/models` with
tables which are written here, so that a test controls every row.
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

MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "lotka_volterra.xml"

SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    variable_step_size=False,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-12,
)

#: the species at the times 1 to 10, simulated with the model
PREY = [0.1996, 0.4843, 1.6064, 5.4941, 3.0782, 0.1952, 0.2965, 0.9041, 3.1091, 8.8516]


def write_problem(
    directory: Path,
    observables: dict[str, str],
    experiments: dict[str, str | None],
    conditions: list[tuple[str, str, str]] = (),  # ty: ignore[invalid-parameter-default]
) -> Path:
    """Write a PEtab problem of the model with the observables and experiments.

    Every observable is measured in every experiment at the times `1` to
    `10`, with the values of `prey`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lv.xml").write_bytes(MODEL_PATH.read_bytes())

    def table(name: str, rows: list[dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_csv(directory / f"{name}.tsv", sep="\t", index=False)

    table(
        "observables",
        [
            {
                "observableId": sid,
                "observableFormula": formula,
                "noiseFormula": 0.05,
                "noiseDistribution": "normal",
            }
            for sid, formula in observables.items()
        ],
    )
    table(
        "measurements",
        [
            {
                "observableId": sid,
                "experimentId": experiment_id,
                "measurement": value,
                "time": float(k + 1),
            }
            for experiment_id in experiments
            for sid in observables
            for k, value in enumerate(PREY)
        ],
    )
    table(
        "experiments",
        [
            {"experimentId": sid, "time": 0.0, "conditionId": condition or ""}
            for sid, condition in experiments.items()
        ],
    )
    table(
        "parameters",
        [
            {
                "parameterId": sid,
                "lowerBound": 0.0,
                "upperBound": 15.0,
                "nominalValue": value,
                "estimate": True,
            }
            for sid, value in [("alpha", 1.3), ("beta", 0.9)]
        ],
    )
    config: dict[str, Any] = {
        "format_version": "2.0.0",
        "model_files": {"lv": {"location": "lv.xml", "language": "sbml"}},
        "measurement_files": ["measurements.tsv"],
        "observable_files": ["observables.tsv"],
        "experiment_files": ["experiments.tsv"],
        "parameter_files": ["parameters.tsv"],
    }
    if conditions:
        table(
            "conditions",
            [
                {"conditionId": condition, "targetId": target, "targetValue": value}
                for condition, target, value in conditions
            ],
        )
        config["condition_files"] = ["conditions.tsv"]
    path = directory / "problem.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_an_observable_measured_in_several_experiments(tmp_path: Path) -> None:
    """One fit mapping per observable and experiment, named after both."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey", "predator_o": "predator"},
        experiments={"e1": "cond1", "e2": "cond2"},
        conditions=[("cond1", "delta", "1.8"), ("cond2", "delta", "3.6")],
    )
    reader = PetabReader.from_yaml(path)
    problem = reader.to_optimization_problem()
    assert [c.sid for c in problem.mapping_collections] == ["e1", "e2"]
    assert problem.mapping_collections[0].mappings == ["prey_o_e1", "predator_o_e1"]
    assert problem.mapping_collections[1].mappings == ["prey_o_e2", "predator_o_e2"]
    assert reader.observable_id("prey_o_e2") == "prey_o"
    with pytest.raises(ValueError, match="no fit mapping 'prey_o'"):
        reader.observable_id("prey_o")

    problem.initialize(SETTINGS)
    assert problem.simulation_keys == ["e1", "e1", "e2", "e2"]
    for k in problem.indices():
        assert len(problem.y_references[k]) == 10
    x = np.asarray(problem.x0, dtype=float)
    predictions = problem.predictions(x, indices=problem.indices())
    # the experiments differ in their condition
    assert not np.allclose(predictions[0], predictions[2])
    assert len(problem.noise_models) == 4
    assert all(noise is not None for noise in problem.noise_models)


def test_an_observable_measured_in_one_experiment_keeps_its_id(
    tmp_path: Path,
) -> None:
    """A problem with one experiment per observable reads as before."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey", "predator_o": "predator"},
        experiments={"e1": None},
    )
    problem = PetabReader.from_yaml(path).to_optimization_problem()
    assert problem.mapping_collections[0].mappings == ["prey_o", "predator_o"]


def test_a_mapping_named_like_an_observable(tmp_path: Path) -> None:
    """The key of a mapping is not the id of another observable."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey", "prey_o_e1": "predator"},
        experiments={"e1": None, "e2": None},
    )
    with pytest.raises(ValueError, match="'prey_o_e1', which is the id of another"):
        PetabReader.from_yaml(path)


def _add_sbmlsim_block(path: Path, parameters: dict[str, dict[str, Any]]) -> None:
    """Add a block of the `sbmlsim` extension with the info of the parameters."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["extensions"] = {
        "sbmlsim": {"version": "0.1.0", "required": True, "parameters": parameters}
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_the_scale_of_a_parameter_of_the_extension(tmp_path: Path) -> None:
    """The `sbmlsim` block carries the scale of a parameter by its name."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey", "predator_o": "predator"},
        experiments={"e1": None},
    )
    _add_sbmlsim_block(path, {"alpha": {"scale": "LOG"}, "beta": {"unit": None}})
    by_id = {p.pid: p for p in PetabReader.from_yaml(path).fit_parameters()}
    assert by_id["alpha"].scale is ParameterScaleType.LOG
    # a parameter without a scale has the scale of the settings
    assert by_id["beta"].scale is None


def test_a_scale_of_the_extension_which_is_not_one(tmp_path: Path) -> None:
    """The error names the parameter and the scales."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey"},
        experiments={"e1": None},
    )
    _add_sbmlsim_block(path, {"alpha": {"scale": "lin"}})
    with pytest.raises(ValueError, match=r"'alpha'.*'lin'.*\['LINEAR', 'LOG', 'LOG10'"):
        PetabReader.from_yaml(path).fit_parameters()


def test_the_math_of_an_observable_is_translated(tmp_path: Path) -> None:
    """`log` of PEtab is the natural logarithm, `log` of SBML the decadic one."""
    path = write_problem(
        tmp_path,
        observables={
            "log_prey": "log(prey)",
            "log10_prey": "log10(prey)",
            "square": "prey^2",
            "prey_o": "prey",
        },
        experiments={"e1": None},
    )
    reader = PetabReader.from_yaml(path)
    reader.derived_dir = tmp_path / "derived"
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    predictions = problem.predictions(np.asarray(problem.x0, dtype=float))
    keys = problem.mapping_keys
    prey = predictions[keys.index("prey_o")]
    np.testing.assert_allclose(predictions[keys.index("log_prey")], np.log(prey))
    np.testing.assert_allclose(predictions[keys.index("log10_prey")], np.log10(prey))
    np.testing.assert_allclose(predictions[keys.index("square")], prey**2)
