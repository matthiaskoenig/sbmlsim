"""Tests of the reader on a small problem: its observables and experiments.

The problem is the model of Lotka and Volterra of `tests/data/models` with
tables which are written here, so that a test controls every row.
"""

from collections.abc import Callable, Mapping, Sequence
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
    conditions: Sequence[tuple[str, str, str]] = (),
    measured: Mapping[str, Sequence[str]] | None = None,
    noise_parameters: Callable[[str, int], str] | None = None,
) -> Path:
    """Write a PEtab problem of the model with the observables and experiments.

    Every observable is measured in the experiments of `measured`, in every
    experiment by default, at the times `1` to `10`, with the values of
    `prey`. With `noise_parameters` the noise formula of the observables is
    the placeholder `sd`, and the function gives the noise parameter of the
    measurement of an experiment at the index of its time.
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
                "noiseFormula": 0.05 if noise_parameters is None else "sd",
                "noisePlaceholders": "" if noise_parameters is None else "sd",
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
                "noiseParameters": ""
                if noise_parameters is None
                else noise_parameters(experiment_id, k),
            }
            for sid in observables
            for experiment_id in (measured[sid] if measured else experiments)
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


def test_derived_keys_do_not_collide(tmp_path: Path) -> None:
    """Two observables in two experiments which give one key are refused."""
    path = write_problem(
        tmp_path,
        observables={"a": "prey", "a_b": "predator"},
        experiments={"b_c": None, "x": None, "c": None, "y": None},
        measured={"a": ["b_c", "x"], "a_b": ["c", "y"]},
    )
    with pytest.raises(ValueError, match=r"'a_b_c'.*\('a', 'b_c'\).*\('a_b', 'c'\)"):
        PetabReader.from_yaml(path)


def test_the_noise_model_of_a_key_which_is_no_mapping(tmp_path: Path) -> None:
    """The observable of a multi-experiment problem is not a fit mapping."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey"},
        experiments={"e1": None, "e2": None},
    )
    reader = PetabReader.from_yaml(path)
    with pytest.raises(ValueError, match="no fit mapping 'prey_o'"):
        reader.noise_model("prey_o")
    with pytest.raises(ValueError, match="no fit mapping 'nothing'"):
        reader.noise_model("nothing")


def test_the_noise_parameters_follow_the_experiment(tmp_path: Path) -> None:
    """The placeholder values of a fit mapping are the ones of its experiment."""
    path = write_problem(
        tmp_path,
        observables={"prey_o": "prey"},
        experiments={"e1": None, "e2": None},
        noise_parameters=lambda experiment_id, k: str(
            (0.1 if experiment_id == "e1" else 0.2) + 0.01 * k
        ),
    )
    reader = PetabReader.from_yaml(path)
    for key, offset in [("prey_o_e1", 0.1), ("prey_o_e2", 0.2)]:
        noise = reader.noise_model(key)
        assert noise.placeholders == ("sd",)
        values = np.array(noise.placeholder_values, dtype=float)
        np.testing.assert_allclose(values, [[offset + 0.01 * k] for k in range(10)])


def test_the_model_source(tmp_path: Path) -> None:
    """The model of the fit is the model of the problem, or the one with the observables."""
    path = write_problem(
        tmp_path / "entities",
        observables={"prey_o": "prey"},
        experiments={"e1": None},
    )
    reader = PetabReader.from_yaml(path)
    assert reader.model_source() == reader.model_source("lv")
    assert reader.model_source().name == "lv.xml"
    with pytest.raises(ValueError, match="no model 'other'"):
        reader.model_source("other")

    path = write_problem(
        tmp_path / "formulas",
        observables={"log_prey": "log(prey)"},
        experiments={"e1": None},
    )
    reader = PetabReader.from_yaml(path)
    reader.derived_dir = tmp_path / "derived"
    source = reader.model_source()
    assert source == tmp_path / "derived" / "lv_observables.xml"
    assert source.exists()
    assert reader.model_source("lv") == source
