"""Tests of the reader on a small problem: its observables and experiments.

The problem is the model of Lotka and Volterra of `tests/data/models` with
tables which are written here, so that a test controls every row.
"""

import logging
import re
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


def test_a_formula_observable_is_an_observable_model(tmp_path: Path) -> None:
    """A formula is evaluated on the simulation, the model is not rewritten."""
    path = write_problem(
        tmp_path,
        observables={"total": "prey + predator", "prey_o": "prey"},
        experiments={"e1": None},
    )
    reader = PetabReader.from_yaml(path)
    assert reader.model_source().name == "lv.xml"
    problem = reader.to_optimization_problem()
    problem.initialize(SETTINGS)
    total = problem.observable_models[problem.mapping_keys.index("total")]
    assert total is not None
    assert total.symbols == ("predator", "prey")
    assert problem.observable_models[problem.mapping_keys.index("prey_o")] is None
    assert problem.yid_observable[problem.mapping_keys.index("total")] == "total"


def _with_observable_parameters(
    directory: Path, placeholders: str, values: Callable[[int], str]
) -> None:
    """Give the observables placeholders and the measurements their values."""
    observables = pd.read_csv(directory / "observables.tsv", sep="\t")
    observables["observablePlaceholders"] = placeholders
    observables.to_csv(directory / "observables.tsv", sep="\t", index=False)
    measurements = pd.read_csv(directory / "measurements.tsv", sep="\t")
    measurements["observableParameters"] = [values(k) for k in range(len(measurements))]
    measurements.to_csv(directory / "measurements.tsv", sep="\t", index=False)


def test_the_placeholders_of_an_observable_per_measurement(tmp_path: Path) -> None:
    """Every measurement has its own values of the placeholders (case 0006).

    A value is a number or a parameter, here `alpha`, which the fit estimates.
    """
    reference = PetabReader.from_yaml(
        write_problem(tmp_path / "reference", {"prey_o": "prey"}, {"e1": None})
    ).to_optimization_problem()
    reference.initialize(SETTINGS)

    path = write_problem(
        tmp_path / "placeholders",
        observables={"prey_o": "scale_prey * prey + offset_prey"},
        experiments={"e1": None},
    )
    _with_observable_parameters(
        tmp_path / "placeholders",
        "scale_prey;offset_prey",
        lambda k: f"{2.0 if k < 5 else 3.0};{'alpha' if k % 2 else 0.5}",
    )
    problem = PetabReader.from_yaml(path).to_optimization_problem()
    problem.initialize(SETTINGS)

    x = np.asarray(problem.x0, dtype=float)
    alpha = x[problem.pids.index("alpha")]
    prey = reference.predictions(x)[0]
    k = np.arange(prey.size)
    expected = np.where(k < 5, 2.0, 3.0) * prey + np.where(k % 2, alpha, 0.5)
    np.testing.assert_allclose(problem.predictions(x)[0], expected, rtol=1e-8)


def test_the_model_source(tmp_path: Path) -> None:
    """The model of the fit is the model of the problem."""
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


def test_a_prior_of_a_parameter_is_dropped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_problem(tmp_path / "problem", {"prey_o": "prey"}, {"e1": None})
    table = tmp_path / "problem" / "parameters.tsv"
    parameters = pd.read_csv(table, sep="\t")
    # sigma is estimated but is no entity of the model, i.e. it is dropped
    sigma = parameters.iloc[[0]].copy()
    sigma["parameterId"] = "sigma"
    parameters = pd.concat([parameters, sigma], ignore_index=True)
    parameters["priorDistribution"] = ["normal", "", "normal"]
    parameters["priorParameters"] = ["1.0;0.5", "", "1.0;0.5"]
    parameters.to_csv(table, sep="\t", index=False)
    with caplog.at_level(logging.WARNING, logger="sbmlsim.fit.petab_v2.reader"):
        fit_parameters = PetabReader.from_yaml(path).fit_parameters()
    assert [p.pid for p in fit_parameters] == ["alpha", "beta"]
    priors = [
        r.getMessage() for r in caplog.records if "gap 'priors'" in r.getMessage()
    ]
    assert len(priors) == 2
    assert any("The parameter 'alpha' has the prior" in m for m in priors)
    assert any("The parameter 'sigma' has the prior" in m for m in priors)
    assert not any("'beta'" in m for m in priors)


def _read_with_base_path(tmp_path: Path, base_path: str) -> PetabReader:
    """Read the problem with `petab` and give its configuration `base_path`."""
    from petab.v2 import Problem as PetabProblem
    from pydantic import AnyUrl

    path = write_problem(
        tmp_path, observables={"prey_o": "prey"}, experiments={"e1": None}
    )
    petab_problem = PetabProblem.from_yaml(path)
    # the URL `petab` makes of a location which is not a path of this machine
    petab_problem.config.base_path = AnyUrl(base_path)
    return PetabReader(petab_problem)


def test_the_base_path_of_the_configuration_as_a_file_url(tmp_path: Path) -> None:
    """`petab` keeps a `file` URL as a URL, the reader reads the directory of it."""
    reader = _read_with_base_path(tmp_path, tmp_path.as_uri())
    assert reader.base_path == tmp_path
    assert reader.to_optimization_problem().mapping_collections


def test_the_base_path_of_the_configuration_with_a_drive() -> None:
    """`petab` parses a path of windows as a URL whose scheme is the drive."""
    from petab.v2.core import ProblemConfig
    from pydantic import AnyUrl

    from sbmlsim.fit.petab_v2.reader import _local_path

    config = ProblemConfig(base_path="C:\\Users\\runner\\problem")
    assert isinstance(config.base_path, AnyUrl)
    assert _local_path(config.base_path) == Path("c:\\Users\\runner\\problem")


def test_the_base_path_of_the_configuration_on_a_server(tmp_path: Path) -> None:
    """The files of a problem are read from a directory, not from a server."""
    with pytest.raises(ValueError, match=re.escape("'https://example.org/problem'")):
        _read_with_base_path(tmp_path, "https://example.org/problem")
