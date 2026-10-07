"""Tests of reading and running a case of the PEtab v2 test suite.

The cases are written by the tests, nothing is downloaded.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import yaml

from sbmlsim.fit.petab_v2.likelihood import chi2, log_likelihood
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.petab_v2.testsuite import (
    CASE_SETTINGS,
    MATH,
    PETAB_SUITE_COMMIT,
    SBML,
    CaseStatus,
    MathCase,
    PetabCase,
    PetabSuite,
    math_cases,
)
from sbmlsim.testsuite import cache
from tests.fit.test_petab_v2_reader import write_problem


def _case(directory: Path, cid: str = "9001", **solution: Any) -> Path:
    """Write a case of the Lotka-Volterra model whose solution is its result.

    Args:
        directory: directory of the suite, i.e. of the case directories.
        cid: number of the case.
        solution: entries which replace the ones of the solution.

    Returns:
        The directory of the case.
    """
    case_dir = directory / cid
    path = write_problem(case_dir, {"prey_o": "prey"}, {"e1": None})
    path.rename(case_dir / f"_{cid}.yaml")

    reader = PetabReader.from_yaml(case_dir / f"_{cid}.yaml")
    problem = reader.to_optimization_problem()
    problem.initialize(CASE_SETTINGS)
    nominal = reader.nominal_parameters(problem)
    (prediction,) = problem.predictions(nominal.x(problem.pids)).values()
    simulations = pd.read_csv(case_dir / "measurements.tsv", sep="\t")
    simulations = simulations.rename(columns={"measurement": "simulation"})
    simulations["simulation"] = prediction
    simulations.to_csv(case_dir / "_simulations.tsv", sep="\t", index=False)

    content = {
        "llh": log_likelihood(problem, nominal),
        "chi2": chi2(problem, nominal),
        "simulation_files": ["_simulations.tsv"],
        "tol_llh": 1e-3,
        "tol_chi2": 1e-3,
        "tol_simulations": 1e-3,
    }
    content.update(solution)
    (case_dir / f"_{cid}_solution.yaml").write_text(yaml.safe_dump(content))
    return case_dir


def test_a_case_is_read(tmp_path: Path) -> None:
    """A case is its problem and its solution."""
    case = PetabCase.from_directory(_case(tmp_path))
    assert case.cid == "9001"
    assert case.key == f"{SBML}/9001"
    assert case.problem_file.name == "_9001.yaml"
    assert case.solution["tol_llh"] == 1e-3


def test_a_case_passes(tmp_path: Path) -> None:
    """A case whose values agree with the solution passes."""
    result = PetabCase.from_directory(_case(tmp_path)).run()
    assert result.status is CaseStatus.PASS, result.message
    assert result.key == f"{SBML}/9001"
    assert result.max_difference is not None
    assert result.max_difference < 1e-9


@pytest.mark.parametrize("entry", ["llh", "chi2"])
def test_a_value_outside_of_the_tolerance(tmp_path: Path, entry: str) -> None:
    """The message names the value which differs."""
    case_dir = _case(tmp_path)
    solution_file = case_dir / "_9001_solution.yaml"
    solution = yaml.safe_load(solution_file.read_text())
    solution[entry] += 1.0
    solution_file.write_text(yaml.safe_dump(solution))
    result = PetabCase.from_directory(case_dir).run()
    assert result.status is CaseStatus.TOLERANCE
    assert entry in result.message
    assert result.max_difference == pytest.approx(1.0, abs=1e-6)


def test_a_simulation_outside_of_the_tolerance(tmp_path: Path) -> None:
    """A simulation which differs is named with its observable and experiment."""
    case_dir = _case(tmp_path)
    simulations = pd.read_csv(case_dir / "_simulations.tsv", sep="\t")
    simulations.loc[3, "simulation"] += 0.5
    simulations.to_csv(case_dir / "_simulations.tsv", sep="\t", index=False)
    result = PetabCase.from_directory(case_dir).run()
    assert result.status is CaseStatus.TOLERANCE
    assert "prey_o" in result.message
    assert "e1" in result.message


def test_the_priors_of_a_case(tmp_path: Path) -> None:
    """The log prior of every parameter and the log posterior are compared."""
    case_dir = _case(tmp_path)
    parameters = pd.read_csv(case_dir / "parameters.tsv", sep="\t")
    parameters["priorDistribution"] = ["normal", ""]
    parameters["priorParameters"] = ["1.0;0.5", ""]
    parameters.to_csv(case_dir / "parameters.tsv", sep="\t", index=False)
    solution_file = case_dir / "_9001_solution.yaml"
    solution = yaml.safe_load(solution_file.read_text())
    # the normal distribution truncated at the bounds [0, 15]
    alpha = scipy.stats.truncnorm.logpdf(
        1.3, (0.0 - 1.0) / 0.5, (15.0 - 1.0) / 0.5, loc=1.0, scale=0.5
    )
    solution["log_prior"] = {"alpha": float(alpha), "beta": float(-np.log(15.0))}
    solution["tol_log_prior"] = 1e-10
    solution["unnorm_log_posterior"] = (
        solution["llh"] + solution["log_prior"]["alpha"] + solution["log_prior"]["beta"]
    )
    solution["tol_unnorm_log_posterior"] = 1e-3
    solution_file.write_text(yaml.safe_dump(solution))
    result = PetabCase.from_directory(case_dir).run()
    assert result.status is CaseStatus.PASS, result.message

    solution["log_prior"]["beta"] = 0.0
    solution_file.write_text(yaml.safe_dump(solution))
    result = PetabCase.from_directory(case_dir).run()
    assert result.status is CaseStatus.TOLERANCE
    assert "beta" in result.message


def test_a_case_which_cannot_be_read(tmp_path: Path) -> None:
    """An error while the case is read or run is the status `error`."""
    case_dir = _case(tmp_path)
    (case_dir / "observables.tsv").write_text("nothing\n")
    result = PetabCase.from_directory(case_dir).run()
    assert result.status is CaseStatus.ERROR
    assert result.message


def test_a_directory_which_is_not_a_case(tmp_path: Path) -> None:
    """A directory without a solution is refused with its path."""
    (tmp_path / "9001").mkdir()
    with pytest.raises(ValueError, match="9001"):
        PetabCase.from_directory(tmp_path / "9001")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("1 + 2 * 3", 7.0), ("log(exp(1))", 1.0), ("-inf", -np.inf), ("!!2", 1.0)],
)
def test_a_math_case_passes(expression: str, expected: float) -> None:
    """A numeric expression is compiled and evaluated."""
    result = MathCase(index=0, expression=expression, expected=expected).run()
    assert result.status is CaseStatus.PASS, result.message
    assert result.key == f"{MATH}/000"


def test_a_symbolic_math_case_passes() -> None:
    """A symbolic expression is compared on values of its symbols."""
    result = MathCase(index=3, expression="b * a", expected="a * b").run()
    assert result.status is CaseStatus.PASS, result.message


def test_a_math_case_which_differs() -> None:
    """A value which differs is the status `tolerance`."""
    result = MathCase(index=1, expression="1 + 2", expected=4.0).run()
    assert result.status is CaseStatus.TOLERANCE
    assert "1 + 2" in result.message


def test_a_math_case_which_is_not_math() -> None:
    """An expression which is not math is the status `error`."""
    result = MathCase(index=2, expression="1 +", expected=1.0).run()
    assert result.status is CaseStatus.ERROR


def test_the_math_cases_of_a_file(tmp_path: Path) -> None:
    """The cases of `math_tests.yaml` are numbered in their order."""
    path = tmp_path / "math_tests.yaml"
    path.write_text(
        yaml.safe_dump(
            {"cases": [{"expression": "1", "expected": 1.0}, {"expression": "a"}]}
        )
    )
    with pytest.raises(ValueError, match="expected"):
        math_cases(path)
    path.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {"expression": "1", "expected": 1.0},
                    {"expression": "2", "expected": 2},
                ]
            }
        )
    )
    cases = math_cases(path)
    assert [case.index for case in cases] == [0, 1]
    assert [case.expected for case in cases] == [1.0, 2]


def test_the_suite_iterates_its_cases(tmp_path: Path) -> None:
    """The suite runs the cases of the models of SBML and the math cases."""
    _case(tmp_path / SBML, "9001")
    (tmp_path / SBML / "README.md").write_text("not a case")
    (tmp_path / MATH).mkdir()
    (tmp_path / MATH / "math_tests.yaml").write_text(
        yaml.safe_dump({"cases": [{"expression": "1", "expected": 1.0}]})
    )
    suite = PetabSuite(path=tmp_path, commit="abc")
    assert suite.case_ids() == ["9001"]
    results = suite.run()
    assert [result.key for result in results] == [f"{SBML}/9001", f"{MATH}/000"]
    assert all(result.passed for result in results)


def test_the_cache_is_per_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache holds one directory per commit, the variable overrides it."""
    monkeypatch.delenv("SBMLSIM_PETAB_SUITE_PATH", raising=False)
    path = PetabSuite.cache_path(PETAB_SUITE_COMMIT)
    assert path.parent.name == "petab-test-suite"
    assert path.name == PETAB_SUITE_COMMIT
    assert path.parent.parent == cache.cache_root()
    monkeypatch.setenv("SBMLSIM_PETAB_SUITE_PATH", "/somewhere")
    assert PetabSuite.cache_path(PETAB_SUITE_COMMIT) == Path("/somewhere")
