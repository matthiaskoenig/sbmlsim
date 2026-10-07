"""The cases of the PEtab v2 test suite on disk.

The [test suite](https://github.com/PEtab-dev/petab_test_suite) has a group of
cases per model format and the cases of the math of PEtab, below
`petabtests/cases/v2.0.0`:

| group | case | compared |
| --- | --- | --- |
| `sbml` | `PetabCase` | log-likelihood, chi2, simulations, log prior, log posterior |
| `math` | `MathCase` | the value of an expression of `math_tests.yaml` |

A case of a model is a directory named by its number with the problem
`_<id>.yaml`, the solution `_<id>_solution.yaml` and the simulations it
names. It is read with `PetabReader`, simulated at the nominal values of its
parameter table (`PetabReader.nominal_parameters`) and compared with the
tolerances of its solution. A math case compiles its expression with
`sbmlsim.simulator.formula.compile_formula`, which is what the formulas of
conditions, observables and noise are evaluated with.

The suite has no releases, so it is pinned by a commit. `PetabSuite` is the
directory of the groups with the download and the cache of the commit.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2.likelihood import (
    chi2,
    log_likelihood,
    log_prior,
    unnorm_log_posterior,
)
from sbmlsim.fit.petab_v2.reader import DEFAULT_EXPERIMENT, PetabReader
from sbmlsim.simulator.formula import compile_formula
from sbmlsim.testsuite import cache

logger = logging.getLogger(__name__)

#: commit of the test suite the tests run against
PETAB_SUITE_COMMIT = "4c17947afdced710bd1cbb4360bb36177ed07857"

#: archive of a commit of the suite
PETAB_SUITE_URL = "https://github.com/PEtab-dev/petab_test_suite/archive/{commit}.zip"

#: the environment variable which points at the cases when they are not in
#: the cache, i.e. at the directory `petabtests/cases/v2.0.0` of a checkout
PETAB_SUITE_PATH_VARIABLE = "SBMLSIM_PETAB_SUITE_PATH"

#: the version of PEtab whose cases are run, a directory of the suite
FORMAT_VERSION = "v2.0.0"

#: the groups of cases, which are the directories of the version
SBML = "sbml"
MATH = "math"

#: the file of the math cases in the directory `math`
MATH_FILE = "math_tests.yaml"

#: the settings a case is simulated with. The output are the times of the
#: measurements, the integrator is tighter than the tolerances of the cases
CASE_SETTINGS = FitSettings(
    parameter_scale=ParameterScaleType.LINEAR,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-12,
)

#: relative and absolute tolerance of a math case
MATH_TOLERANCE = 1e-12


class CaseStatus(StrEnum):
    """The outcome of a case."""

    PASS = "pass"
    TOLERANCE = "tolerance"
    ERROR = "error"


@dataclass(frozen=True)
class CaseResult:
    """The outcome of a case.

    Attributes:
        group: the group of the case, `sbml` or `math`.
        cid: the number of the case, e.g. `0001`.
        status: the outcome.
        message: what failed, empty for a case which passes.
        max_difference: the largest absolute difference to the reference
            values, `None` when nothing was compared.
    """

    group: str
    cid: str
    status: CaseStatus
    message: str = ""
    max_difference: float | None = None

    @property
    def passed(self) -> bool:
        """Check whether the case passes."""
        return self.status == CaseStatus.PASS

    @property
    def key(self) -> str:
        """Get the key of the case in the baseline, e.g. `sbml/0001`."""
        return f"{self.group}/{self.cid}"


@dataclass(frozen=True)
class _Difference:
    """A compared value: what it is, the difference and its tolerance."""

    name: str
    difference: float
    tolerance: float

    @property
    def failed(self) -> bool:
        """Check whether the difference is outside of the tolerance."""
        return not self.difference <= self.tolerance


def _difference(
    name: str, value: float, expected: float, tolerance: float
) -> _Difference:
    """Get the absolute difference of a value to its reference."""
    if value == expected:
        # equal infinities, e.g. a log prior of `-inf`
        return _Difference(name, 0.0, tolerance)
    return _Difference(name, abs(float(value) - float(expected)), tolerance)


@dataclass(frozen=True)
class PetabCase:
    """A case of a model of the PEtab v2 test suite.

    Attributes:
        path: the directory of the case.
        cid: the number of the case, the name of its directory.
        group: the group of the case, the format of its model.
        problem_file: the YAML file of the problem.
        solution: the content of the solution.
    """

    path: Path
    cid: str
    group: str
    problem_file: Path
    solution: dict[str, Any]

    @classmethod
    def from_directory(cls, path: Path, group: str = SBML) -> PetabCase:
        """Read a case from its directory.

        Args:
            path: the directory of the case.
            group: the group of the case.

        Returns:
            The case.

        Raises:
            ValueError: if the directory has no problem or no solution.
        """
        cid = path.name
        problem_file = path / f"_{cid}.yaml"
        solution_file = path / f"_{cid}_solution.yaml"
        for file in (problem_file, solution_file):
            if not file.is_file():
                raise ValueError(
                    f"'{path}' is not a case of the PEtab test suite, it has no "
                    f"'{file.name}'."
                )
        solution = yaml.safe_load(solution_file.read_text(encoding="utf-8"))
        return cls(
            path=path,
            cid=cid,
            group=group,
            problem_file=problem_file,
            solution=solution,
        )

    @property
    def key(self) -> str:
        """Get the key of the case in the baseline, e.g. `sbml/0001`."""
        return f"{self.group}/{self.cid}"

    def run(self) -> CaseResult:
        """Run the case.

        Returns:
            The outcome, `error` when the case cannot be read or simulated.
        """
        try:
            differences = self._compare()
        except Exception as err:  # every failure is the outcome of the case
            logger.debug("Case '%s' failed", self.key, exc_info=True)
            return CaseResult(
                group=self.group,
                cid=self.cid,
                status=CaseStatus.ERROR,
                message=f"{type(err).__name__}: {err}",
            )
        failed = [d for d in differences if d.failed]
        max_difference = max((d.difference for d in differences), default=None)
        if not failed:
            return CaseResult(
                group=self.group,
                cid=self.cid,
                status=CaseStatus.PASS,
                max_difference=max_difference,
            )
        return CaseResult(
            group=self.group,
            cid=self.cid,
            status=CaseStatus.TOLERANCE,
            message="; ".join(
                f"{d.name}: difference {d.difference:.3g} > {d.tolerance:.3g}"
                for d in failed
            ),
            max_difference=max_difference,
        )

    def _compare(self) -> list[_Difference]:
        """Simulate the case and compare it with its solution.

        Raises:
            ValueError: if a simulation of the solution has no measurement of
                the problem or the other way round.
        """
        solution = self.solution
        reader = PetabReader.from_yaml(self.problem_file)
        problem = reader.to_optimization_problem(opid=f"case_{self.cid}")
        problem.initialize(CASE_SETTINGS)
        nominal = reader.nominal_parameters(problem)
        # the problem is simulated once, every value is calculated from it
        evaluations = problem.evaluations(nominal.x(problem.pids), problem.indices())

        differences = [
            _difference(
                "llh",
                log_likelihood(problem, nominal, evaluations),
                solution["llh"],
                solution["tol_llh"],
            ),
            _difference(
                "chi2",
                chi2(problem, nominal, evaluations),
                solution["chi2"],
                solution["tol_chi2"],
            ),
        ]
        if "log_prior" in solution:
            priors = log_prior(problem, nominal)
            missing = sorted(set(solution["log_prior"]) - set(priors))
            if missing:
                raise ValueError(
                    f"The parameters {missing} of the log prior of the solution "
                    f"are not parameters of the fit."
                )
            differences.extend(
                _difference(
                    f"log_prior {pid}",
                    priors[pid],
                    expected,
                    solution["tol_log_prior"],
                )
                for pid, expected in solution["log_prior"].items()
            )
        if "unnorm_log_posterior" in solution:
            differences.append(
                _difference(
                    "unnorm_log_posterior",
                    unnorm_log_posterior(problem, nominal, evaluations),
                    solution["unnorm_log_posterior"],
                    solution["tol_unnorm_log_posterior"],
                )
            )

        expected = pd.concat(
            [
                pd.read_csv(self.path / file, sep="\t")
                for file in solution["simulation_files"]
            ],
            ignore_index=True,
        )
        experiments = (
            expected["experimentId"].fillna("").astype(str)
            if "experimentId" in expected
            else pd.Series([""] * len(expected))
        )
        compared = 0
        for k, evaluation in evaluations.items():
            prediction = evaluation.prediction
            observable_id = reader.observable_id(problem.mapping_keys[k])
            experiment_id = problem.simulation_keys[k]
            selected = (expected["observableId"] == observable_id) & (
                (experiments == experiment_id)
                | ((experiments == "") & (experiment_id == DEFAULT_EXPERIMENT))
            )
            rows = expected[selected]
            name = f"simulation {observable_id} in {experiment_id}"
            if len(rows) != len(prediction):
                raise ValueError(
                    f"The {name} has '{len(prediction)}' values, the solution "
                    f"'{len(rows)}'."
                )
            compared += len(rows)
            difference = np.abs(
                np.asarray(prediction, dtype=float)
                - rows["simulation"].to_numpy(dtype=float)
            )
            differences.append(
                _Difference(
                    name,
                    float(np.max(difference)) if difference.size else 0.0,
                    solution["tol_simulations"],
                )
            )
        if compared != len(expected):
            raise ValueError(
                f"The problem has '{compared}' of the '{len(expected)}' "
                f"simulations of the solution."
            )
        return differences


@dataclass(frozen=True)
class MathCase:
    """A case of the math of PEtab.

    Attributes:
        index: the position of the case in `math_tests.yaml`.
        expression: the math expression.
        expected: its value, a number or a symbolic expression.
    """

    index: int
    expression: str
    expected: float | str

    @property
    def cid(self) -> str:
        """Get the number of the case, its position with three digits."""
        return f"{self.index:03d}"

    def run(self) -> CaseResult:
        """Compile the expression and compare its value.

        A symbolic expression is evaluated with both at values of its
        symbols.

        Returns:
            The outcome, `error` when the expression is not compiled.
        """
        try:
            compiled = compile_formula(str(self.expression))
            if compiled.symbols:
                values = [1.3 + 0.7 * k for k in range(len(compiled.symbols))]
                by_symbol = dict(zip(compiled.symbols, values, strict=True))
                expected_formula = compile_formula(str(self.expected))
                expected = expected_formula.evaluate(
                    [by_symbol[symbol] for symbol in expected_formula.symbols]
                )
                value = compiled.evaluate(values)
            else:
                expected = float(self.expected)
                value = compiled.evaluate([])
        except Exception as err:  # every failure is the outcome of the case
            return CaseResult(
                group=MATH,
                cid=self.cid,
                status=CaseStatus.ERROR,
                message=f"'{self.expression}': {type(err).__name__}: {err}",
            )
        if value == expected or math.isclose(
            value, expected, rel_tol=MATH_TOLERANCE, abs_tol=MATH_TOLERANCE
        ):
            return CaseResult(
                group=MATH, cid=self.cid, status=CaseStatus.PASS, max_difference=0.0
            )
        return CaseResult(
            group=MATH,
            cid=self.cid,
            status=CaseStatus.TOLERANCE,
            message=f"'{self.expression}' is '{value}', expected '{expected}'",
            max_difference=abs(value - expected),
        )


def math_cases(path: Path) -> list[MathCase]:
    """Read the math cases of a `math_tests.yaml`.

    Args:
        path: the file.

    Returns:
        The cases in the order of the file.

    Raises:
        ValueError: if a case has no expression or no expected value.
    """
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases: list[MathCase] = []
    for index, case in enumerate(content["cases"]):
        for entry in ("expression", "expected"):
            if entry not in case:
                raise ValueError(
                    f"The math case '{index}' of '{path}' has no '{entry}'."
                )
        cases.append(
            MathCase(
                index=index,
                expression=str(case["expression"]),
                expected=case["expected"],
            )
        )
    return cases


@dataclass(frozen=True)
class PetabSuite:
    """The cases of a commit of the PEtab v2 test suite.

    Attributes:
        path: the directory which holds the directories of the groups, i.e.
            `petabtests/cases/v2.0.0` of the suite.
        commit: the commit of the suite.
    """

    path: Path
    commit: str

    @staticmethod
    def cache_path(commit: str = PETAB_SUITE_COMMIT) -> Path:
        """Get the directory a commit of the suite is unpacked into.

        `SBMLSIM_PETAB_SUITE_PATH` overrides it. Otherwise it is
        `sbmlsim/petab-test-suite/<commit>` in the user cache.

        Args:
            commit: the commit of the suite.

        Returns:
            The directory the groups of the commit live in.
        """
        return cache.cache_path(PETAB_SUITE_PATH_VARIABLE, "petab-test-suite", commit)

    @classmethod
    def cached(cls, commit: str = PETAB_SUITE_COMMIT) -> PetabSuite | None:
        """Get a commit of the suite if it is already on this machine.

        Args:
            commit: the commit of the suite.

        Returns:
            The suite, or `None` if it was not downloaded yet.
        """
        path = cls.cache_path(commit)
        return cls(path=path, commit=commit) if path.is_dir() else None

    @classmethod
    def load(cls, commit: str = PETAB_SUITE_COMMIT) -> PetabSuite:
        """Get a commit of the suite, downloading it if it is not cached.

        Args:
            commit: the commit of the suite.

        Returns:
            The suite with its cases unpacked in the cache.

        Raises:
            OSError: if the commit cannot be downloaded.
        """
        suite = cls.cached(commit)
        if suite is not None:
            if not cache.is_overridden(PETAB_SUITE_PATH_VARIABLE):
                cache.remove_stale(suite.path)
            return suite
        path = cls.cache_path(commit)
        url = PETAB_SUITE_URL.format(commit=commit)
        logger.info("Downloading the PEtab test suite '%s'", commit)
        cache.fetch(url, path, select=cls._cases_dir)
        return cls(path=path, commit=commit)

    @staticmethod
    def _cases_dir(staging: Path) -> Path:
        """Get the directory of the groups of an unpacked archive.

        Args:
            staging: directory the archive was unpacked into.

        Returns:
            The directory `cases/v2.0.0`.

        Raises:
            OSError: if the archive holds no such directory.
        """
        for candidate in sorted(staging.glob(f"**/cases/{FORMAT_VERSION}")):
            if candidate.is_dir():
                return candidate
        raise OSError(
            f"No directory 'cases/{FORMAT_VERSION}' in the unpacked suite '{staging}'"
        )

    def case_ids(self, group: str = SBML) -> list[str]:
        """Get the numbers of the cases of a group of models.

        Args:
            group: the group, i.e. the name of its directory.

        Returns:
            The names of the case directories, sorted, empty for a group the
            suite does not have.
        """
        directory = self.path / group
        if not directory.is_dir():
            return []
        return sorted(
            p.name for p in directory.iterdir() if p.is_dir() and p.name.isdigit()
        )

    def cases(self, group: str = SBML) -> Iterator[PetabCase]:
        """Iterate the cases of a group of models."""
        for cid in self.case_ids(group):
            yield PetabCase.from_directory(self.path / group / cid, group=group)

    def math_cases(self) -> list[MathCase]:
        """Get the math cases, empty for a suite without them."""
        path = self.path / MATH / MATH_FILE
        return math_cases(path) if path.is_file() else []

    def run(self) -> list[CaseResult]:
        """Run the cases of the suite.

        Returns:
            The results of the group `sbml` and of the math cases, in this
            order.
        """
        results = [case.run() for case in self.cases(SBML)]
        results.extend(case.run() for case in self.math_cases())
        return results
