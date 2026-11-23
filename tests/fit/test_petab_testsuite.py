"""The cases of the PEtab v2 test suite.

Every case is a test, so a failure names the case. They are marked
`petab_testsuite` and deselected by default, because they need the download of
the suite: `pytest -m petab_testsuite` runs them and `tox r -e petab`
downloads the cases first. A case is compared with
`tests/data/petab_baseline.json`, which records the cases that do not pass,
each with its reason.

The baseline fails in both directions. A case which passed and now fails is a
regression. A case which is in the baseline and passes is a baseline which is
out of date.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from sbmlsim.engine import Engine
from sbmlsim.fit import fit_core
from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ObjectiveType
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.fit.petab_v2.testsuite import (
    CASE_SETTINGS,
    MATH,
    PETAB_SUITE_COMMIT,
    SBML,
    V1_SBML,
    CaseResult,
    PetabSuite,
)
from tests.engines import require_core_fit

pytestmark = pytest.mark.petab_testsuite

#: the expected outcome of the cases which do not pass
BASELINE_PATH = Path(__file__).parent.parent / "data" / "petab_baseline.json"

SKIP_REASON = (
    f"the PEtab test suite '{PETAB_SUITE_COMMIT}' is not on this machine, run "
    f"`uv run python scripts/petab_testsuite.py download` to fetch it"
)


def _case_ids() -> list[tuple[str, str]]:
    """Get the groups and cases of the models of the cached suite.

    The cases of PEtab v1 are the ones converted to PEtab v2, which a
    download converts.
    """
    suite = PetabSuite.cached()
    if suite is None:
        return []
    return [(group, cid) for group in (SBML, V1_SBML) for cid in suite.case_ids(group)]


def _engine_ids() -> list[str]:
    """Get the cases of the group `sbml`, which the fit of the engine runs on."""
    suite = PetabSuite.cached()
    return [] if suite is None else list(suite.case_ids(SBML))


def _engine_problem(suite: PetabSuite, cid: str) -> OptimizationProblem:
    """Get the problem of a case, initialized on sbmlsim-core."""
    reader = PetabReader.from_yaml(suite.case(cid, SBML).problem_file)
    problem = reader.to_optimization_problem(opid=f"case_{cid}")
    problem.engine = Engine.CORE
    problem.initialize(CASE_SETTINGS)
    return problem


def _math_ids() -> list[int]:
    """Get the positions of the math cases of the cached suite."""
    suite = PetabSuite.cached()
    return [] if suite is None else [case.index for case in suite.math_cases()]


@pytest.fixture(scope="session")
def baseline() -> dict:
    """Get the expected outcome of the cases."""
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def suite() -> PetabSuite:
    """Get the cached suite, skipping the tests when it was not downloaded."""
    cached = PetabSuite.cached()
    if cached is None:
        pytest.skip(SKIP_REASON)
    return cached


def _check(result: CaseResult, baseline: dict) -> None:
    """Compare the outcome of a case with the baseline."""
    expected = baseline["expected_failures"].get(result.key)
    if expected is None:
        assert result.passed, (
            f"'{result.key}' is a regression, it passed before: "
            f"{result.status.value} {result.message}"
        )
    else:
        assert not result.passed, (
            f"'{result.key}' passes now, it is expected to fail with "
            f"'{expected['status']}' ({expected['reason']}). Refresh the "
            f"baseline: uv run python scripts/petab_testsuite.py baseline"
        )
        assert result.status.value == expected["status"], (
            f"'{result.key}' fails differently than recorded: "
            f"'{expected['status']}' -> '{result.status.value}' ({result.message})"
        )


@pytest.mark.parametrize(
    ("group", "cid"), _case_ids(), ids=[f"{g}/{c}" for g, c in _case_ids()]
)
def test_case(group: str, cid: str, suite: PetabSuite, baseline: dict) -> None:
    """A case of a model has the outcome the baseline records."""
    _check(suite.run_case(cid, group), baseline)


@pytest.mark.parametrize("index", _math_ids())
def test_math_case(index: int, suite: PetabSuite, baseline: dict) -> None:
    """A math case has the outcome the baseline records."""
    _check(suite.math_cases()[index].run(), baseline)


def test_the_baseline_matches_the_suite(suite: PetabSuite, baseline: dict) -> None:
    """The baseline was recorded for the commit which is pinned and cached."""
    assert baseline["suite_commit"] == PETAB_SUITE_COMMIT == suite.commit
    assert baseline["n_cases"] == len(_case_ids()) + len(_math_ids())
    assert all(
        key.rsplit("/", 1)[0] in {SBML, V1_SBML, MATH}
        for key in baseline["expected_failures"]
    )
    for key, expected in baseline["expected_failures"].items():
        assert expected["reason"] != "reason missing", f"'{key}' has no reason"


# the cases as fits of sbmlsim-core: a case which has no gap is fitted by the
# engine itself, and its residuals are the residuals of the problem


#: the cases of the group `sbml` which are no fit of sbmlsim-core, each with
#: the code of its gap, see `sbmlsim.fit.fit_core.gaps_of`. 0025 has parameters
#: which have a prior and no bounds. The other four set a concentration at a
#: change by a formula which reads one, `C + [S]` (0016, 0031) and `[A] + 5`
#: (0028, 0030): such a formula is no linear form over the entities of the
#: model, since the parser of the math of PEtab does not read the brackets of
#: a concentration, so the request can carry it neither as a form nor as a
#: number the engine would hold where the fit started, and the fit of those
#: cases runs in python (#617). The gaps which are gone: the values before the
#: initialization which read a parameter the fit estimates are the `derived`
#: of a condition (sbmlsim-core 0.17.0), the state of the model before the
#: events of the time of a change is the `before_events` of a request and the
#: steady state after the end of a simulation of a model with events its
#: `steady_state_after` (both sbmlsim-core 0.18.0), and a change of a linear
#: formula over the entities of the model is the `forms` of a change
#: (sbmlsim-core 0.20.0).
ENGINE_GAPS: dict[str, str] = {
    "0016": "change_value",
    "0025": "parameter",
    "0028": "change_value",
    "0030": "change_value",
    "0031": "change_value",
}

#: the cases whose likelihood the engine does not fit, by the gap which keeps
#: them out, see `_gaps_of_the_objective` (#613): a noise formula which reads
#: the observable (0015, 0021, 0024) and the parameter of 0025, whose bounds
#: are no bounds of an optimizer of the engine, and the change of a formula of
#: a concentration of the four cases of `ENGINE_GAPS`, which keeps a fit of
#: either objective out
ENGINE_LIKELIHOOD_GAPS: dict[str, str] = {
    "0015": "objective",
    "0016": "change_value",
    "0021": "objective",
    "0024": "objective",
    "0025": "objective",
    "0028": "change_value",
    "0030": "change_value",
    "0031": "change_value",
}


@pytest.mark.parametrize("cid", _engine_ids(), ids=_engine_ids())
def test_the_engine_fits_the_case_as_the_problem_computes_it(
    cid: str, suite: PetabSuite
) -> None:
    """A case is a fit of the engine, or has the gap which keeps it out.

    The residuals the engine computes from the request are the ones the
    problem computes at the nominal parameters, point for point, which is what
    makes the fit of the engine the fit of the problem.
    """
    require_core_fit()
    problem = _engine_problem(suite, cid)
    gaps = fit_core.gaps_of(problem)
    expected = ENGINE_GAPS.get(cid)

    if expected is not None:
        assert gaps, f"'{cid}' is a fit of the engine now, remove it from ENGINE_GAPS"
        assert {gap.code for gap in gaps} == {expected}, [gap.message for gap in gaps]
        return

    assert gaps == [], [gap.message for gap in gaps]
    core = fit_core.loaded_model(problem).fit_problem(
        fit_core.fit_request(problem), threads=1
    )
    x = np.asarray(problem.to_scale(problem.xmodel), dtype=float)
    reference = np.asarray(problem.residuals(x), dtype=float)

    residuals = np.asarray(core.residuals(x), dtype=float)

    assert residuals.shape == reference.shape
    np.testing.assert_array_equal(residuals, reference)


@pytest.mark.parametrize("cid", _engine_ids(), ids=_engine_ids())
def test_the_engine_fits_the_likelihood_of_the_case_as_the_problem_does(
    cid: str, suite: PetabSuite
) -> None:
    """A case is a likelihood fit of the engine, or has the gap which keeps it out.

    PEtab defines the objective of a problem as its likelihood, which is what
    a problem read without the `sbmlsim` extension is fitted with, so this is
    the objective of the cases as PEtab means them (#613). The residuals of
    the engine are the residuals of the data the problem computes, point for
    point: the problem adds the residuals of the scales of the noise, which
    are constant for a noise the fit does not move and which the cost of a
    run of the engine carries as `likelihood_constant`.
    """
    require_core_fit()
    reader = PetabReader.from_yaml(suite.case(cid, SBML).problem_file)
    problem = reader.to_optimization_problem(opid=f"case_{cid}")
    problem.engine = Engine.CORE
    problem.initialize(
        dataclasses.replace(CASE_SETTINGS, objective=ObjectiveType.LIKELIHOOD)
    )
    gaps = fit_core.gaps_of(problem)
    expected = ENGINE_LIKELIHOOD_GAPS.get(cid)

    if expected is not None:
        assert gaps, (
            f"'{cid}' is a likelihood fit of the engine now, remove it from "
            f"ENGINE_LIKELIHOOD_GAPS"
        )
        assert {gap.code for gap in gaps} == {expected}, [gap.message for gap in gaps]
        return

    assert gaps == [], [gap.message for gap in gaps]
    core = fit_core.loaded_model(problem).fit_problem(
        fit_core.fit_request(problem), threads=1
    )
    x = np.asarray(problem.to_scale(problem.xmodel), dtype=float)
    # a likelihood gives every mapping the residuals of its data and then the
    # ones of the scale of its noise; the engine computes the first of each
    whole = np.asarray(problem.residuals(x), dtype=float)
    reference_parts: list[np.ndarray] = []
    at = 0
    for k in problem.indices(MappingKind.TRAINING):
        size = len(problem.y_references[k])
        reference_parts.append(whole[at : at + size])
        at += 2 * size
    assert at == whole.size
    reference = np.concatenate(reference_parts)

    residuals = np.asarray(core.residuals(x), dtype=float)

    assert residuals.shape == reference.shape
    np.testing.assert_array_equal(residuals, reference)
