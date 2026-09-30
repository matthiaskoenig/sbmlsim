"""The cases of the PEtab SciML test suite.

Every case is a test, so a failure names the case. They are marked
`sciml_testsuite` and deselected by default, because they need the download
of the suite: `pytest -m sciml_testsuite` runs them and `tox r -e sciml`
downloads the cases first. A case is compared with
`tests/data/sciml_baseline.json`, which records the cases that do not pass,
each with its reason.

The baseline fails in both directions. A case which passed and now fails is a
regression. A case which is in the baseline and passes is a baseline which is
out of date.
"""

import json
from pathlib import Path

import pytest
from petab import v2 as petab_v2

from sbmlsim.sciml.testsuite import (
    INITIALIZATION,
    MODEL_IMPORT,
    PROBLEM_IMPORT,
    SCIML_SUITE_COMMIT,
    CaseResult,
    InitializationCase,
    ModelImportCase,
    ProblemImportCase,
    SciMLSuite,
)

pytestmark = pytest.mark.sciml_testsuite

#: the expected outcome of the cases which do not pass
BASELINE_PATH = Path(__file__).parent.parent / "data" / "sciml_baseline.json"

SKIP_REASON = (
    f"the PEtab SciML test suite '{SCIML_SUITE_COMMIT}' is not on this machine, "
    f"run `uv run python scripts/sciml_testsuite.py download` to fetch it"
)


def _case_ids(group: str) -> list[str]:
    """Get the cases of a group of the cached suite, for the parametrization."""
    suite = SciMLSuite.cached()
    return [] if suite is None else suite.case_ids(group)


MODEL_IMPORT_IDS = _case_ids(MODEL_IMPORT)
INITIALIZATION_IDS = _case_ids(INITIALIZATION)
PROBLEM_IMPORT_IDS = _case_ids(PROBLEM_IMPORT)


@pytest.fixture(scope="session")
def baseline() -> dict:
    """Get the expected outcome of the cases."""
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def suite() -> SciMLSuite:
    """Get the cached suite, skipping the tests when it was not downloaded."""
    cached = SciMLSuite.cached()
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
            f"baseline: uv run python scripts/sciml_testsuite.py baseline"
        )
        assert result.status.value == expected["status"], (
            f"'{result.key}' fails differently than recorded: "
            f"'{expected['status']}' -> '{result.status.value}' ({result.message})"
        )


@pytest.mark.parametrize("cid", MODEL_IMPORT_IDS)
def test_model_import(cid: str, suite: SciMLSuite, baseline: dict) -> None:
    """The forward pass of the case has the outcome the baseline records."""
    case = ModelImportCase.from_directory(suite.path / MODEL_IMPORT / cid)
    _check(case.run(), baseline)


@pytest.mark.parametrize("cid", INITIALIZATION_IDS)
def test_initialization(cid: str, suite: SciMLSuite, baseline: dict) -> None:
    """The nominal values of the case have the outcome the baseline records."""
    case = InitializationCase.from_directory(suite.path / INITIALIZATION / cid)
    _check(case.run(), baseline)


@pytest.mark.parametrize("cid", PROBLEM_IMPORT_IDS)
def test_problem_import(cid: str, suite: SciMLSuite, baseline: dict) -> None:
    """The log-likelihood, the simulations and the gradient of the case agree."""
    case = ProblemImportCase.from_directory(suite.path / PROBLEM_IMPORT / cid)
    _check(case.run(), baseline)


@pytest.mark.parametrize("cid", PROBLEM_IMPORT_IDS)
def test_problem_round_trip(cid: str, suite: SciMLSuite, tmp_path: Path) -> None:
    """A case which is read is written as PEtab SciML and read back exactly."""
    case = ProblemImportCase.from_directory(suite.path / PROBLEM_IMPORT / cid)
    if case.llh is None:
        pytest.skip("the case states a log-posterior and is not read (sciml-priors)")
    assert case.round_trip(tmp_path) == []
    # the export is valid PEtab SciML, `petab` reads its networks with torch
    pytest.importorskip("torch")
    issues = petab_v2.Problem.from_yaml(tmp_path / "petab" / "problem.yaml").validate()
    assert not issues.has_errors(), str(issues)


def test_the_baseline_matches_the_suite(suite: SciMLSuite, baseline: dict) -> None:
    """The baseline was recorded for the commit which is pinned and cached."""
    assert baseline["suite_commit"] == SCIML_SUITE_COMMIT == suite.commit
    assert [f"{i:03d}" for i in range(1, 55)] == MODEL_IMPORT_IDS
    assert INITIALIZATION_IDS == ["001", "002", "003"]
    assert [f"{i:03d}" for i in range(1, 40)] == PROBLEM_IMPORT_IDS
    assert baseline["n_cases"] == (
        len(MODEL_IMPORT_IDS) + len(INITIALIZATION_IDS) + len(PROBLEM_IMPORT_IDS)
    )
    assert baseline["n_passed"] == baseline["n_cases"] - len(
        baseline["expected_failures"]
    )
    keys = {f"{MODEL_IMPORT}/{cid}" for cid in MODEL_IMPORT_IDS}
    keys |= {f"{INITIALIZATION}/{cid}" for cid in INITIALIZATION_IDS}
    keys |= {f"{PROBLEM_IMPORT}/{cid}" for cid in PROBLEM_IMPORT_IDS}
    assert set(baseline["expected_failures"]) <= keys


def test_every_failure_has_a_reason(baseline: dict) -> None:
    """A case is not listed without saying why it does not pass."""
    for key, expected in baseline["expected_failures"].items():
        assert expected["status"] != "pass", key
        assert len(expected["reason"]) > 20, key
