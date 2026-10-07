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

import json
from pathlib import Path

import pytest

from sbmlsim.fit.petab_v2.testsuite import (
    MATH,
    PETAB_SUITE_COMMIT,
    SBML,
    CaseResult,
    PetabCase,
    PetabSuite,
)

pytestmark = pytest.mark.petab_testsuite

#: the expected outcome of the cases which do not pass
BASELINE_PATH = Path(__file__).parent.parent / "data" / "petab_baseline.json"

SKIP_REASON = (
    f"the PEtab test suite '{PETAB_SUITE_COMMIT}' is not on this machine, run "
    f"`uv run python scripts/petab_testsuite.py download` to fetch it"
)


def _case_ids() -> list[str]:
    """Get the cases of the models of the cached suite, for the parametrization."""
    suite = PetabSuite.cached()
    return [] if suite is None else suite.case_ids(SBML)


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


@pytest.mark.parametrize("cid", _case_ids())
def test_case(cid: str, suite: PetabSuite, baseline: dict) -> None:
    """A case of a model has the outcome the baseline records."""
    case = PetabCase.from_directory(suite.path / SBML / cid)
    _check(case.run(), baseline)


@pytest.mark.parametrize("index", _math_ids())
def test_math_case(index: int, suite: PetabSuite, baseline: dict) -> None:
    """A math case has the outcome the baseline records."""
    _check(suite.math_cases()[index].run(), baseline)


def test_the_baseline_matches_the_suite(suite: PetabSuite, baseline: dict) -> None:
    """The baseline was recorded for the commit which is pinned and cached."""
    assert baseline["suite_commit"] == PETAB_SUITE_COMMIT == suite.commit
    assert baseline["n_cases"] == len(_case_ids()) + len(_math_ids())
    assert all(
        key.split("/")[0] in {SBML, MATH} for key in baseline["expected_failures"]
    )
    for key, expected in baseline["expected_failures"].items():
        assert expected["reason"] != "reason missing", f"'{key}' has no reason"
