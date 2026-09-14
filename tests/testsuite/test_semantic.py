"""The semantic cases of the SBML Test Suite.

Every case is a test, so a failure names the case and the tests distribute
over the workers of pytest-xdist. They are marked `testsuite` and deselected
by default: they take longer than the rest of the tests together and they
answer what libroadrunner supports, which a change to `sbmlsim` rarely moves.
`pytest -m testsuite` runs them and the release workflow does. A case is
compared with
`tests/data/testsuite_baseline.json`, which records the cases that do not
pass: the suite is not fully green, libroadrunner does not support algebraic
rules, delay equations or fast reactions, and a test which demanded a green
suite could only be skipped.

The baseline fails in both directions. A case which passed and now fails is a
regression. A case which is in the baseline and passes is a baseline which is
out of date, and the run says so instead of hiding the improvement.
"""

from pathlib import Path

import pytest

from sbmlsim.testsuite import SemanticSuite, run_case
from sbmlsim.testsuite.cases import SUITE_VERSION, SemanticCase

#: the cases are the conformance of libroadrunner and not of a change, so they
#: are not part of a normal run: `pytest -m testsuite` runs them, and so does
#: the release, see `.github/workflows/ci-cd.yml`
pytestmark = pytest.mark.testsuite


def _case_ids() -> list[str]:
    """Get the case directories of the cached suite, for the parametrization.

    Only the directory names are read: parsing every case at collection time
    would be paid once per worker of pytest-xdist. A directory which is not a
    timecourse case is skipped by the test itself.

    Returns:
        The case identifiers, empty when the suite is not on this machine.
    """
    path = SemanticSuite.cache_path(SUITE_VERSION)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.is_dir() and p.name.isdigit())


CASE_IDS = _case_ids()


@pytest.mark.parametrize("cid", CASE_IDS)
def test_case(cid: str, suite: SemanticSuite, baseline: dict) -> None:
    """The case has the outcome the baseline records."""
    case = SemanticCase.from_directory(Path(suite.path) / cid)
    if case is None:
        pytest.skip(f"'{cid}' is not a timecourse case")

    result = run_case(case)
    expected = baseline["expected_failures"].get(cid)

    if expected is None:
        assert result.passed, (
            f"'{cid}' is a regression, it passed before: {result.status.value} "
            f"{result.message}"
        )
    else:
        assert not result.passed, (
            f"'{cid}' passes now, it is expected to fail with '{expected}'. "
            f"Refresh the baseline: uv run python scripts/testsuite.py baseline"
        )
        assert result.status.value == expected, (
            f"'{cid}' fails differently than recorded: '{expected}' -> "
            f"'{result.status.value}' ({result.message})"
        )


def test_the_baseline_matches_the_suite(suite: SemanticSuite, baseline: dict) -> None:
    """The baseline was recorded for the release which is pinned and cached."""
    assert baseline["suite_version"] == SUITE_VERSION == suite.version
    assert baseline["n_cases"] == len(CASE_IDS) - _n_not_timecourse(suite)
    assert baseline["n_passed"] == baseline["n_cases"] - len(
        baseline["expected_failures"]
    )


def _n_not_timecourse(suite: SemanticSuite) -> int:
    """Count the case directories which are not a timecourse case."""
    return len(CASE_IDS) - len(list(suite.cases()))
