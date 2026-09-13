"""Fixtures of the SBML Test Suite tests."""

import json
from pathlib import Path

import pytest

from sbmlsim.testsuite import SemanticSuite
from sbmlsim.testsuite.cases import SUITE_VERSION

#: the expected outcome of every case, see `scripts/testsuite.py baseline`
BASELINE_PATH = Path(__file__).parent.parent / "data" / "testsuite_baseline.json"

#: what a run of the suite is compared with when it is not cached
SKIP_REASON = (
    f"the SBML Test Suite '{SUITE_VERSION}' is not on this machine, "
    f"run `uv run python scripts/testsuite.py download` to fetch it"
)


@pytest.fixture(scope="session")
def baseline() -> dict:
    """Get the expected outcome of the cases."""
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def suite() -> SemanticSuite:
    """Get the cached suite, skipping the tests when it was not downloaded.

    The suite is not downloaded by the tests: a test run must not depend on
    the network, and 13 MB is not something a test fetches behind the back of
    whoever started it.
    """
    cached = SemanticSuite.cached()
    if cached is None:
        pytest.skip(SKIP_REASON)
    return cached
