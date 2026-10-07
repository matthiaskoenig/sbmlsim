"""The problems of the PEtab benchmark collection.

Every problem is a test, so a failure names the problem. They are marked
`petab_benchmark` and deselected by default, because they need the download
and the conversion of the collection and some problems take minutes:
`pytest -m petab_benchmark` runs them and `tox r -e benchmark` downloads the
collection first. A problem is compared with
`tests/data/benchmark_baseline.json`, which records the problems that do not
pass, each with its reason.

The baseline fails in both directions. A problem which passed and now fails is
a regression. A problem which is in the baseline and passes is a baseline
which is out of date.
"""

import json
from pathlib import Path

import pytest

from sbmlsim.fit.petab_v2.benchmark import (
    BENCHMARK_COMMIT,
    BENCHMARK_TIMEOUT,
    BenchmarkCollection,
)

pytestmark = pytest.mark.petab_benchmark

#: the expected outcome of the problems which do not pass
BASELINE_PATH = Path(__file__).parent.parent / "data" / "benchmark_baseline.json"

SKIP_REASON = (
    f"the PEtab benchmark collection '{BENCHMARK_COMMIT}' is not on this machine, "
    f"run `uv run python scripts/petab_benchmark.py download` to fetch it"
)


def _problem_names() -> list[str]:
    """Get the problems of the cached collection, for the parametrization."""
    collection = BenchmarkCollection.cached()
    return [] if collection is None else collection.problem_names()


@pytest.fixture(scope="session")
def baseline() -> dict:
    """Get the expected outcome of the problems."""
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def collection() -> BenchmarkCollection:
    """Get the cached collection, skipping the tests when it was not downloaded."""
    cached = BenchmarkCollection.cached()
    if cached is None:
        pytest.skip(SKIP_REASON)
    return cached


@pytest.mark.parametrize("name", _problem_names())
def test_problem(name: str, collection: BenchmarkCollection, baseline: dict) -> None:
    """A problem has the outcome the baseline records."""
    result = collection.problem(name).run(
        llh_reference=collection.references().get(name), timeout=BENCHMARK_TIMEOUT
    )
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
            f"baseline: uv run python scripts/petab_benchmark.py baseline"
        )
        assert result.status.value == expected["status"], (
            f"'{result.key}' fails differently than recorded: "
            f"'{expected['status']}' -> '{result.status.value}' ({result.message})"
        )


def test_the_baseline_matches_the_collection(
    collection: BenchmarkCollection, baseline: dict
) -> None:
    """The baseline was recorded for the commit which is pinned and cached."""
    assert baseline["suite_commit"] == BENCHMARK_COMMIT == collection.commit
    assert baseline["n_cases"] == len(_problem_names())
    for key, expected in baseline["expected_failures"].items():
        assert expected["reason"] != "reason missing", f"'{key}' has no reason"
