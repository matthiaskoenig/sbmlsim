"""Tests of the command line of the PEtab v2 test suite."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from sbmlsim.fit.petab_v2.testsuite import CaseResult, CaseStatus

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "petab_testsuite.py"


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the script, which is not a module of a package.

    The script configures the logging of the package, which stays for the
    tests which run after it, so it is left out.
    """
    spec = importlib.util.spec_from_file_location("petab_testsuite", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.log, "enable_rich_logging", lambda: None)
    return module


class FakeSuite:
    """A suite with given results, which needs no download."""

    commit = "abc"
    path = Path("/suite")

    def __init__(self, results: list[CaseResult]) -> None:
        self.results = results

    def run(self) -> list[CaseResult]:
        return self.results


RESULTS = [
    CaseResult("sbml", "0001", CaseStatus.PASS),
    CaseResult("math", "002", CaseStatus.TOLERANCE, "'1' is '2'", 1.0),
]

REASON = "the reason why the case does not pass"


@pytest.mark.parametrize(
    ("failures", "code"),
    [
        ({"math/002": {"status": "tolerance", "reason": REASON}}, 0),
        ({}, 1),
        ({"math/002": {"status": "error", "reason": REASON}}, 1),
    ],
)
def test_run_fails_when_a_case_differs_from_the_baseline(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failures: dict,
    code: int,
) -> None:
    """`run` returns 1 when a case does not have the outcome of the baseline."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"suite_commit": "abc", "expected_failures": failures}))
    monkeypatch.setattr(script.PetabSuite, "load", lambda commit: FakeSuite(RESULTS))
    monkeypatch.setattr(script, "BASELINE_PATH", path)
    assert script.main(["run"]) == code


def test_the_baseline_is_written(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`baseline` records the cases which do not pass, without a reason yet."""
    monkeypatch.setattr(script.PetabSuite, "load", lambda commit: FakeSuite(RESULTS))
    monkeypatch.setattr(script, "BASELINE_PATH", tmp_path / "baseline.json")
    assert script.main(["baseline"]) == 0
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline == {
        "suite_commit": "abc",
        "n_cases": 2,
        "n_passed": 1,
        "expected_failures": {
            "math/002": {"status": "tolerance", "reason": script.MISSING_REASON}
        },
    }
