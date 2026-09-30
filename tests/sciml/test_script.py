"""Tests of the command line of the PEtab SciML test suite."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from sbmlsim.sciml.testsuite import CaseResult, CaseStatus

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "sciml_testsuite.py"


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the script, which is not a module of a package.

    The script configures the logging of the package, which stays for the
    tests which run after it, so it is left out.
    """
    spec = importlib.util.spec_from_file_location("sciml_testsuite", SCRIPT)
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
    CaseResult("ml_model_import", "001", CaseStatus.PASS),
    CaseResult("ml_model_import", "002", CaseStatus.TOLERANCE, "too far", 0.1),
]


def _baseline(path: Path, failures: dict[str, dict[str, str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "suite_commit": "abc",
                "n_cases": 2,
                "n_passed": 2 - len(failures),
                "expected_failures": failures,
            }
        )
    )
    return path


REASON = "the reason why the case does not pass, long enough"


@pytest.mark.parametrize(
    ("failures", "code"),
    [
        ({"ml_model_import/002": {"status": "tolerance", "reason": REASON}}, 0),
        ({}, 1),
        ({"ml_model_import/002": {"status": "shape", "reason": REASON}}, 1),
        (
            {
                "ml_model_import/001": {"status": "error", "reason": REASON},
                "ml_model_import/002": {"status": "tolerance", "reason": REASON},
            },
            1,
        ),
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
    monkeypatch.setattr(script.SciMLSuite, "load", lambda commit: FakeSuite(RESULTS))
    monkeypatch.setattr(
        script, "BASELINE_PATH", _baseline(tmp_path / "baseline.json", failures)
    )
    assert script.main(["run"]) == code


def test_the_baseline_keeps_a_reason_of_the_same_status(
    script: ModuleType, tmp_path: Path
) -> None:
    """A recorded reason is kept while the status of the case is the same."""
    path = _baseline(
        tmp_path / "baseline.json",
        {"ml_model_import/002": {"status": "tolerance", "reason": REASON}},
    )
    script.write_baseline(RESULTS, FakeSuite(RESULTS), path)
    baseline = json.loads(path.read_text())
    assert baseline["expected_failures"] == {
        "ml_model_import/002": {"status": "tolerance", "reason": REASON}
    }
    assert baseline["n_passed"] == 1


def test_the_baseline_drops_the_reason_of_a_changed_status(
    script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reason of another status is stale, it is dropped and reported."""
    path = _baseline(
        tmp_path / "baseline.json",
        {"ml_model_import/002": {"status": "shape", "reason": REASON}},
    )
    script.write_baseline(RESULTS, FakeSuite(RESULTS), path)
    baseline = json.loads(path.read_text())
    assert baseline["expected_failures"]["ml_model_import/002"] == {
        "status": "tolerance",
        "reason": script.MISSING_REASON,
    }
    output = capsys.readouterr().out
    assert "ml_model_import/002" in output
    assert "'shape' -> 'tolerance'" in output
