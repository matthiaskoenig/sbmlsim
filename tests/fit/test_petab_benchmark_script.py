"""Tests of the command line of the PEtab benchmark collection."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from sbmlsim.fit.petab_v2.benchmark import BenchmarkResult, BenchmarkStatus

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "petab_benchmark.py"


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the script, which is not a module of a package."""
    spec = importlib.util.spec_from_file_location("petab_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.log, "enable_rich_logging", lambda: None)
    return module


RESULTS = [
    BenchmarkResult(
        "A_First",
        BenchmarkStatus.PASS,
        n_simulations=3,
        max_difference=1e-6,
        llh=-1.5,
        timings={"read": 0.1, "initialize": 0.2, "simulate": 0.01, "llh": 0.02},
    ),
    BenchmarkResult("B_Second", BenchmarkStatus.ERROR, "ValueError: a | b"),
]


def test_the_report_of_a_run(script: ModuleType, tmp_path: Path) -> None:
    """The report is a markdown table with a row per problem."""
    script.write_results(RESULTS, tmp_path)
    path = script.report(tmp_path)
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("| A_First | pass | 3 | 1.00e-06 | -1.5 |")
    assert "ValueError: a / b" in lines[3]


@pytest.mark.parametrize(
    ("failures", "code"),
    [
        ({"B_Second": {"status": "error", "reason": "the reason"}}, 0),
        ({}, 1),
    ],
)
def test_run_fails_when_a_problem_differs_from_the_baseline(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failures: dict,
    code: int,
) -> None:
    """`run` returns 1 when a problem does not have the outcome of the baseline."""
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"suite_commit": "abc", "expected_failures": failures}))
    collection = SimpleNamespace(commit="abc", path=tmp_path)
    monkeypatch.setattr(script.BenchmarkCollection, "load", lambda commit: collection)
    monkeypatch.setattr(script, "run", lambda c, names, processes: RESULTS)
    monkeypatch.setattr(script, "BASELINE_PATH", path)
    assert script.main(["run", "--output", str(tmp_path / "out")]) == code
    assert (tmp_path / "out" / "benchmark.json").is_file()
