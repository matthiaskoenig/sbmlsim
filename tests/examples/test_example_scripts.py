"""Run the example scripts.

Every example is run as a module in a temporary working directory, so the
files it writes do not end up in the repository. An example which breaks
fails the test suite.

The examples which need optional tools (the AMICI and COPASI scripts of
`examples.comparison`) are not run here, see `examples/README.md`.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

#: the root of the repository, `python -m examples.<module>` is run from here
REPO_DIR = Path(__file__).parent.parent.parent

#: the arguments an example runs with in the tests: the samples and the fits
#: of the examples are sized to show what they do, a test only needs them to run
ARGUMENTS: dict[str, list[str]] = {
    "examples.petab.benchmark": ["--no-identifiability", "--runs=2"],
    "examples.sciml.lotka_volterra_fit": ["--max-nfev=4"],
    "examples.sensitivity.sensitivity_example": ["--quick", "--cores=1"],
}

#: examples which run offline and without optional dependencies
SCRIPTS = [
    "examples.timecourse",
    "examples.scan",
    "examples.observables",
    "examples.experiment_scans",
    "examples.fit_sampling",
    "examples.units",
    "examples.model_sensitivity",
    "examples.curve_types.experiment",
    "examples.initial_assignment.initial_assignment",
    "examples.glucose.glucose",
    "examples.hctz_fitting.simulations",
    "examples.hctz_fitting.fitting.petab_problem",
    "examples.petab.benchmark",
    "examples.sciml.lotka_volterra_fit",
    "examples.sensitivity.sensitivity_example",
    "examples.comparison.diff_example",
    "examples.demo.demo",
    "examples.repressilator.repressilator",
    "examples.repressilator.repressilator_scans",
]


def run_example(
    module: str, cwd: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """Run an example as a module in `cwd`."""
    if module.startswith("examples.sciml"):
        pytest.importorskip("petab_sciml", reason="the extra `sciml` is not installed")
    # the output of rich is utf-8 on every platform, the locale of windows is not
    env = dict(os.environ, PYTHONPATH=str(REPO_DIR), MPLBACKEND="Agg", PYTHONUTF8="1")
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize("module", SCRIPTS)
def test_example_script(module: str, tmp_path: Path) -> None:
    """Every example runs without an error and writes into the working directory."""
    run_example(module, tmp_path, *ARGUMENTS.get(module, []))


def test_neural_ode_example(tmp_path: Path) -> None:
    """The neural ODE starts from its network, is fitted in parallel and written."""
    result = run_example(
        "examples.sciml.neural_ode.fitting",
        tmp_path,
        "--max-nfev=4",
        "--runs=2",
        "--cores=2",
    )
    output = result.stdout
    number = r"([-+0-9.e]+)"

    def value(pattern: str) -> float:
        match = re.search(pattern.replace("NUMBER", number), output)
        assert match is not None, pattern
        return float(match.group(1))

    # the first run starts from the values of the network and improves them,
    # the first best cost of the console is the one of that fit
    start = value(r"(?m)^start\s+cost NUMBER")
    fit = value(r"best cost\s+NUMBER")
    # the cost of the start is printed before the fit
    start_line = re.search(r"(?m)^start\s+cost", output)
    assert start_line is not None
    assert start_line.start() < output.index("best cost")
    assert 0.0 < fit <= start

    # the multistart runs in the two workers
    assert re.search(r"workers\s+2\b", output)

    # the fitted problem is written and read back with the same likelihood
    written = value(r"the fit\s+log-likelihood NUMBER")
    read = value(r"read again\s+log-likelihood NUMBER")
    assert read == pytest.approx(written, rel=1e-6)
    results = tmp_path / "results" / "neural_ode"
    assert (results / "petab" / "problem.yaml").exists()
    assert (results / "fit" / "neural_ode_start" / "index.html").exists()
    assert (results / "fit" / "neural_ode_multistart" / "index.html").exists()
