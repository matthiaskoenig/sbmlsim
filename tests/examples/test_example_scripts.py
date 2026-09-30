"""Run the example scripts.

Every example is run as a module in a temporary working directory, so the
files it writes do not end up in the repository. An example which breaks
fails the test suite.

The examples which need optional tools (`examples.julia`, the AMICI and COPASI
scripts of `examples.comparison`) or the simulation experiment pipeline which
is being reworked (`examples.demo`, `examples.repressilator`) are not run
here, see `examples/README.md`.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

#: the root of the repository, `python -m examples.<module>` is run from here
REPO_DIR = Path(__file__).parent.parent.parent

#: examples which run offline and without optional dependencies
SCRIPTS = [
    "examples.timecourse",
    "examples.scan",
    "examples.fit_sampling",
    "examples.model_change",
    "examples.units",
    "examples.model_sensitivity",
    "examples.datagenerator",
    "examples.interpolation.interpolation_example",
    "examples.curve_types.experiment",
    "examples.initial_assignment.initial_assignment",
    "examples.glucose.glucose",
    "examples.hctz_fitting.simulations",
    "examples.hctz_fitting.fitting.petab_problem",
    "examples.petab.benchmark",
    "examples.sciml.lotka_volterra_fit",
    "examples.sensitivity.sensitivity_example",
    "examples.comparison.diff_example",
]


def run_example(
    module: str, cwd: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """Run an example as a module in `cwd`."""
    if module.startswith("examples.sciml"):
        pytest.importorskip("petab_sciml", reason="the extra `sciml` is not installed")
    env = dict(os.environ, PYTHONPATH=str(REPO_DIR), MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize("module", SCRIPTS)
def test_example_script(module: str, tmp_path: Path) -> None:
    """Every example runs without an error and writes into the working directory."""
    run_example(module, tmp_path)


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

    # the first run starts from the values of the network and improves them
    start = value(r"(?m)^start\s+cost NUMBER")
    fit = value(r"fit from start\s+cost NUMBER")
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
