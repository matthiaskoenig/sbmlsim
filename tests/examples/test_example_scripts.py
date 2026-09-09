"""Run the example scripts.

Every example is run as a module in a temporary working directory, so the
files it writes do not end up in the repository. An example which breaks
fails the test suite.

The examples which need optional tools (`examples.julia`) or the simulation
experiment pipeline which is being reworked (`examples.demo`,
`examples.repressilator`) are not run here, see `examples/README.md`.
"""

import os
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
    "examples.hctz.simulations",
    "examples.hctz.fitting.petab_problem",
    "examples.petab.benchmark",
    "examples.sensitivity.sensitivity_example",
]


@pytest.mark.parametrize("module", SCRIPTS)
def test_example_script(module: str, tmp_path: Path) -> None:
    """Every example runs without an error and writes into the working directory."""
    env = dict(os.environ, PYTHONPATH=str(REPO_DIR), MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
