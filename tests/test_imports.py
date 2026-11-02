"""The core of sbmlsim imports neither petab nor torch.

petab.v2 imports its SciML extension and with it torch, which costs seconds in
every process: in every worker of the tests, of a parallel fit and in every
example. PEtab is imported when a formula is compiled or a PEtab problem is read.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module", ["sbmlsim", "sbmlsim.simulator", "sbmlsim.experiment", "sbmlsim.fit"]
)
def test_the_core_does_not_import_petab(module: str) -> None:
    code = (
        f"import sys, {module}; "
        "print(sorted(m for m in ('petab', 'torch', 'petab_sciml') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_a_run_without_uncertain_values_does_not_import_the_uncertainty() -> None:
    """The record of a scan checks its values for uncertain ones (#413).

    `sbmlsim.uncertainty` imports the fit, matplotlib and scipy.stats, about a
    second which a run without an uncertain value does not pay.
    """
    code = (
        "import sys\n"
        "from sbmlsim.model import RoadrunnerSBMLModel\n"
        "from sbmlsim.resources import REPRESSILATOR_SBML\n"
        "from sbmlsim.simulation import Dimension, Scan, Simulation\n"
        "from sbmlsim.simulator import Simulator\n"
        "scan = Scan(Simulation(end=10, steps=10, preinit_changes={'n': 2.0}),"
        " [Dimension('d', values={'n': [1.5, 2.0]})])\n"
        "result = Simulator(n_workers=1).run("
        "RoadrunnerSBMLModel(source=REPRESSILATOR_SBML), scan)\n"
        "assert result.ds.attrs['scan']\n"
        "print(sorted(m for m in ('sbmlsim.uncertainty', 'sbmlsim.fit')"
        " if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"
