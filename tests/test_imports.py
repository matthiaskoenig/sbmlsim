"""What the modules of sbmlsim import: petab, torch and roadrunner only when needed.

petab.v2 imports its SciML extension and with it torch, which costs seconds in
every process: in every worker of the tests, of a parallel fit and in every
example. PEtab is imported when a formula is compiled or a PEtab problem is read.

A library which must not load a simulator, e.g. the PBPK library of pkpdsim,
imports the light modules (`sbmlsim.licenses`, `sbmlsim.distrib`,
`sbmlsim.result.scan`, `sbmlsim.units`, `sbmlsim.release.settings`), which
import no roadrunner.
"""

import subprocess
import sys
from pathlib import Path

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


def test_import_sbmlsim_does_not_import_the_run_layer() -> None:
    code = (
        "import sys, sbmlsim; "
        "print(sorted(m for m in ('sbmlsim.run', 'sbmlsim.project', 'sbmlsim.simrun',"
        " 'simrun', 'sbmlsim.release', 'sbmlsim.project_checks') if m in sys.modules))"
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


def test_the_run_layer_imports_no_analysis() -> None:
    """`sbmlsim.run` imports the fit, the simulator and the project when it runs."""
    code = (
        "import sys, sbmlsim.run, sbmlsim.run.cli, sbmlsim.run.profiling, "
        "sbmlsim.run.project_kinds; "
        "print(sorted(m for m in ('sbmlsim.fit', 'sbmlsim.simulator', 'sbmlsim.project',"
        " 'matplotlib', 'sbmlsim.simrun', 'simrun') if m in sys.modules))"
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


@pytest.mark.parametrize(
    "module",
    [
        "sbmlsim.licenses",
        "sbmlsim.distrib",
        "sbmlsim.result.scan",
        "sbmlsim.units",
        "sbmlsim.release.settings",
    ],
)
def test_a_light_module_does_not_import_roadrunner(module: str) -> None:
    """A library which must not load a simulator imports these modules.

    The PBPK library of pkpdsim reads the support of a distrib formula and the
    start values of a scan result, defines units and writes the license texts
    of a model without loading roadrunner; a model package defines the release
    metadata of its project.
    """
    code = (
        f"import sys, {module}; "
        "print(sorted(m for m in ('roadrunner', 'COPASI', 'amici') if m in sys.modules))"
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


def test_the_project_checks_import_no_analysis() -> None:
    """The checks import the experiments, the fits and roadrunner when they run."""
    code = (
        "import sys, sbmlsim.project_checks; "
        "print(sorted(m for m in ('roadrunner', 'sbmlsim.fit', 'sbmlsim.experiment',"
        " 'sbmlsim.run') if m in sys.modules))"
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


def test_a_project_with_sensitivity_analyses_imports_no_runner() -> None:
    """A project lists its runners by reference, `sbmlsim` and the project import none.

    The runners of `sbmlsim.sensitivity.runner` and the table of their
    parameters, `sbmlsim.sensitivity.parameters`, are imported by the module of
    a runner, which the project resolves when an analysis runs.
    """
    code = (
        "import sys, sbmlsim, sbmlsim.project, tests.project_toy; "
        "print(sorted(m for m in ('sbmlsim.sensitivity.runner',"
        " 'sbmlsim.sensitivity.parameters', 'roadrunner') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
