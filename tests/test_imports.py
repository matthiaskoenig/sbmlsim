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
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
