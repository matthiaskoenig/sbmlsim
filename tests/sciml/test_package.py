"""Tests of the package `sbmlsim.sciml` and its optional dependency."""

import re
import subprocess
import sys
from pathlib import Path

import sbmlsim


def _python(code: str) -> subprocess.CompletedProcess[str]:
    """Run code in a new interpreter, the imports of the tests do not count."""
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_the_missing_extra_is_named() -> None:
    """Without `petab_sciml` the import says how to install it."""
    result = _python(
        "import sys; sys.modules['petab_sciml'] = None; import sbmlsim.sciml"
    )
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert "pip install sbmlsim[sciml]" in result.stderr


def test_the_package_does_not_import_the_networks() -> None:
    """`sbmlsim` works without the extra, nothing imports `sbmlsim.sciml`."""
    result = _python(
        "import sys; sys.modules['petab_sciml'] = None\n"
        "import sbmlsim, sbmlsim.fit, sbmlsim.testsuite, sbmlsim.fit.petab_v2\n"
        "assert 'sbmlsim.sciml' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_the_package_does_not_import_torch() -> None:
    """`torch` is a dependency of the tests, no module of the package names it."""
    pattern = re.compile(r"^\s*(import|from)\s+torch\b", re.MULTILINE)
    modules = sorted(Path(sbmlsim.__file__).parent.rglob("*.py"))
    assert modules
    assert [str(path) for path in modules if pattern.search(path.read_text())] == []
