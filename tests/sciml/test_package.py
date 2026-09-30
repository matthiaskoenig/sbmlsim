"""Tests of the package `sbmlsim.sciml` and its optional dependency."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import sbmlsim
import sbmlsim.sciml
from sbmlsim.sciml import NetworkError, NetworkImportError, UnsupportedLayerError


def _python(code: str, path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run code in a new interpreter, the imports of the tests do not count."""
    if path is not None:
        code = f"import sys; sys.path.insert(0, {str(path)!r})\n{code}"
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


def test_a_missing_module_of_the_extra_is_not_the_extra(tmp_path: Path) -> None:
    """A module which `petab_sciml` misses is reported as it is."""
    (tmp_path / "petab_sciml").mkdir()
    (tmp_path / "petab_sciml" / "__init__.py").write_text("import not_a_module_xyz\n")
    result = _python("import sbmlsim.sciml", path=tmp_path)
    assert result.returncode != 0
    assert "No module named 'not_a_module_xyz'" in result.stderr
    assert "sbmlsim[sciml]" not in result.stderr


def test_the_package_does_not_import_the_networks() -> None:
    """`sbmlsim` works without the extra, nothing imports `sbmlsim.sciml`."""
    result = _python(
        "import sys; sys.modules['petab_sciml'] = None\n"
        "import sbmlsim, sbmlsim.fit, sbmlsim.testsuite, sbmlsim.fit.petab_v2\n"
        "import sbmlsim.fit.cli, sbmlsim.fit.derived, sbmlsim.fit.runner\n"
        "from sbmlsim.fit.petab_v2.extension import known_extensions\n"
        "assert known_extensions() == {'sbmlsim'}, known_extensions()\n"
        "assert 'sbmlsim.sciml' not in sys.modules\n"
        "assert 'sbmlsim.fit.petab_v2.sciml' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def _imported_modules(path: Path) -> set[str]:
    """Get the top level packages a module imports."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize(
    ("code", "imported"),
    [
        ("import torch", True),
        ("import numpy, torch", True),
        ("import torch.nn as nn", True),
        ("from torch import nn", True),
        ("def f():\n    import torch\n", True),
        ("import torchvision", False),
        ("from . import torch", False),
        ("x = 'import torch'", False),
    ],
)
def test_the_imports_of_a_module(tmp_path: Path, code: str, imported: bool) -> None:
    """The imports are read from the syntax tree, not from the text."""
    path = tmp_path / "module.py"
    path.write_text(code)
    assert ("torch" in _imported_modules(path)) == imported


def test_the_package_does_not_import_torch() -> None:
    """`torch` is a dependency of the tests, no module of the package names it."""
    modules = sorted(Path(sbmlsim.__file__).parent.rglob("*.py"))
    assert modules
    assert [str(path) for path in modules if "torch" in _imported_modules(path)] == []


def test_the_exports() -> None:
    """The package exports what the user of a network needs, not the backends."""
    assert sorted(sbmlsim.sciml.__all__) == [
        "Hybridization",
        "Network",
        "NetworkCompilationError",
        "NetworkError",
        "NetworkHybridizationError",
        "NetworkImportError",
        "NetworkInput",
        "NetworkParameters",
        "NetworkPattern",
        "UnsupportedLayerError",
        "compile_network",
        "compiled_path",
        "network_fit_parameters",
        "nominal_parameters",
    ]


def test_the_errors_have_a_common_base() -> None:
    """A caller catches every error of a network with `NetworkError`."""
    assert issubclass(NetworkImportError, NetworkError)
    assert issubclass(NetworkImportError, ValueError)
    assert issubclass(UnsupportedLayerError, NetworkError)
    assert issubclass(UnsupportedLayerError, NotImplementedError)
    error = UnsupportedLayerError("net1", "layer1", "LSTM", "it is not implemented")
    assert isinstance(error, NetworkError)
