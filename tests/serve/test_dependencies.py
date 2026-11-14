"""The service imports only what the package and its extra `web` declare.

A module of `sbmlsim.serve` which imports a package another one brings along,
e.g. anyio, which starlette depends on, works until that package drops it, so
every third-party import of the service is a dependency of the package or of
the extra `web` in `pyproject.toml`.
"""

import ast
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import sbmlsim

ROOT = Path(__file__).parents[2]
SERVE = Path(sbmlsim.__file__).parent / "serve"


def _imported(path: Path) -> set[str]:
    """Get the top level modules a module imports, at any depth of its code."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.partition(".")[0])
    return names


def _declared() -> set[str]:
    """Get the distributions of the dependencies and of the extra `web`."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [
        *project["project"]["dependencies"],
        *project["project"]["optional-dependencies"]["web"],
    ]
    return {canonicalize_name(Requirement(one).name) for one in requirements}


def test_the_service_imports_only_declared_dependencies() -> None:
    imported = set().union(*(_imported(path) for path in SERVE.glob("*.py")))
    third_party = imported - set(sys.stdlib_module_names) - {"sbmlsim"}
    distributions = packages_distributions()
    declared = _declared()
    undeclared = {
        name: distributions.get(name, [name])
        for name in sorted(third_party)
        if not any(
            canonicalize_name(one) in declared
            for one in distributions.get(name, [name])
        )
    }
    assert undeclared == {}
    # the import which pinned this test, anyio of `create_app`
    assert "anyio" in third_party
