"""The check of the installed libroadrunner at `import sbmlsim`.

`sbmlsim` simulates with a build of the `develop` branch of libroadrunner,
which the index of the builds serves and PyPI does not have, so an
environment with another libroadrunner, e.g. the release of PyPI, gives wrong
results rather than an error. The check reports it at the import, with the
commands which install the build.
"""

import subprocess
import sys

import pytest

from sbmlsim import engine
from sbmlsim.engine import (
    ROADRUNNER_INDEX,
    SKIP_ROADRUNNER_CHECK,
    check_roadrunner,
    installed_roadrunner,
    roadrunner_problem,
    roadrunner_requirement,
    satisfies_requirement,
)

#: the requirement of the package, as `pyproject.toml` has it
REQUIREMENT = "libroadrunner>=2.11.1.dev655"


def test_the_release_of_pypi_is_reported() -> None:
    problem = roadrunner_problem("2.11.0", REQUIREMENT)
    assert problem is not None
    assert "2.11.0" in problem
    assert REQUIREMENT in problem
    assert ROADRUNNER_INDEX in problem
    assert "pip install" in problem


@pytest.mark.parametrize(
    "version", ["2.11.1.dev655", "2.11.1.dev700", "2.11.1.post0.dev12", "2.12.0"]
)
def test_a_build_of_the_index_is_no_problem(version: str) -> None:
    """A later build, a build after a release and a release itself satisfy it."""
    assert roadrunner_problem(version, REQUIREMENT) is None


def test_a_missing_libroadrunner_is_reported() -> None:
    problem = roadrunner_problem(None, REQUIREMENT)
    assert problem is not None
    assert "not installed" in problem
    assert ROADRUNNER_INDEX in problem


def test_the_environment_of_the_tests_satisfies_the_requirement() -> None:
    """The environment is installed from the lock, which pins a build of the index."""
    requirement = roadrunner_requirement()
    assert requirement is not None
    assert requirement.replace(" ", "") == REQUIREMENT
    assert installed_roadrunner() is not None
    check_roadrunner()


def test_the_check_raises_for_the_release_of_pypi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "installed_roadrunner", lambda: "2.11.0")
    monkeypatch.setattr(engine, "roadrunner_requirement", lambda: REQUIREMENT)
    with pytest.raises(ImportError, match="libroadrunner"):
        check_roadrunner()


def test_the_variable_skips_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "installed_roadrunner", lambda: None)
    monkeypatch.setattr(engine, "roadrunner_requirement", lambda: REQUIREMENT)
    monkeypatch.setenv(SKIP_ROADRUNNER_CHECK, "1")
    check_roadrunner()


#: an import of the package with another libroadrunner in the metadata
IMPORT_WITH_THE_RELEASE = (
    "import importlib.metadata as metadata\n"
    "version = metadata.version\n"
    "metadata.version = lambda name: "
    "'2.11.0' if name == 'libroadrunner' else version(name)\n"
    "import sbmlsim\n"
)


def test_an_import_with_the_release_of_pypi_names_the_install() -> None:
    """The import fails with the commands which install the build on the console."""
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITH_THE_RELEASE],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode != 0, result.stdout
    assert "libroadrunner 2.11.0" in result.stderr
    assert REQUIREMENT in result.stderr
    assert ROADRUNNER_INDEX in result.stderr
    assert "pip install" in result.stderr


def test_an_import_with_the_variable_set_does_not_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SKIP_ROADRUNNER_CHECK, "1")
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITH_THE_RELEASE],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("installed", "requirement", "satisfied"),
    [
        ("2.11.1.dev655", REQUIREMENT, True),
        ("2.11.0", REQUIREMENT, False),
        # a requirement of another form, parsed with `packaging.requirements`
        ("2.11.1.dev655", "libroadrunner>=2.11.1.dev655,<3", True),
        ("3.0.0", "libroadrunner>=2.11.1.dev655,<3", False),
        ("2.11.1.dev655", "libroadrunner==2.11.1.dev655", True),
        ("2.11.1.dev700", "libroadrunner==2.11.1.dev655", False),
    ],
)
def test_the_comparison_of_a_requirement(
    installed: str, requirement: str, satisfied: bool
) -> None:
    assert satisfies_requirement(installed, requirement) is satisfied


def test_the_import_does_not_pull_the_requirements_of_packaging() -> None:
    """The check of the install compares with `packaging.version`, 2 ms.

    `packaging.requirements` imports `packaging.tags` and costs 30 ms of every
    import of the package, which the comparison of the floor of the
    requirement does not pay.
    """
    code = (
        "import sys, sbmlsim; "
        "print(sorted(m for m in ('packaging.requirements', 'packaging.specifiers',"
        " 'packaging.tags') if m in sys.modules))"
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


def test_the_requirement_of_the_package_is_found_in_its_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A distribution whose name begins with it and a requirement of an extra are not it."""
    monkeypatch.setattr(
        engine.metadata,
        "requires",
        lambda name: [
            "libroadrunner-plugins>=1.0",
            'libroadrunner>=9.9; extra == "dev"',
            REQUIREMENT,
        ],
    )
    assert roadrunner_requirement() == REQUIREMENT
