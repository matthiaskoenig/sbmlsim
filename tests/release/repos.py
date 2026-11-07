"""Git repositories for the tests of the release of a model repository.

The tests run real git on temporary repositories: a source repository (a
monorepo of models) with a copy of the toy project as the package
`release_toy`, and a release repository with its bare remote. The fixtures are
in `conftest.py`; the helpers here are imported by the tests.
"""

import json
import shutil
import sys
from pathlib import Path

from sbmlsim.project import Project, load_project
from sbmlsim.release.source import run
from tests.project_toy import TOY_PATH

#: a git without the configuration of the user, with a fixed author
GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.org",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.org",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}

#: the package of the copy of the toy project in the source repository
PACKAGE = "release_toy"
#: the reference of the project of the copy
REFERENCE = f"{PACKAGE}:PROJECT"
#: the lock of the source repository, sbmlsim 0.8.0 and sbmlutils 0.9.0
SOURCE_LOCK = Path(__file__).parents[1] / "data" / "release" / "uv.lock"
#: the sbmlsim of the lock of the source repository
SBMLSIM = "0.8.0"


def git(cwd: Path, *args: str) -> str:
    """Run git in a folder and return its output."""
    return run(["git", *args], cwd)


def init_repo(path: Path) -> None:
    """Create a git repository on the branch `main`."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-b", "main")


def copy_toy(target: Path, *, credibility: bool = True) -> Path:
    """Copy the toy project as the package `release_toy`.

    Args:
        target: the folder of the package, created.
        credibility: keep the declaration of the credibility of the toy, a
            plain object which the kind `credibility` does not assess.

    Returns:
        The folder of the package.
    """
    shutil.copytree(
        TOY_PATH,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "results"),
    )
    for path in target.glob("*.py"):
        text = path.read_text(encoding="utf-8").replace("tests.project_toy", PACKAGE)
        if not credibility and path.name == "__init__.py":
            text = text.replace('    credibility=".credibility:CREDIBILITY",\n', "")
        path.write_text(text, encoding="utf-8")
    return target


def adopt_records(package: Path, records_out: Path) -> str:
    """Copy the adopted fit of an out of `adopt_toy_fit` into `fits/` of a package.

    Returns:
        The id of the adopted fit.
    """
    (fit_id,) = json.loads((records_out / "adopted.json").read_text(encoding="utf-8"))[
        "fits"
    ].values()
    fits = package / "fits"
    shutil.copytree(records_out / "fits" / fit_id, fits / fit_id)
    (fits / "adopted.json").write_text(
        json.dumps({"fits": {"DECAY": fit_id}}), encoding="utf-8"
    )
    return fit_id


def forget_package() -> None:
    """Remove the modules of the copied toy, so the next copy is imported anew."""
    for name in list(sys.modules):
        if name == PACKAGE or name.startswith(f"{PACKAGE}."):
            del sys.modules[name]


def load_toy() -> Project:
    """Import the project of the copied toy."""
    forget_package()
    return load_project(REFERENCE)
