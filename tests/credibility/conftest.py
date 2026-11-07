"""The adopted fit of the toy credibility package, fitted once per process."""

from pathlib import Path

import pytest

from sbmlsim import run
from sbmlsim.project import Project
from tests.credibility_toy import PROJECT

#: the fit of the toy credibility package with what an adopted fit needs
CREDIBILITY_FIT = {
    "project": "tests.credibility_toy:PROJECT",
    "subset": "DECAY",
    "repeats": 1,
    "seed": 1,
    "identifiability": True,
    "profile": {"max_points": 5},
    "fisher": False,
    "report": True,
    "mapping_figures": False,
    "adopt": True,
}


@pytest.fixture(scope="session")
def credibility_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Get the out of the adopted fit, with `fits/` and `adopted.json`.

    Session-scoped: a process, every worker of pytest-xdist, fits it once.
    """
    out = tmp_path_factory.mktemp("credibility") / "out"
    run.execute("fit", CREDIBILITY_FIT, out, workers=1)
    return out


@pytest.fixture(scope="session")
def credibility_project(credibility_out: Path) -> Project:
    """Get the toy credibility project with its adopted fit of every kind of evidence."""
    return PROJECT.replace(
        fits_dir=credibility_out / "fits", adopted=credibility_out / "adopted.json"
    )
