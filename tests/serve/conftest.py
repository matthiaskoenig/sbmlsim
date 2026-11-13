"""The model, the panel and the application the tests of the service share.

The panel and the app are declared in `examples/panels/hctz.py`, which the end
to end tests of sbmlsim-web serve as well, and come with its `PROJECT`.

The time unit of the HCTZ model is minutes, which is the unit the `end` of a
`PanelSimulation` and the x of a result are in.

The app of the project is one page of that one panel, so the route of the app
answers what a model app is configured with (`sbmlsim.panel.App`).

The panel draws the plasma and the urine of hydrochlorothiazide and declares
the cumulative amount in the urine of Beermann 1976 as the study data under
the curve of the urine: the data of that study is a mass, which `MR_HCTZ`
converts into the substance the model carries.

The panel, the project and the registry are of the module, so a model is
loaded once per file, while the application and its client are of the test:
the cache of the responses belongs to the application
(`sbmlsim.serve.cache`), so every test is answered by an empty one and a test
of a run reaches the runner.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.panels.hctz import HCTZ_PANEL, PROJECT
from sbmlsim.panel import Panel
from sbmlsim.project import Project
from sbmlsim.serve.app import create_app
from sbmlsim.serve.runner import ModelEntry, ModelRegistry, PanelRunner

#: the panel of the tests
PANEL = "/api/models/hctz/panels/renal"
#: the simulation of that panel
SIMULATE = f"{PANEL}/simulate"
#: the study data of that panel
DATA = f"{PANEL}/data"
#: the app of the model of the tests
APP = "/api/models/hctz/app"


@pytest.fixture(scope="module")
def panel() -> Panel:
    return HCTZ_PANEL


@pytest.fixture(scope="module")
def project() -> Project:
    """The project of the HCTZ panel example, which carries its panel and its app."""
    return PROJECT


@pytest.fixture(scope="module")
def registry(project: Project) -> ModelRegistry:
    return ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=project, title="Hydrochlorothiazide")}
    )


@pytest.fixture
def app(registry: ModelRegistry) -> FastAPI:
    """An application of its own, whose cache of the responses is empty."""
    return create_app(registry)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


@pytest.fixture
def never_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if a panel is run at all."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the request reached the runner")

    monkeypatch.setattr(PanelRunner, "run", refuse)


@pytest.fixture
def never_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if a model is loaded or read at all."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the request reached the model")

    monkeypatch.setattr(PanelRunner, "_warm_model", refuse)
