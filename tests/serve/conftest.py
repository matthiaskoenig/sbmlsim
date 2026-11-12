"""The model, the panel and the application the tests of the service share.

The panel is built here and bound to the project of the HCTZ example with
`Project.replace(panels=...)`, as a model whose SBML carries the annotation of
its panels is not part of these tests.

The time unit of the HCTZ model is minutes, which is the unit the `end` of a
`PanelSimulation` and the x of a result are in.

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

from examples.hctz_fitting import MODEL_PATH
from examples.hctz_fitting.project import PROJECT
from sbmlsim.panel import Choice, Knob, Panel, PanelCurve, PanelData, PanelSimulation
from sbmlsim.project import Project
from sbmlsim.serve.app import create_app
from sbmlsim.serve.runner import ModelEntry, ModelRegistry, PanelRunner

#: one day in the time unit of the model
END = 24 * 60.0
#: the output points of the panel, fewer than a real panel so the tests are quick
POINTS = 50
#: the panel of the tests
PANEL = "/api/models/hctz/panels/renal"
#: the simulation of that panel
SIMULATE = f"{PANEL}/simulate"
#: the study data of that panel
DATA = f"{PANEL}/data"

#: the inverse of the molecular weight of hydrochlorothiazide, with which the
#: mass a study measured is the substance the model carries, as the
#: experiments of the HCTZ example convert it
MR_HCTZ = "1/(297.7 g/mole)"

EGFR = Knob(
    id="egfr",
    label="renal function",
    target="f_renal_function",
    unit="dimensionless",
    kind="range",
    min=0.2,
    max=1.0,
    steps=5,
    default=1.0,
)
DOSE = Knob(
    id="dose",
    label="dose",
    target="PODOSE_hctz",
    unit="mg",
    kind="choice",
    choices=(
        Choice(label="12.5 mg", value=12.5),
        Choice(label="25 mg", value=25.0),
        Choice(label="50 mg", value=50.0),
    ),
    default=25.0,
)

#: the cumulative amount of hydrochlorothiazide in the urine of the fourth
#: subject of Beermann 1976 after 5 mg orally, measured in mg and delivered in
#: the unit of the curve it belongs under, the `umol` of `Aurine_hctz`
URINE = PanelData(
    id="beermann1976_urine",
    label="Beermann 1976 (5 mg, oral)",
    curve="Aurine_hctz",
    dataset="Beermann1976_Tab1A",
    x="time",
    y="value",
    sel={"label": "amount_urine_hctz5po_4"},
    conversion=MR_HCTZ,
)


@pytest.fixture(scope="module")
def panel() -> Panel:
    return Panel(
        id="renal",
        title="Hydrochlorothiazide in renal impairment",
        subtitle="the plasma concentration over the renal function",
        x_label="time (min)",
        knobs=(EGFR, DOSE),
        curves=(
            PanelCurve(id="Cve_hctz", label="hydrochlorothiazide plasma", unit="umol"),
            PanelCurve(
                id="Aurine_hctz", label="hydrochlorothiazide urine", unit="umol"
            ),
        ),
        data=(URINE,),
        # the panel pre-initializes the dose to zero, which the knob overrides
        simulation=PanelSimulation(
            end=END, points=POINTS, preinit={"PODOSE_hctz": 0.0, "BW": 75.0}
        ),
    )


@pytest.fixture(scope="module")
def project(panel: Panel) -> Project:
    """The project of the HCTZ example with the panel of the tests."""
    return PROJECT.replace(variants={"default": MODEL_PATH}, panels={"renal": panel})


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
