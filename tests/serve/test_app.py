"""The HTTP API over the panels of the served models.

The panel of the tests is built here and bound to the project of the HCTZ
example with `Project.replace(panels=...)`, as a model whose SBML carries the
annotation of its panels is not part of this task; the tests of the runner
build their own, the two share no fixture.

The time unit of the HCTZ model is minutes, which is the unit the `end` of a
`PanelSimulation` and the x of a result are in.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sbmlsim
from examples.hctz_fitting import MODEL_PATH
from examples.hctz_fitting.project import PROJECT
from sbmlsim.panel import (
    CONTRACT,
    Choice,
    Knob,
    Panel,
    PanelCurve,
    PanelSimulation,
)
from sbmlsim.project import Project
from sbmlsim.serve.app import create_app
from sbmlsim.serve.runner import ModelEntry, ModelRegistry, PanelRunner

#: the reference of the project of the HCTZ example
HCTZ_REFERENCE = "examples.hctz_fitting.project:PROJECT"
#: one day in the time unit of the model
END = 24 * 60.0
#: the output points of the panel, fewer than a real panel so the tests are quick
POINTS = 50
#: the panel of the tests
PANEL = "/api/models/hctz/panels/renal"
#: the simulation of that panel
SIMULATE = f"{PANEL}/simulate"

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
        ),
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


@pytest.fixture(scope="module")
def app(registry: ModelRegistry) -> FastAPI:
    return create_app(registry)


@pytest.fixture(scope="module")
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


@pytest.fixture
def never_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if a panel is run at all."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the request reached the runner")

    monkeypatch.setattr(PanelRunner, "run", refuse)


def test_healthz_answers_without_a_model(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["sbmlsim"] == sbmlsim.__version__


def test_models_lists_the_registered_models(client: TestClient) -> None:
    body = client.get("/api/models").json()
    assert [model["id"] for model in body["models"]] == ["hctz"]
    assert body["models"][0]["title"] == "Hydrochlorothiazide"
    assert body["contract"] == CONTRACT


def test_the_panels_of_a_model_are_listed(client: TestClient) -> None:
    body = client.get("/api/models/hctz/panels").json()
    assert [panel["id"] for panel in body["panels"]] == ["renal"]
    assert body["panels"][0]["title"] == "Hydrochlorothiazide in renal impairment"
    assert body["contract"] == CONTRACT


def test_a_panel_is_served_as_the_contract(client: TestClient) -> None:
    body = client.get(PANEL).json()
    assert body["contract"] == "panel/1"
    assert [axis["id"] for axis in body["axes"]] == ["egfr", "dose"]
    assert body["limits"]["grid"] == 64


def test_simulate_answers_a_grid_with_provenance_and_a_hash(
    client: TestClient,
) -> None:
    body = client.post(
        SIMULATE, json={"values": {"dose": 25.0}, "grid": {"egfr": 5}}
    ).json()
    assert body["contract"] == CONTRACT
    assert len(body["curves"]["Cve_hctz"]) == 5
    assert len(body["x"]) == POINTS + 1
    assert [axis["id"] for axis in body["dims"]] == ["egfr"]
    assert len(body["dims"][0]["values"]) == 5
    assert body["units"]["x"] == "min"
    assert body["units"]["Cve_hctz"] == "umol"
    assert body["status"] == {"ok": True, "points": 5, "failed": []}
    provenance = body["provenance"]
    assert provenance["sbmlsim"] == sbmlsim.__version__
    assert provenance["model"] == "hctz"
    assert provenance["panel"] == "renal"
    assert provenance["engine"] == "roadrunner"
    assert len(provenance["model_sha256"]) == 64
    assert provenance["seconds"] > 0.0
    assert len(body["spec_hash"]) == 64


def test_the_hash_is_the_one_of_the_resolved_request(client: TestClient) -> None:
    """A knob which a request leaves out is its default, the same request."""
    given = client.post(SIMULATE, json={"values": {"dose": 25.0}, "grid": {}}).json()
    left_out = client.post(SIMULATE, json={"values": {}, "grid": {}}).json()
    assert left_out["spec_hash"] == given["spec_hash"]
    other = client.post(SIMULATE, json={"values": {"dose": 12.5}, "grid": {}}).json()
    assert other["spec_hash"] != given["spec_hash"]


def test_the_hash_covers_the_declaration_of_the_panel(
    client: TestClient, project: Project, panel: Panel
) -> None:
    """A panel of a project changes no SBML, so the model hash alone is not enough.

    A response is read back by its hash, so a changed declaration must not
    answer under the hash of the one before it.
    """
    shorter = panel.model_copy(
        update={"simulation": panel.simulation.model_copy(update={"end": END / 2.0})}
    )
    changed = ModelRegistry(
        {
            "hctz": ModelEntry(
                id="hctz",
                project=project.replace(panels={"renal": shorter}),
                title="Hydrochlorothiazide",
            )
        }
    )
    body = {"values": {}, "grid": {}}
    declared = client.post(SIMULATE, json=body).json()
    with TestClient(create_app(changed)) as other:
        redeclared = other.post(SIMULATE, json=body).json()
    # the same model, the same ids and the same request, another panel
    assert redeclared["provenance"]["model"] == declared["provenance"]["model"]
    assert redeclared["provenance"]["panel"] == declared["provenance"]["panel"]
    assert (
        redeclared["provenance"]["model_sha256"]
        == declared["provenance"]["model_sha256"]
    )
    assert redeclared["x"][-1] == declared["x"][-1] / 2.0
    assert redeclared["spec_hash"] != declared["spec_hash"]


def test_a_point_which_does_not_integrate_is_null_and_flagged(
    client: TestClient,
) -> None:
    """JSON has no `NaN`, a point which failed is `null` and is in the status."""
    body = client.post(SIMULATE, json={"values": {"egfr": -1.0}, "grid": {}}).json()
    assert body["status"] == {"ok": False, "points": 1, "failed": [0]}
    assert body["curves"]["Cve_hctz"][0][-1] is None


def test_an_unknown_model_or_panel_is_404(client: TestClient) -> None:
    assert client.get("/api/models/nope/panels/renal").status_code == 404
    reference = "/api/models/pkdb_models.models.hctz:PROJECT/panels/renal"
    assert client.get(reference).status_code == 404
    assert client.get("/api/models/hctz/panels/nope").status_code == 404
    assert client.get("/api/models/nope/panels").status_code == 404
    unknown = client.post(
        SIMULATE.replace("hctz", "nope"), json={"values": {}, "grid": {}}
    )
    assert unknown.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"values": {}, "grid": {"egfr": 100}},
        {"values": {}, "grid": {"egfr": 0}},
        {"values": {}, "grid": {"egfr": 9, "dose": 8}},
        {"values": {}, "grid": {"a": 2, "b": 2, "c": 2, "d": 2, "e": 2}},
        {"values": {str(knob): 1.0 for knob in range(10_000)}, "grid": {}},
    ],
)
def test_a_request_over_the_limits_is_422_and_runs_nothing(
    client: TestClient, never_runs: None, body: dict[str, Any]
) -> None:
    response = client.post(SIMULATE, json=body)
    assert response.status_code == 422


def test_a_request_of_an_unknown_field_is_422_and_runs_nothing(
    client: TestClient, never_runs: None
) -> None:
    response = client.post(SIMULATE, json={"values": {}, "grid": {}, "steps": 3})
    assert response.status_code == 422


def test_an_unknown_knob_id_is_422(client: TestClient) -> None:
    response = client.post(SIMULATE, json={"values": {"kidney": 1.0}, "grid": {}})
    assert response.status_code == 422
    assert "kidney" in response.json()["detail"]
    response = client.post(SIMULATE, json={"values": {}, "grid": {"kidney": 3}})
    assert response.status_code == 422
    assert "kidney" in response.json()["detail"]


def test_the_error_body_carries_no_traceback_and_no_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"'{MODEL_PATH}' of {HCTZ_REFERENCE} did not integrate")

    monkeypatch.setattr(PanelRunner, "run", broken)
    response = client.post(SIMULATE, json={"values": {}, "grid": {}})
    assert response.status_code == 500
    assert set(response.json()) == {"detail"}
    text = response.text
    assert "Traceback" not in text
    assert "RuntimeError" not in text
    assert str(MODEL_PATH) not in text
    assert HCTZ_REFERENCE not in text


def test_a_static_directory_is_served_under_the_api(
    registry: ModelRegistry, tmp_path: Path
) -> None:
    (tmp_path / "index.html").write_text("<p>the panel</p>", encoding="utf-8")
    with TestClient(create_app(registry, static=tmp_path)) as client:
        assert client.get("/").text == "<p>the panel</p>"
        body = client.get("/api/models").json()
        assert [model["id"] for model in body["models"]] == ["hctz"]


def test_the_origins_of_a_request_are_the_ones_the_service_is_given(
    registry: ModelRegistry,
) -> None:
    origin = "https://example.org"
    with TestClient(create_app(registry, allow_origins=[origin])) as client:
        response = client.get("/api/models", headers={"origin": origin})
        assert response.headers["access-control-allow-origin"] == origin
    with TestClient(create_app(registry)) as client:
        response = client.get("/api/models", headers={"origin": origin})
        assert "access-control-allow-origin" not in response.headers
