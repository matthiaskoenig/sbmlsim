"""The HTTP API over the panels of the served models.

The model, the panel and the application of the tests are the fixtures of
`tests/serve/conftest.py`; the tests of the runner build their own.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import sbmlsim
from examples.hctz_fitting import MODEL_PATH
from examples.panels.hctz import EGFR_STAGES, END, POINTS
from sbmlsim.panel import CONTRACT, Panel
from sbmlsim.project import Project
from sbmlsim.serve.app import FAILED, MAX_ERRORS, NO_APP, create_app
from sbmlsim.serve.runner import ModelEntry, ModelRegistry, PanelRunner
from tests.serve.conftest import APP, PANEL, SIMULATE

#: the reference of the project of the HCTZ example
HCTZ_REFERENCE = "examples.hctz_fitting.project:PROJECT"


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
    assert [panel["id"] for panel in body["panels"]] == [
        "renal",
        "dose_plasma",
        "dose_urine",
    ]
    assert body["panels"][0]["title"] == "Hydrochlorothiazide in renal impairment"
    assert body["contract"] == CONTRACT


def test_a_panel_is_served_as_the_contract(client: TestClient) -> None:
    body = client.get(PANEL).json()
    assert body["contract"] == "panel/1"
    assert [axis["id"] for axis in body["axes"]] == ["egfr", "dose"]
    assert body["limits"]["grid"] == 64


def test_the_app_route_answers_the_contract(client: TestClient) -> None:
    body = client.get(APP).json()
    assert body["contract"] == "app/1"
    assert body["pages"][0]["panels"] == ["renal"]
    assert body["title"] == "Hydrochlorothiazide"
    assert body["pages"][0]["id"] == "renal"
    assert [page["panels"] for page in body["pages"]] == [
        ["renal"],
        ["dose_plasma", "dose_urine"],
    ]
    assert body["licenses"] == ["CC BY-SA 4.0"]
    assert body["doi"] == "10.5281/zenodo.0000000"
    assert body["variants"] == []


def test_the_app_route_loads_no_model(client: TestClient, never_loads: None) -> None:
    """The app is the declaration of the project, no model is read for it."""
    assert client.get(APP).status_code == 200


def test_a_model_without_an_app_is_404(project: Project) -> None:
    without = ModelRegistry(
        {
            "hctz": ModelEntry(
                id="hctz",
                project=project.replace(app=None),
                title="Hydrochlorothiazide",
            )
        }
    )
    with TestClient(create_app(without)) as client:
        response = client.get(APP)
        assert response.status_code == 404
        assert response.json() == {"detail": NO_APP}
        assert "hctz" not in response.text


def test_the_app_of_an_unknown_model_is_404(client: TestClient) -> None:
    assert client.get("/api/models/nope/app").status_code == 404
    reference = "/api/models/pkdb_models.models.hctz:PROJECT/app"
    assert client.get(reference).status_code == 404


def test_simulate_answers_a_grid_with_provenance_and_a_hash(
    client: TestClient,
) -> None:
    # the renal function is a choice knob, so a grid over it runs its stages
    stages = len(EGFR_STAGES.choices)
    body = client.post(
        SIMULATE, json={"values": {"dose": 25.0}, "grid": {"egfr": stages}}
    ).json()
    assert body["contract"] == CONTRACT
    assert len(body["curves"]["[Cve_hctz]"]) == stages
    assert len(body["x"]) == POINTS + 1
    assert [axis["id"] for axis in body["dims"]] == ["egfr"]
    assert len(body["dims"][0]["values"]) == stages
    assert body["units"]["x"] == "min"
    assert body["units"]["[Cve_hctz]"] == "uM"
    assert body["status"] == {"ok": True, "points": stages, "failed": []}
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


def test_the_hash_covers_the_settings_of_the_simulator(
    client: TestClient, project: Project
) -> None:
    """The tolerances are the numbers of the curves as much as the engine is."""
    tightened = project.replace(
        simulator=project.simulator.model_copy(update={"relative_tolerance": 1e-10})
    )
    changed = ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=tightened, title="Hydrochlorothiazide")}
    )
    body = {"values": {}, "grid": {}}
    loose = client.post(SIMULATE, json=body).json()
    with TestClient(create_app(changed)) as other:
        tight = other.post(SIMULATE, json=body).json()
    assert tight["provenance"]["model_sha256"] == loose["provenance"]["model_sha256"]
    assert tight["provenance"]["engine"] == loose["provenance"]["engine"]
    assert tight["spec_hash"] != loose["spec_hash"]


def test_a_point_which_does_not_integrate_is_null_and_flagged(
    client: TestClient,
) -> None:
    """JSON has no `NaN`, a point which failed is `null` and is in the status."""
    body = client.post(SIMULATE, json={"values": {"egfr": -1.0}, "grid": {}}).json()
    assert body["status"] == {"ok": False, "points": 1, "failed": [0]}
    assert body["curves"]["[Cve_hctz]"][0][-1] is None


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


def test_an_unknown_knob_id_is_422_and_loads_no_model(
    client: TestClient, never_loads: None
) -> None:
    """The knobs are checked against the panel before a model is loaded."""
    response = client.post(SIMULATE, json={"values": {"kidney": 1.0}, "grid": {}})
    assert response.status_code == 422
    assert "kidney" in response.json()["detail"]
    response = client.post(SIMULATE, json={"values": {}, "grid": {"kidney": 3}})
    assert response.status_code == 422
    assert "kidney" in response.json()["detail"]


def test_a_value_error_of_a_run_is_500_and_carries_nothing_of_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the knob ids of a request are answered, no other `ValueError`.

    A `ValueError` below the runner, e.g. of a knob whose target is no entity
    of the model, carries the inside of the service in its message.
    """

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("'f_renal_function' is no selection of src/sbmlsim")

    monkeypatch.setattr(PanelRunner, "run", broken)
    response = client.post(SIMULATE, json={"values": {}, "grid": {}})
    assert response.status_code == 500
    assert response.json() == {"detail": FAILED}
    assert "f_renal_function" not in response.text


def test_the_body_of_an_invalid_request_does_not_repeat_it(
    client: TestClient, never_runs: None
) -> None:
    """The errors name what is wrong, they do not send the input back."""
    sent = "a_knob_of_no_panel"
    response = client.post(
        SIMULATE, json={"values": {sent: 1.0}, "grid": {"egfr": 1000}}
    )
    assert response.status_code == 422
    assert sent not in response.text
    assert set(response.json()) == {"detail"}
    error = response.json()["detail"][0]
    assert set(error) == {"type", "loc", "msg"}
    assert error["loc"] == ["body", "grid", "egfr"]


def test_a_large_body_is_not_answered_with_a_larger_one(
    client: TestClient, never_runs: None
) -> None:
    """A body of a value per error is answered with `MAX_ERRORS` of them."""
    body = {"values": {str(knob): float("inf") for knob in range(10_000)}, "grid": {}}
    response = client.post(
        SIMULATE,
        content=json.dumps(body),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert len(response.json()["detail"]) <= MAX_ERRORS
    assert len(response.content) < 1024


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_a_value_which_is_no_number_is_422_and_runs_nothing(
    client: TestClient, never_runs: None, value: str
) -> None:
    """`json.loads` reads them, the knobs of a panel do not take them."""
    response = client.post(
        SIMULATE,
        content=f'{{"values": {{"egfr": {value}}}, "grid": {{}}}}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


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


@pytest.fixture
def two_variants(project: Project, tmp_path: Path) -> TestClient:
    """A client of the HCTZ project with a second variant of another model.

    The variant is the SBML with one parameter changed, so a request which
    ignores its variant after validating it runs the wrong model.
    """
    text = MODEL_PATH.read_text(encoding="utf-8")
    marker = 'id="f_GFR_Qki" name="relationship GFR and Qki" value="151"'
    assert marker in text
    other = tmp_path / "other.xml"
    other.write_text(
        text.replace(marker, marker.replace('"151"', '"60"')), encoding="utf-8"
    )
    two = project.replace(variants={"default": MODEL_PATH, "other": other})
    entry = ModelEntry(id="hctz", project=two, title="Hydrochlorothiazide")
    return TestClient(create_app(ModelRegistry({"hctz": entry})))


def test_a_request_names_the_variant_which_ran_and_it_selects_another_model(
    two_variants: TestClient,
) -> None:
    body: dict[str, Any] = {"values": {"dose": 25.0}, "grid": {}}
    declared = two_variants.post(SIMULATE, json=body).json()
    named = two_variants.post(SIMULATE, json={**body, "variant": "other"}).json()
    assert declared["provenance"]["variant"] == "default"
    assert named["provenance"]["variant"] == "other"
    assert named["spec_hash"] != declared["spec_hash"]
    assert named["provenance"]["model_sha256"] != declared["provenance"]["model_sha256"]
    # the curves are the ones of the other model and not of the default
    assert named["curves"]["[Cve_hctz]"] != declared["curves"]["[Cve_hctz]"]


def test_a_request_without_a_variant_and_one_naming_the_default_are_one_request(
    two_variants: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[object] = []
    original = PanelRunner.run

    def counted(self: PanelRunner, *args: Any, **kwargs: Any) -> Any:
        runs.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PanelRunner, "run", counted)
    body: dict[str, Any] = {"values": {"dose": 25.0}, "grid": {}}
    left_out = two_variants.post(SIMULATE, json=body).json()
    named = two_variants.post(SIMULATE, json={**body, "variant": "default"}).json()
    assert named["spec_hash"] == left_out["spec_hash"]
    assert named == left_out
    # the second was served from the cache of the first
    assert len(runs) == 1


def test_an_unknown_variant_is_422_and_loads_no_model(
    client: TestClient, never_loads: None
) -> None:
    response = client.post(SIMULATE, json={"values": {}, "grid": {}, "variant": "x"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "'x'" in detail
    assert "default" in detail
    assert "/" not in detail
