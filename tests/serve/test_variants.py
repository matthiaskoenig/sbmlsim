"""The declaration of the panels of a model in each of its variants.

The declaration of a panel in a variant is the one of the SBML of that variant
(`sbmlsim.project.Project.panels_of`), so the panel, its study data, the app
and a simulation which name a variant are of the declaration of that variant.
The variants of these tests run one SBML, a copy of the HCTZ model, so every
difference of an answer is one of the declaration: the variant `other`
declares the renal function with the default of an impaired kidney and other
titles (`Project.variant_panels`), and a panel `extra` which the default
variant does not declare.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from examples.hctz_fitting import MODEL_PATH
from examples.panels.hctz import EGFR_STAGES, URINE
from sbmlsim import panel_sbml
from sbmlsim.panel import AppVariant, Choice, Panel, Scenario
from sbmlsim.project import Project
from sbmlsim.serve import runner as runner_module
from sbmlsim.serve.app import create_app
from sbmlsim.serve.runner import ModelEntry, ModelRegistry
from tests.serve.conftest import APP, DATA, PANEL, SIMULATE

#: the reference of the project of the HCTZ example
HCTZ_REFERENCE = "examples.hctz_fitting.project:PROJECT"

#: the renal function of the variant `other`, whose default is impaired
IMPAIRED = EGFR_STAGES.model_copy(update={"default": 0.4})


def _other(panel: Panel) -> Panel:
    """The declaration of a panel in the variant `other`."""
    return panel.model_copy(
        update={
            "title": f"{panel.title} (other)",
            "knobs": (IMPAIRED, *panel.knobs[1:]),
            "data": tuple(
                one.model_copy(update={"label": f"{one.label} (other)"})
                for one in panel.data
            ),
        }
    )


@pytest.fixture(scope="module")
def declared(project: Project, tmp_path_factory: pytest.TempPathFactory) -> Project:
    """The HCTZ project of two variants of one model, declared otherwise."""
    other = tmp_path_factory.mktemp("variants") / "other.xml"
    other.write_bytes(MODEL_PATH.read_bytes())
    panels = project.panels_of()
    app = project.app
    assert app is not None
    extra = panels["renal_plasma"].model_copy(update={"id": "extra"})
    return project.replace(
        variants={"default": MODEL_PATH, "other": other},
        variant_panels={
            "other": {
                "renal_plasma": _other(panels["renal_plasma"]),
                "renal_urine": _other(panels["renal_urine"]),
                "extra": extra,
            }
        },
        app=app.model_copy(
            update={
                "variants": (
                    AppVariant(id="default", label="healthy"),
                    AppVariant(id="other", label="impaired"),
                )
            }
        ),
    )


@pytest.fixture(scope="module")
def registry(declared: Project) -> ModelRegistry:
    entry = ModelEntry(id="hctz", project=declared, title="Hydrochlorothiazide")
    return ModelRegistry({"hctz": entry})


def test_the_registry_of_the_variants_is_served(
    declared: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every page of the app is checked in both variants, which pass."""
    monkeypatch.setattr(runner_module, "load_project", lambda reference: declared)
    registry = ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    assert registry.entry("hctz").app_variants() == ("default", "other")


def test_the_panel_of_a_variant_is_its_declaration(client: TestClient) -> None:
    default = client.get(PANEL).json()
    other = client.get(PANEL, params={"variant": "other"}).json()
    named = client.get(PANEL, params={"variant": "default"}).json()
    assert default == named
    assert default["panel"]["title"] == "Plasma concentration over the renal function"
    assert other["panel"]["title"] == f"{default['panel']['title']} (other)"
    assert [axis["default"] for axis in other["axes"]] == [0.4, 25.0]
    assert [axis["default"] for axis in default["axes"]] == [1.0, 25.0]


def test_the_study_data_of_a_variant_is_the_one_it_declares(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its units are resolved on the model of the variant, which a run uses."""
    loads: list[Path] = []
    original = runner_module._load

    def counted(entry: ModelEntry, path: Path) -> Any:
        loads.append(path)
        return original(entry, path)

    monkeypatch.setattr(runner_module, "_load", counted)
    default = client.get(DATA).json()
    other = client.get(DATA, params={"variant": "other"}).json()
    assert [path.name for path in loads] == [MODEL_PATH.name, "other.xml"]
    assert [one["label"] for one in default["series"]] == [URINE.label]
    assert [one["label"] for one in other["series"]] == [f"{URINE.label} (other)"]
    # the same rows of the same dataset, delivered in the same units
    assert other["series"][0]["y"] == default["series"][0]["y"]
    assert other["series"][0]["unit"] == default["series"][0]["unit"]


def test_the_app_of_a_variant_is_worked_out_of_the_panels_of_that_variant(
    client: TestClient,
) -> None:
    default = client.get(APP).json()
    other = client.get(APP, params={"variant": "other"}).json()
    assert [one["title"] for one in other["heads"]["renal"]] == [
        f"{one['title']} (other)" for one in default["heads"]["renal"]
    ]
    assert other["heads"]["dose"] == default["heads"]["dose"]
    egfr = {
        name: body["shared"]["renal"][0]["knob"]
        for name, body in (("default", default), ("other", other))
    }
    assert (egfr["default"]["default"], egfr["other"]["default"]) == (1.0, 0.4)
    assert other["shared"]["renal"][0]["panels"][0]["title"].endswith("(other)")
    # the declaration of the app is one for every variant
    assert {key: other[key] for key in ("pages", "variants", "title")} == {
        key: default[key] for key in ("pages", "variants", "title")
    }


def test_a_simulation_of_a_variant_runs_the_declaration_of_that_variant(
    client: TestClient,
) -> None:
    """The variants run one SBML, so the curves differ by the declaration alone.

    A request which leaves the renal function out runs at its default, which
    is 1.0 in the default variant and 0.4 in `other`; the run of `other`
    equals the run of the default declaration at 0.4.
    """
    body: dict[str, Any] = {"values": {"dose": 25.0}, "grid": {}}
    default = client.post(SIMULATE, json=body).json()
    other = client.post(SIMULATE, json={**body, "variant": "other"}).json()
    assert other["provenance"]["variant"] == "other"
    assert other["provenance"]["model_sha256"] == default["provenance"]["model_sha256"]
    assert other["spec_hash"] != default["spec_hash"]
    assert other["curves"] != default["curves"]
    at = client.post(SIMULATE, json={"values": {"dose": 25.0, "egfr": 0.4}, "grid": {}})
    for curve, rows in other["curves"].items():
        assert np.array_equal(np.asarray(rows), np.asarray(at.json()["curves"][curve]))


def test_a_panel_one_variant_declares_is_no_panel_of_the_other(
    client: TestClient,
) -> None:
    extra = "/api/models/hctz/panels/extra"
    assert client.get(extra).status_code == 404
    assert client.get(extra, params={"variant": "default"}).status_code == 404
    declared = client.get(extra, params={"variant": "other"})
    assert declared.status_code == 200
    assert declared.json()["panel"]["id"] == "extra"
    body = {"values": {}, "grid": {}}
    missing = client.post(f"{extra}/simulate", json=body)
    assert missing.status_code == 404
    assert "extra" in missing.json()["detail"]
    ran = client.post(f"{extra}/simulate", json={**body, "variant": "other"})
    assert ran.status_code == 200
    # the list of the panels is the one of the default variant
    listed = client.get("/api/models/hctz/panels").json()["panels"]
    assert "extra" not in [one["id"] for one in listed]


@pytest.mark.parametrize("route", [PANEL, DATA, APP])
def test_an_unknown_variant_of_a_route_is_422_and_loads_no_model(
    client: TestClient, never_loads: None, route: str
) -> None:
    response = client.get(route, params={"variant": "female"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "'female'" in detail
    assert "['default', 'other']" in detail
    assert "/" not in detail


def test_the_app_of_a_variant_it_does_not_name_is_422(
    declared: Project, never_loads: None
) -> None:
    """The app is served in the variants it was checked in and in no other."""
    app = declared.app
    assert app is not None
    named = declared.replace(
        app=app.model_copy(
            update={"variants": (AppVariant(id="default", label="healthy"),)}
        )
    )
    entry = ModelEntry(id="hctz", project=named, title="Hydrochlorothiazide")
    with TestClient(create_app(ModelRegistry({"hctz": entry}))) as client:
        response = client.get(APP, params={"variant": "other"})
        assert response.status_code == 422
        assert "offers no variant 'other'" in response.json()["detail"]
        assert client.get(APP, params={"variant": "default"}).status_code == 200
        # a panel of a variant the app does not name is still a panel
        assert client.get(PANEL, params={"variant": "other"}).status_code == 200


def test_a_page_whose_panel_one_variant_does_not_declare_is_refused(
    declared: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = declared.app
    assert app is not None
    shown = declared.replace(
        app=app.model_copy(
            update={
                "pages": (
                    app.pages[0].model_copy(
                        update={"panels": (*app.pages[0].panels, "extra")}
                    ),
                )
            }
        )
    )
    monkeypatch.setattr(runner_module, "load_project", lambda reference: shown)
    with pytest.raises(ValueError, match="variant does not declare") as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    message = str(error.value)
    for named in ("'hctz'", "variant 'default'", "page 'renal'", "['extra']"):
        assert named in message


def test_a_page_whose_panels_differ_on_a_knob_in_one_variant_is_refused(
    declared: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The variant `other` gives one panel of the page the impaired default only."""
    panels = dict(declared.variant_panels["other"])
    panels["renal_urine"] = panels["renal_urine"].model_copy(
        update={"knobs": (EGFR_STAGES, *panels["renal_urine"].knobs[1:])}
    )
    differing = declared.replace(variant_panels={"other": panels})
    monkeypatch.setattr(runner_module, "load_project", lambda reference: differing)
    with pytest.raises(ValueError, match=r"'egfr' differently \(default\)") as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    assert "variant 'other'" in str(error.value)


def test_a_scenario_of_a_value_one_variant_does_not_take_is_refused(
    declared: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scenarios of a page are checked in every variant of the app."""
    panels = dict(declared.variant_panels["other"])
    two = IMPAIRED.model_copy(
        update={
            "choices": (
                Choice(label="impaired", value=0.4),
                Choice(label="1", value=1.0),
            )
        }
    )
    for one in ("renal_plasma", "renal_urine"):
        panels[one] = panels[one].model_copy(
            update={"knobs": (two, *panels[one].knobs[1:])}
        )
    app = declared.app
    assert app is not None
    scenario = Scenario(id="mild", label="mildly impaired", values={"egfr": 0.7})
    with_scenario = declared.replace(
        variant_panels={"other": panels},
        app=app.model_copy(
            update={
                "pages": (
                    app.pages[0].model_copy(update={"scenarios": (scenario,)}),
                    app.pages[1],
                )
            }
        ),
    )
    monkeypatch.setattr(runner_module, "load_project", lambda reference: with_scenario)
    with pytest.raises(ValueError, match="none of the choices") as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    message = str(error.value)
    for named in ("variant 'other'", "scenario 'mild'", "'renal_plasma'", "0.7"):
        assert named in message


def test_the_derived_targets_are_checked_in_a_variant_no_app_names(
    declared: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    panels = dict(declared.variant_panels["other"])
    extra = panels["extra"]
    panels["extra"] = extra.model_copy(
        update={
            "simulation": extra.simulation.model_copy(
                update={"derived": {"BW": "weight * 1"}}
            )
        }
    )
    app = declared.app
    assert app is not None
    unnamed = declared.replace(
        variant_panels={"other": panels}, app=app.model_copy(update={"variants": ()})
    )
    monkeypatch.setattr(runner_module, "load_project", lambda reference: unnamed)
    with pytest.raises(ValueError, match="no knobs of the panel") as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    for named in ("variant 'other'", "panel 'extra'", "'BW'", "['weight']"):
        assert named in str(error.value)


def test_the_panels_of_every_variant_are_read_once(
    declared: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The startup reads the SBML of each variant once, the cache of `panels_of`.

    The panels and the entities the targets are checked against are of that
    one reading, see `Project.model_entities`.
    """
    fresh = declared.replace()
    reads: list[Path] = []
    original = panel_sbml.read_panels

    def counted(sbml: Path | str) -> Any:
        reads.append(Path(sbml))
        return original(sbml)

    monkeypatch.setattr(panel_sbml, "read_panels", counted)
    monkeypatch.setattr(runner_module, "load_project", lambda reference: fresh)
    ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    assert sorted(path.name for path in reads) == sorted(
        path.name for path in fresh.variants.values()
    )
