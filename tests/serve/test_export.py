"""The static bundle of the panels of a model, `panelbundle/1`.

The bundle carries the bodies the routes of the service answer and the recipe
of the simulation of every panel (`sbmlsim.serve.export`). The tests export
the HCTZ demo of `examples.panels.hctz`, whose page `patient` derives the
renal function from a knob without a target, delivers its x in hours and its
curves in other units than the model, and the device of two variants of
`examples.panels.hctz_variants`, whose second variant derives it otherwise.

The key test simulates the exported SBML with the recipe of the bundle alone,
as a reader of the bundle without the service does, and compares the curves
with the answer of `POST .../simulate`, see `_recipe_curves`.
"""

import itertools
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import pytest

import sbmlsim
from examples.panels.hctz import CRCL, HCTZ_PATIENT_PLASMA, PROJECT
from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel
from sbmlsim.model.model_sbml import SBMLModel, load_model
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.panel import PanelSimulation, panel_from_payload
from sbmlsim.serve import cli
from sbmlsim.serve.export import (
    BUNDLE_FILE,
    CONTRACT,
    KNOB_PREFIX,
    MODELS_DIR,
    SBML_FILE,
    build_bundle,
    export_bundles,
)
from sbmlsim.serve.runner import ModelEntry, ModelRegistry, PanelRunner
from sbmlsim.simulation import Scan, Simulation
from sbmlsim.simulator import Simulator

#: the reference of the HCTZ demo
HCTZ = "examples.panels.hctz:PROJECT"
#: the reference of the device of two variants
VARIANTS = "examples.panels.hctz_variants:PROJECT_VARIANTS"

#: the tolerance of a curve of the recipe against the service, relative to
#: the peak of the curve. Both integrate the same equations with roadrunner and
#: the same settings, while the recipe sets a knob to `v * factor` and scales
#: a curve by its factor where the service converts with pint, and roadrunner
#: computes the derived targets from the MathML where the service computes
#: them with numpy. The curves are bitwise equal on the HCTZ demo (and on the
#: glimepiride model at the optimization level of its project); a difference
#: in the last bit of an input changes the steps of CVODE and moves a curve by
#: the error of the integration at the relative tolerance 1e-10 of the
#: project, e.g. 4e-10 of the peak between the glimepiride model compiled at
#: O0 and at O2, which the bound admits.
RTOL = 1e-9


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The bundle of the HCTZ demo, written as the command line writes it."""
    out = tmp_path_factory.mktemp("bundles")
    export_bundles(ModelRegistry.of({"hctz": HCTZ}), out)
    return out / "hctz"


@pytest.fixture(scope="module")
def hctz_model() -> SBMLModel:
    """The model the service runs the panels of the HCTZ demo on."""
    registry = ModelRegistry.of({"hctz": HCTZ})
    return PanelRunner(registry).model(registry.entry("hctz"))


@pytest.fixture(scope="module")
def bundle(exported: Path) -> dict[str, Any]:
    return json.loads((exported / BUNDLE_FILE).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exported_variants(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The bundle of the device of two variants."""
    out = tmp_path_factory.mktemp("variants")
    export_bundles(ModelRegistry.of({"hctz": VARIANTS}), out)
    return out / "hctz"


@pytest.fixture(scope="module")
def bundle_variants(exported_variants: Path) -> dict[str, Any]:
    return json.loads((exported_variants / BUNDLE_FILE).read_text(encoding="utf-8"))


def test_the_bundle_has_its_contract(bundle: dict[str, Any], exported: Path) -> None:
    assert bundle["contract"] == CONTRACT
    assert bundle["model"] == "hctz"
    assert bundle["title"] == (PROJECT.title or PROJECT.name)
    assert bundle["sbmlsim"] == sbmlsim.__version__
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", bundle["created"])
    datetime.fromisoformat(bundle["created"])
    # no query and the default variant, the one variant of the app
    assert list(bundle["answers"]) == ["", "default"]
    for answer in bundle["answers"].values():
        assert answer["app"]["contract"] == "app/1"
        assert list(answer["panels"]) == list(PROJECT.panels_of())
    for key, model in bundle["models"].items():
        assert model["sbml"] == f"{MODELS_DIR}/{key}/{SBML_FILE}"
        assert model["compiled"] == f"{MODELS_DIR}/{key}"
        assert (exported / model["sbml"]).is_file()
        assert model["variant"] == "default"
    assert sorted(path.name for path in (exported / MODELS_DIR).iterdir()) == sorted(
        bundle["models"]
    )


def test_a_panel_without_derived_targets_runs_on_its_variant(
    bundle: dict[str, Any], exported: Path, hctz_model: SBMLModel
) -> None:
    run = bundle["answers"][""]["panels"]["renal_plasma"]["run"]
    assert run["model"] == "default"
    assert run["variant"] == "default"
    sbml = (exported / MODELS_DIR / "default" / SBML_FILE).read_text(encoding="utf-8")
    assert sbml == hctz_model.sbml
    assert run["knobs"] == {
        "egfr": {
            "target": "f_renal_function",
            "factor": 1.0,
            "parameter": None,
            "unit": "dimensionless",
        },
        "dose": {
            "target": "PODOSE_hctz",
            "factor": 1.0,
            "parameter": None,
            "unit": "mg",
        },
    }
    assert run["curves"] == {
        "[Cve_hctz]": {"selection": "[Cve_hctz]", "factor": pytest.approx(1000.0)},
        "[Cki_plasma_hctz]": {
            "selection": "[Cki_plasma_hctz]",
            "factor": pytest.approx(1000.0),
        },
    }
    assert run["units"] == {"x": "min", "[Cve_hctz]": "uM", "[Cki_plasma_hctz]": "uM"}
    assert run["preinit"] == [["BW", 75.0], ["PODOSE_hctz", 0.0]]
    assert (run["start"], run["end"], run["points"]) == (0.0, 1440.0, 50)
    assert len(run["x"]) == run["points"] + 1


def test_the_panels_of_one_derived_simulation_share_a_model(
    bundle: dict[str, Any], exported: Path
) -> None:
    """The page `patient` derives the renal function from the clearance."""
    panels = bundle["answers"][""]["panels"]
    plasma = panels["patient_plasma"]["run"]
    urine = panels["patient_urine"]["run"]
    assert plasma["model"] == urine["model"]
    key = plasma["model"]
    assert re.fullmatch(r"default-[0-9a-f]{12}", key)
    parameter = f"{KNOB_PREFIX}crcl"
    assert plasma["knobs"]["crcl"] == {
        "target": None,
        "factor": None,
        "parameter": parameter,
        "unit": "ml/min",
    }
    assert plasma["units"]["x"] == "hr"
    assert bundle["models"][key]["selections"] == sorted(
        {"[Cve_hctz]", "[Cki_plasma_hctz]", "Aurine_hctz"}
    )
    doc = libsbml.readSBMLFromFile(str(exported / MODELS_DIR / key / SBML_FILE))
    model = doc.getModel()
    knob = model.getParameter(parameter)
    assert knob.getConstant()
    assert knob.getValue() == CRCL.default
    assignment = model.getInitialAssignmentBySymbol("f_renal_function")
    assert libsbml.formulaToL3String(assignment.getMath()) == (
        f"piecewise({parameter} / 110, ({parameter} / 110) <= 1, 1)"
    )


def test_the_settings_and_tolerances_are_the_ones_of_roadrunner(
    bundle: dict[str, Any], hctz_model: SBMLModel
) -> None:
    assert isinstance(hctz_model, RoadrunnerSBMLModel)
    frame = hctz_model.tolerances()
    expected = dict(zip(frame["sid"], frame["absolute_tolerance"], strict=True))
    for one in bundle["models"].values():
        assert one["absolute_tolerances"] == expected
        assert one["settings"] == {
            "relative_tolerance": 1e-10,
            "absolute_tolerance": {"scalar": 1e-10},
            "maximum_num_steps": 20000,
            "maximum_time_step": 0.0,
            "initial_time_step": 0.0,
            "stiff": True,
            "linear_solver": "auto",
            "jacobian": "auto",
        }


def test_the_settings_of_a_project_are_carried(bundle: dict[str, Any]) -> None:
    project = PROJECT.replace(
        simulator=PROJECT.simulator.model_copy(
            update={
                "relative_tolerance": 1e-8,
                "absolute_tolerance": 1e-12,
                "integrator": {"maximum_num_steps": 5000, "stiff": False},
            }
        )
    )
    registry = ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=project, title="t")}
    )
    content = build_bundle(registry, "hctz").content
    settings = content["models"]["default"]["settings"]
    assert settings["relative_tolerance"] == 1e-8
    assert settings["absolute_tolerance"] == {"scalar": 1e-12}
    assert settings["maximum_num_steps"] == 5000
    assert settings["stiff"] is False
    # the tolerance of every state is the one of the project, a concentration
    # species scaled by the volume of its compartment
    default = bundle["models"]["default"]["absolute_tolerances"]
    tolerances = content["models"]["default"]["absolute_tolerances"]
    assert tolerances == {
        sid: pytest.approx(value * 1e-2, rel=1e-12) for sid, value in default.items()
    }


def test_the_variants_of_an_app_are_answered(bundle_variants: dict[str, Any]) -> None:
    answers = bundle_variants["answers"]
    assert list(answers) == ["", "default", "low_reference"]
    assert answers["low_reference"]["app"]["variant"] == "low_reference"
    assert answers[""]["app"]["variant"] == "default"
    default = answers["default"]["panels"]["patient_plasma"]["run"]
    low = answers["low_reference"]["panels"]["patient_plasma"]["run"]
    assert low["variant"] == "low_reference"
    assert low["model"].startswith("low_reference-")
    assert default["model"].startswith("default-")
    assert low["model_sha256"] == default["model_sha256"]
    models = bundle_variants["models"]
    assert models[low["model"]]["variant"] == "low_reference"
    assert answers[""]["panels"]["patient_plasma"]["run"]["model"] == default["model"]


def test_a_derived_target_which_is_no_parameter_is_refused() -> None:
    panel = HCTZ_PATIENT_PLASMA.model_copy(
        update={
            "simulation": HCTZ_PATIENT_PLASMA.simulation.model_copy(
                update={"derived": {"hctz": "crcl / 110"}}
            )
        }
    )
    _refused(panel.id, panel, "'hctz' of the panel 'patient_plasma' is no parameter")


def test_a_derived_target_which_a_rule_sets_is_refused() -> None:
    panel = HCTZ_PATIENT_PLASMA.model_copy(
        update={
            "simulation": HCTZ_PATIENT_PLASMA.simulation.model_copy(
                update={"derived": {"GFR": "crcl / 110"}}
            )
        }
    )
    _refused(panel.id, panel, "'GFR' of the panel 'patient_plasma' is set by a rule")


def _refused(panel_id: str, panel: Any, message: str) -> None:
    project = PROJECT.replace(panels={panel_id: panel}, app=None)
    registry = ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=project, title="t")}
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        build_bundle(registry, "hctz")


def test_a_parameter_of_a_knob_does_not_clash_with_the_model(
    tmp_path: Path, hctz_model: SBMLModel
) -> None:
    sbml = tmp_path / "model.xml"
    doc = libsbml.readSBMLFromString(hctz_model.sbml)
    taken = doc.getModel().createParameter()
    taken.setId(f"{KNOB_PREFIX}crcl")
    taken.setValue(1.0)
    taken.setConstant(True)
    sbml.write_text(libsbml.writeSBMLToString(doc), encoding="utf-8")
    project = PROJECT.replace(
        variants={"default": sbml},
        panels={"patient_plasma": HCTZ_PATIENT_PLASMA},
        app=None,
    )
    registry = ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=project, title="t")}
    )
    run = build_bundle(registry, "hctz").content["answers"][""]["panels"][
        "patient_plasma"
    ]["run"]
    assert run["knobs"]["crcl"]["parameter"] == f"{KNOB_PREFIX}crcl_2"


def test_the_command_line_exports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "bundles"
    assert cli.main(["--model", f"hctz={HCTZ}", "--export", str(out)]) == cli.EXIT_OK
    assert capsys.readouterr().out == f"{out / 'hctz'}\n"
    assert (out / "hctz" / BUNDLE_FILE).is_file()


def test_the_command_line_refuses_a_model_which_is_no_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["--model", "hctz=sbmlsim:Q", "--export", str(tmp_path)])
    assert code == cli.EXIT_CONFIGURATION
    assert "not exported" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def _client(reference: str) -> Any:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sbmlsim.serve.app import create_app

    return TestClient(create_app(ModelRegistry.of({"hctz": reference})))


@pytest.mark.parametrize("reference", [HCTZ, VARIANTS])
def test_the_bodies_are_the_ones_the_service_answers(
    reference: str,
    bundle: dict[str, Any],
    bundle_variants: dict[str, Any],
) -> None:
    content = bundle if reference == HCTZ else bundle_variants
    with _client(reference) as client:
        for query, answer in content["answers"].items():
            params = {"variant": query} if query else {}
            app = client.get("/api/models/hctz/app", params=params)
            assert app.status_code == 200
            assert app.json() == answer["app"]
            for panel_id, panel in answer["panels"].items():
                path = f"/api/models/hctz/panels/{panel_id}"
                assert client.get(path, params=params).json() == panel["declaration"]
                assert client.get(f"{path}/data", params=params).json() == panel["data"]


#: the requests of the simulations the recipe reproduces: a grid over a range
#: knob of the derived renal function, a point of it in each variant, and a
#: grid over two choice knobs of a panel without derived targets
REQUESTS = [
    (HCTZ, "", "patient_plasma", {"dose": 12.5}, {"crcl": 3}),
    (HCTZ, "", "patient_urine", {"crcl": 45.0, "dose": 50.0}, {}),
    (HCTZ, "default", "renal_plasma", {}, {"egfr": 3, "dose": 4}),
    (VARIANTS, "low_reference", "patient_plasma", {"crcl": 75.0}, {}),
]


@pytest.mark.parametrize(("reference", "query", "panel_id", "values", "grid"), REQUESTS)
def test_the_recipe_reproduces_the_simulation_of_the_service(
    reference: str,
    query: str,
    panel_id: str,
    values: dict[str, float],
    grid: dict[str, int],
    exported: Path,
    exported_variants: Path,
) -> None:
    directory = exported if reference == HCTZ else exported_variants
    content = json.loads((directory / BUNDLE_FILE).read_text(encoding="utf-8"))
    request: dict[str, Any] = {"values": values, "grid": grid}
    if query:
        request["variant"] = query
    with _client(reference) as client:
        response = client.post(
            f"/api/models/hctz/panels/{panel_id}/simulate", json=request
        )
    assert response.status_code == 200
    expected = response.json()
    run = content["answers"][query]["panels"][panel_id]["run"]
    assert run["x"] == expected["x"]
    assert run["units"] == expected["units"]
    assert run["variant"] == expected["provenance"]["variant"]
    assert run["model_sha256"] == expected["provenance"]["model_sha256"]
    for dim in expected["dims"]:
        assert dim["unit"] == run["knobs"][dim["id"]]["unit"]
    curves = _recipe_curves(directory, content, query, panel_id, values, grid)
    assert expected["status"]["failed"] == []
    for curve_id, rows in curves.items():
        served = np.asarray(expected["curves"][curve_id], dtype=float)
        assert rows.shape == served.shape
        peak = np.max(np.abs(served))
        np.testing.assert_allclose(rows, served, rtol=RTOL, atol=RTOL * peak)


def _recipe_curves(
    directory: Path,
    content: Mapping[str, Any],
    query: str,
    panel_id: str,
    values: Mapping[str, float],
    grid: Mapping[str, int],
) -> dict[str, np.ndarray]:
    """Simulate a request with the recipe of the bundle alone.

    The points of a grid are the product of `Knob.values(n)` of its knobs in
    C order, every other knob takes its value or its default; a knob sets its
    target to `v * factor` and its parameter to `v`, over the `preinit` of the
    run; the exported SBML runs with the settings and the tolerance of every
    state of its model, on the output times of the run, and a curve is its
    selection times its factor.
    """
    answer = content["answers"][query]["panels"][panel_id]
    run = answer["run"]
    model_of = content["models"][run["model"]]
    panel = panel_from_payload(answer["declaration"])
    gridded = [knob for knob in panel.knobs if knob.id in grid]
    points = list(itertools.product(*(k.values(grid[k.id]) for k in gridded)))
    settings = model_of["settings"]
    scalar = settings["absolute_tolerance"]["scalar"]
    simulator = Simulator(
        n_workers=1,
        relative_tolerance=settings["relative_tolerance"],
        absolute_tolerance=AbsoluteTolerance(
            amount=scalar,
            concentration=scalar,
            other=scalar,
            ids=model_of["absolute_tolerances"],
        ),
        maximum_num_steps=settings["maximum_num_steps"],
        maximum_time_step=settings["maximum_time_step"],
        initial_time_step=settings["initial_time_step"],
        stiff=settings["stiff"],
    )
    model = simulator.load(load_model(directory / model_of["sbml"], "roadrunner"))
    assert isinstance(model, RoadrunnerSBMLModel)
    frame = model.tolerances()
    assert (
        dict(zip(frame["sid"], frame["absolute_tolerance"], strict=True))
        == (model_of["absolute_tolerances"])
    )
    times = np.linspace(run["start"], run["end"], run["points"] + 1)
    rows: dict[str, list[np.ndarray]] = {curve: [] for curve in run["curves"]}
    for point in points:
        state = {
            knob.id: float(values.get(knob.id, knob.default)) for knob in panel.knobs
        }
        state.update({k.id: v for k, v in zip(gridded, point, strict=True)})
        preinit: dict[str, float] = dict(run["preinit"])
        for knob_id, value in state.items():
            recipe = run["knobs"][knob_id]
            if recipe["target"] is not None:
                preinit[recipe["target"]] = value * recipe["factor"]
            if recipe["parameter"] is not None:
                preinit[recipe["parameter"]] = value
        simulation = Simulation(
            end=run["end"],
            preinit_changes=dict(sorted(preinit.items())),
            times=times.tolist(),
        )
        result = simulator.run(
            model,
            Scan(simulation),
            time=times,
            keep=model_of["selections"],
            progress=False,
        )
        for curve_id, curve in run["curves"].items():
            column = np.asarray(result.ds[curve["selection"]].values, dtype=float)
            rows[curve_id].append(column.reshape(-1) * curve["factor"])
    return {curve: np.vstack(one) for curve, one in rows.items()}


def test_a_panel_simulation_of_the_bundle_is_the_declared_one() -> None:
    """The run carries the end and points of the declared simulation."""
    simulation: PanelSimulation = HCTZ_PATIENT_PLASMA.simulation
    registry = ModelRegistry.of({"hctz": HCTZ})
    run = build_bundle(registry, "hctz").content["answers"][""]["panels"][
        "patient_plasma"
    ]["run"]
    assert run["end"] == simulation.end
    assert run["points"] == simulation.points
    assert run["x"][-1] == pytest.approx(simulation.end / 60.0)


def test_a_model_id_which_is_no_directory_is_refused(tmp_path: Path) -> None:
    registry = ModelRegistry({"a/b": ModelEntry(id="a/b", project=PROJECT, title="t")})
    with pytest.raises(ValueError, match=re.escape("'a/b' is no name of a directory")):
        export_bundles(registry, tmp_path)
