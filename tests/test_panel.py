"""The panel declaration of a model and its `panel/1` contract."""

import ast
import json
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sbmlsim.panel import (
    CONTRACT,
    MAX_CURVES,
    MAX_END,
    MAX_GRID,
    MAX_KNOBS,
    MAX_POINTS,
    Choice,
    ContractError,
    Knob,
    Panel,
    PanelCurve,
    PanelData,
    PanelSimulation,
    panel_from_payload,
    panel_payload,
)

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
    choices=(Choice(label="12.5 mg", value=12.5), Choice(label="25 mg", value=25.0)),
    default=25.0,
)


@pytest.fixture
def panel() -> Panel:
    return Panel(
        id="renal",
        title="Hydrochlorothiazide in renal impairment",
        subtitle="the plasma concentration over the renal function",
        x_label="time (hr)",
        knobs=(EGFR, DOSE),
        curves=(
            PanelCurve(
                id="[Cve_hctz]",
                label="hydrochlorothiazide plasma",
                unit="mmol/l",
                scale="log",
            ),
        ),
        data=(
            PanelData(
                id="beermann1976",
                label="Beermann 1976",
                curve="[Cve_hctz]",
                dataset="Beermann1976_Fig2",
                x="time",
                y="mean",
                yerr="mean_sd",
                sel={"label": "hctz5po_4"},
                conversion="1/(297.7 g/mole)",
                unit="mmol/l",
            ),
        ),
        simulation=PanelSimulation(
            variant="default",
            end=24.0,
            points=200,
            x_unit="hr",
            preinit={"BW": 75.0},
        ),
    )


@pytest.fixture
def minimal() -> Panel:
    """A panel of the required fields only, so the defaults round trip too."""
    return Panel(
        id="decay",
        title="decay",
        curves=(PanelCurve(id="[S1]", label="S1"),),
        simulation=PanelSimulation(end=10.0),
    )


def test_range_knob_values_are_its_grid() -> None:
    knob = Knob(
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
    assert knob.values(None) == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert knob.values(3) == (0.2, 0.6, 1.0)


def test_a_grid_of_one_point_is_the_lower_bound() -> None:
    assert EGFR.values(1) == (0.2,)


def test_choice_knob_values_are_its_choices() -> None:
    knob = Knob(
        id="dose",
        label="dose",
        target="PODOSE_hctz",
        unit="mg",
        kind="choice",
        choices=(
            Choice(label="12.5 mg", value=12.5),
            Choice(label="25 mg", value=25.0),
        ),
        default=25.0,
    )
    assert knob.values(None) == (12.5, 25.0)
    assert knob.values(8) == (12.5, 25.0)


def test_a_range_knob_without_bounds_is_refused() -> None:
    with pytest.raises(ValidationError, match="bounds"):
        Knob(id="egfr", label="x", target="t", kind="range", default=1.0)


def test_a_knob_default_outside_its_range_is_refused() -> None:
    with pytest.raises(ValidationError, match="default"):
        Knob(
            id="egfr",
            label="x",
            target="t",
            kind="range",
            min=0.2,
            max=1.0,
            default=2.0,
        )


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            lambda: Knob(id="k", label="x", target="t", kind="choice", default=1.0),
            "choices",
        ),
        (
            lambda: Knob(
                id="k",
                label="x",
                target="t",
                kind="choice",
                choices=(Choice(label="one", value=1.0),),
                default=2.0,
            ),
            "default",
        ),
        (
            lambda: Knob(id="k", label="x", target="t", min=1.0, max=1.0, default=1.0),
            "min < max",
        ),
    ],
)
def test_an_incoherent_knob_is_refused(build: Callable[[], Knob], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        build()


@pytest.mark.parametrize(
    "dataset",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "data/Beermann1976_Fig2",
        "data\\Beermann1976_Fig2",
        "Beermann1976_Fig2.tsv",
        ".Beermann1976_Fig2",
        "..",
        ".",
        "",
        "Beermann 1976",
    ],
)
def test_a_dataset_which_is_no_dataset_id_is_refused(dataset: str) -> None:
    """A dataset is the id of a figure or a table and never a path."""
    with pytest.raises(ValidationError, match="no dataset"):
        PanelData(id="d", label="d", dataset=dataset, x="time", y="mean")


def test_the_curve_of_a_series_is_its_own_id_without_one() -> None:
    series = PanelData(id="[S1]", label="S1", dataset="Study2000_Fig1", x="t", y="mean")
    assert series.curve is None
    assert series.curve_id == "[S1]"
    assert series.model_copy(update={"curve": "[S2]"}).curve_id == "[S2]"


def test_two_series_of_one_curve_are_accepted_and_of_one_id_are_not() -> None:
    """A panel shows two studies under one curve, with distinct ids."""
    first = PanelData(
        id="study2000",
        label="Study 2000",
        curve="[S1]",
        dataset="Study2000_Fig1",
        x="t",
        y="mean",
    )
    second = first.model_copy(update={"id": "study2010", "label": "Study 2010"})
    panel = Panel(
        id="p",
        title="p",
        curves=(PanelCurve(id="[S1]", label="S1"),),
        data=(first, second),
        simulation=PanelSimulation(end=1.0),
    )
    assert [series.curve_id for series in panel.data] == ["[S1]", "[S1]"]
    with pytest.raises(ValidationError, match="study2000"):
        Panel(
            id="p",
            title="p",
            curves=(PanelCurve(id="[S1]", label="S1"),),
            data=(first, first),
            simulation=PanelSimulation(end=1.0),
        )


def test_two_knobs_of_one_id_are_refused() -> None:
    with pytest.raises(ValidationError, match="egfr"):
        Panel(
            id="p",
            title="p",
            knobs=(EGFR, EGFR),
            curves=(PanelCurve(id="[S1]", label="S1"),),
            simulation=PanelSimulation(end=1.0),
        )


def test_two_curves_of_one_id_are_refused() -> None:
    curve = PanelCurve(id="[S1]", label="S1")
    with pytest.raises(ValidationError, match=r"\[S1\]"):
        Panel(
            id="p",
            title="p",
            curves=(curve, curve),
            simulation=PanelSimulation(end=1.0),
        )


@pytest.mark.parametrize(
    "simulation",
    [
        {"end": 1e9},
        {"end": MAX_END + 1.0},
        {"end": 0.0},
        {"end": 1.0, "points": 10**6},
        {"end": 1.0, "points": MAX_POINTS + 1},
    ],
)
def test_a_simulation_over_the_limits_is_refused(simulation: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PanelSimulation(**simulation)


def test_a_simulation_at_the_limits_is_accepted() -> None:
    simulation = PanelSimulation(end=MAX_END, points=MAX_POINTS)
    assert (simulation.end, simulation.points) == (MAX_END, MAX_POINTS)


def test_the_time_axis_is_the_unit_of_the_model_without_an_x_unit() -> None:
    """`x_unit` is the unit the time axis of a result is delivered in."""
    assert PanelSimulation(end=10.0).x_unit is None
    # the end and the points stay in the time unit of the model, minutes here
    declared = PanelSimulation(end=1440.0, x_unit="hr")
    assert (declared.end, declared.x_unit) == (1440.0, "hr")
    panel = Panel(
        id="p",
        title="p",
        curves=(PanelCurve(id="[S1]", label="S1"),),
        simulation=declared,
    )
    assert panel_payload(panel)["simulation"]["x_unit"] == "hr"


def test_more_curves_than_the_limit_are_refused() -> None:
    curves = tuple(
        PanelCurve(id=f"[S{i}]", label=f"S{i}") for i in range(MAX_CURVES + 1)
    )
    with pytest.raises(ValidationError):
        Panel(id="p", title="p", curves=curves, simulation=PanelSimulation(end=1.0))


def test_a_panel_is_frozen_and_forbids_unknown_fields(panel: Panel) -> None:
    with pytest.raises(ValidationError):
        panel.title = "another title"
    with pytest.raises(ValidationError, match="legend"):
        Panel(
            id="p",
            title="p",
            curves=(PanelCurve(id="[S1]", label="S1"),),
            simulation=PanelSimulation(end=1.0),
            legend=True,
        )


def test_the_contract_of_a_panel_is_its_version() -> None:
    assert CONTRACT == "panel/1"


def test_the_payload_round_trips_a_panel_exactly(panel: Panel) -> None:
    payload = panel_payload(panel)
    assert payload["contract"] == "panel/1"
    assert panel_from_payload(payload) == panel


def test_the_payload_round_trips_a_minimal_panel_exactly(minimal: Panel) -> None:
    assert panel_from_payload(panel_payload(minimal)) == minimal


def test_a_payload_of_the_declared_keys_only_is_read(minimal: Panel) -> None:
    """The reader of the annotation builds its payload itself (Task 2)."""
    assert (
        panel_from_payload(
            {
                "contract": CONTRACT,
                "panel": {"id": "decay", "title": "decay"},
                "curves": [{"id": "[S1]", "label": "S1"}],
                "simulation": {"end": 10.0},
            }
        )
        == minimal
    )


def test_a_panel_is_picklable(panel: Panel) -> None:
    assert pickle.loads(pickle.dumps(panel)) == panel


def test_the_payload_names_the_knobs_as_the_axes_of_the_grid(panel: Panel) -> None:
    payload = panel_payload(panel)
    assert [axis["id"] for axis in payload["axes"]] == ["egfr", "dose"]
    assert payload["panel"] == {
        "id": "renal",
        "title": panel.title,
        "subtitle": panel.subtitle,
        "x_label": panel.x_label,
    }


def test_the_payload_carries_the_limits_of_a_request(panel: Panel) -> None:
    assert panel_payload(panel)["limits"] == {
        "end": MAX_END,
        "points": MAX_POINTS,
        "knobs": MAX_KNOBS,
        "grid": MAX_GRID,
        "curves": MAX_CURVES,
    }


def test_a_payload_of_another_contract_version_is_refused() -> None:
    with pytest.raises(ContractError, match="panel/2"):
        panel_from_payload({"contract": "panel/2", "panel": {}})


def test_a_payload_without_a_contract_is_refused() -> None:
    with pytest.raises(ContractError):
        panel_from_payload({"panel": {}})


def test_an_unknown_field_of_a_payload_is_refused(panel: Panel) -> None:
    payload = panel_payload(panel)
    payload["axes"][0]["unknown"] = 1
    with pytest.raises(ValidationError, match="unknown"):
        panel_from_payload(payload)


def test_the_payload_is_json_and_carries_no_python_type(panel: Panel) -> None:
    payload = panel_payload(panel)
    assert json.loads(json.dumps(payload)) == payload
    for value in _leaves(payload):
        assert value is None or isinstance(value, (str, int, float, bool)), value


def _leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _leaves(item)]
    return [value]


def test_the_module_imports_pydantic_and_the_standard_library_only() -> None:
    """`sbmlsim.project` imports it, so it imports no simulator and no libsbml."""
    path = Path(__file__).parents[1] / "src" / "sbmlsim" / "panel.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {*sys.stdlib_module_names, "__future__", "pydantic"}


def test_a_project_declares_the_panels_of_a_model_without_an_annotation(
    tmp_path: Path, panel: Panel
) -> None:
    from sbmlsim.project import Project

    project = Project(name="toy", root=tmp_path, panels={panel.id: panel})
    assert project.panels == {"renal": panel}
    assert Project(name="toy", root=tmp_path).panels == {}
