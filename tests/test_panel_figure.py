"""Tests of the panels of the readouts sidecar of a figure."""

import json
import re
from pathlib import Path

import pytest

from sbmlsim.panel import ContractError, PanelSimulation
from sbmlsim.panel_figure import panels_from_figure

DATA = Path(__file__).parent / "data" / "figure"
SVG = DATA / "model.svg"
READOUTS = DATA / "model.readouts.json"
SIMULATION = PanelSimulation(end=10.0)


def _readout(
    id: str, *, unit: str = "mM", kind: str = "concentration", label: str = "L"
) -> dict[str, str]:
    return {"id": id, "label": label, "unit": unit, "kind": kind}


def _sidecar(tmp_path: Path, keys: dict[str, object]) -> Path:
    path = tmp_path / "x.readouts.json"
    path.write_text(
        json.dumps({"contract": "figurereadouts/1", "model": "m", "keys": keys})
    )
    return path


def test_a_group_of_the_sidecar_becomes_a_panel_of_its_readouts() -> None:
    panels, figure = panels_from_figure(
        SVG, READOUTS, simulation=SIMULATION, title="The model"
    )
    panel = panels["li_gli"]
    assert [c.id for c in panel.curves] == ["LI__Cpl_gli", "LI__Cis_gli", "LI__Cic_gli"]
    assert panel.title == "Liver, glimepiride"
    assert panel.y_label == "concentration"
    assert panel.simulation == SIMULATION
    assert figure.panels["li/sub:gli"] == ("li_gli",)
    assert figure.panels["li"] == ("li",)


def test_a_key_of_two_units_opens_one_panel_per_unit() -> None:
    panels, figure = panels_from_figure(
        SVG, READOUTS, simulation=SIMULATION, title="T", prefix="m_"
    )
    assert figure.panels["urine"] == ("m_urine_amount", "m_urine_rate")
    for pid, unit, kind in (
        ("m_urine_amount", "mmol", "amount"),
        ("m_urine_rate", "mmol/h", "rate"),
    ):
        panel = panels[pid]
        assert [c.unit for c in panel.curves] == [unit]
        assert panel.y_label == kind
    assert "m_urine" not in panels


def test_panels_of_one_kind_are_told_apart_by_an_index(tmp_path: Path) -> None:
    path = _sidecar(
        tmp_path,
        {
            "x": {
                "label": "X",
                "readouts": [_readout("a", unit="mM"), _readout("b", unit="uM")],
            }
        },
    )
    panels, figure = panels_from_figure(SVG, path, simulation=SIMULATION, title="T")
    assert figure.panels["x"] == ("x_concentration_1", "x_concentration_2")
    assert set(panels) == set(figure.panels["x"])


def test_kinds_of_one_unit_are_joined_in_the_y_label(tmp_path: Path) -> None:
    path = _sidecar(
        tmp_path,
        {
            "x": {
                "label": "X",
                "readouts": [_readout("a", kind="rate"), _readout("b", kind="flux")],
            }
        },
    )
    panels, _ = panels_from_figure(SVG, path, simulation=SIMULATION, title="T")
    assert panels["x"].y_label == "rate / flux"


def test_an_empty_unit_becomes_none(tmp_path: Path) -> None:
    path = _sidecar(
        tmp_path, {"x": {"label": "X", "readouts": [_readout("a", unit="")]}}
    )
    panels, _ = panels_from_figure(SVG, path, simulation=SIMULATION, title="T")
    assert panels["x"].curves[0].unit is None


def test_an_empty_keys_mapping_gives_no_panels(tmp_path: Path) -> None:
    path = _sidecar(tmp_path, {})
    panels, figure = panels_from_figure(SVG, path, simulation=SIMULATION, title="T")
    assert panels == {}
    assert figure.panels == {}


def test_a_payload_which_is_no_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "list.readouts.json"
    path.write_text("[1, 2]")
    with pytest.raises(ContractError, match="figurereadouts/1"):
        panels_from_figure(SVG, path, simulation=SIMULATION, title="T")


def test_a_payload_without_keys_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "nokeys.readouts.json"
    path.write_text(json.dumps({"contract": "figurereadouts/1", "model": "m"}))
    with pytest.raises(ContractError, match="keys"):
        panels_from_figure(SVG, path, simulation=SIMULATION, title="T")


@pytest.mark.parametrize(
    ("group", "match"),
    [
        ("text", "not an object"),
        ({"readouts": []}, "no label"),
        ({"label": " ", "readouts": []}, "no label"),
        ({"label": "X", "readouts": "a"}, "list of readouts"),
        ({"label": "X", "readouts": ["a"]}, "not an object"),
        ({"label": "X", "readouts": [{"id": "a", "label": "a"}]}, "'kind'"),
        ({"label": "X", "readouts": [{"label": "a", "kind": "rate"}]}, "'id'"),
        ({"label": "X", "readouts": [_readout("a", label=" ")]}, "'label'"),
        ({"label": "X", "readouts": [_readout("a"), _readout("a")]}, "twice"),
    ],
)
def test_a_malformed_group_is_refused_naming_its_key(
    tmp_path: Path, group: object, match: str
) -> None:
    path = _sidecar(tmp_path, {"bad_key": group})
    with pytest.raises(ContractError, match="'bad_key'") as raised:
        panels_from_figure(SVG, path, simulation=SIMULATION, title="T")
    assert re.search(match, str(raised.value))


def test_a_decoration_selects_nothing() -> None:
    panels, figure = panels_from_figure(
        SVG, READOUTS, simulation=SIMULATION, title="The model"
    )
    assert figure.panels["bladder"] == ()
    assert "bladder" not in panels


def test_the_figure_carries_the_svg_inline() -> None:
    _, figure = panels_from_figure(
        SVG, READOUTS, simulation=SIMULATION, title="T", credit="Somebody"
    )
    assert figure.svg == SVG.read_text(encoding="utf-8")
    assert figure.title == "T"
    assert figure.credit == "Somebody"


def test_a_sidecar_of_another_contract_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.readouts.json"
    other.write_text(
        READOUTS.read_text().replace("figurereadouts/1", "figurereadouts/2")
    )
    with pytest.raises(ContractError, match="figurereadouts/1"):
        panels_from_figure(SVG, other, simulation=SIMULATION, title="T")


def test_a_group_of_more_than_sixteen_curves_is_refused(tmp_path: Path) -> None:
    readouts = [
        {"id": f"R{i}", "label": f"R{i}", "unit": "mM", "kind": "concentration"}
        for i in range(17)
    ]
    path = tmp_path / "big.readouts.json"
    path.write_text(
        json.dumps(
            {
                "contract": "figurereadouts/1",
                "model": "m",
                "keys": {"li": {"label": "Liver", "readouts": readouts}},
            }
        )
    )
    with pytest.raises(ContractError, match="'li'"):
        panels_from_figure(SVG, path, simulation=SIMULATION, title="T")


def test_two_keys_of_one_panel_id_are_refused(tmp_path: Path) -> None:
    entry = {
        "label": "X",
        "readouts": [{"id": "a", "label": "a", "unit": "", "kind": "amount"}],
    }
    path = tmp_path / "twice.readouts.json"
    path.write_text(
        json.dumps(
            {
                "contract": "figurereadouts/1",
                "model": "m",
                "keys": {"li/sub:gli": entry, "li/gli": entry},
            }
        )
    )
    with pytest.raises(ContractError, match="twice"):
        panels_from_figure(SVG, path, simulation=SIMULATION, title="T")
