"""Tests of the panels of the readouts sidecar of a figure."""

import json
from pathlib import Path

import pytest

from sbmlsim.panel import ContractError, PanelSimulation
from sbmlsim.panel_figure import panels_from_figure

DATA = Path(__file__).parent / "data" / "figure"
SVG = DATA / "model.svg"
READOUTS = DATA / "model.readouts.json"
OTHER = DATA / "other.readouts.json"
SIMULATION = PanelSimulation(end=10.0)


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


def test_readouts_of_mixed_kinds_leave_the_y_label_empty() -> None:
    panels, _ = panels_from_figure(
        SVG, READOUTS, simulation=SIMULATION, title="T", prefix="m_"
    )
    assert panels["m_urine"].y_label == ""
    assert "urine" not in panels


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


def test_a_sidecar_of_another_contract_is_refused() -> None:
    with pytest.raises(ContractError, match="figurereadouts/1"):
        panels_from_figure(SVG, OTHER, simulation=SIMULATION, title="T")


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
    with pytest.raises(ContractError, match="16"):
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
