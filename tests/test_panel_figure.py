"""Tests of the panels of the readouts sidecar of a figure."""

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from sbmlsim.panel import (
    MAX_CURVES,
    AppFigure,
    ContractError,
    Knob,
    Panel,
    PanelSimulation,
)
from sbmlsim.panel_figure import panels_from_figure, restrict_to_model

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
        json.dumps({"contract": "figurereadouts/1", "model": "m", "keys": keys}),
        encoding="utf-8",
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
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ContractError, match="figurereadouts/1"):
        panels_from_figure(SVG, path, simulation=SIMULATION, title="T")


def test_a_payload_without_keys_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "nokeys.readouts.json"
    path.write_text(
        json.dumps({"contract": "figurereadouts/1", "model": "m"}), encoding="utf-8"
    )
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
        READOUTS.read_text(encoding="utf-8").replace(
            "figurereadouts/1", "figurereadouts/2"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="figurereadouts/1"):
        panels_from_figure(SVG, other, simulation=SIMULATION, title="T")


def test_more_readouts_of_one_unit_than_the_limit_become_several_panels(
    tmp_path: Path,
) -> None:
    """A key of more readouts than a panel draws is split and not refused (#591).

    The limit stays, 16 curves in one plot being more than a reader separates
    by colour, and every readout of the drawing stays reachable.
    """
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
        ),
        encoding="utf-8",
    )

    panels, figure = panels_from_figure(SVG, path, simulation=SIMULATION, title="T")

    assert figure.panels["li"] == ("li_1", "li_2")
    assert [len(panels[pid].curves) for pid in ("li_1", "li_2")] == [16, 1]
    assert panels["li_1"].title == "Liver (1 of 2)"
    assert panels["li_2"].title == "Liver (2 of 2)"
    # the curves are the readouts of the sidecar in its order, each once
    drawn = [curve.id for pid in ("li_1", "li_2") for curve in panels[pid].curves]
    assert drawn == [entry["id"] for entry in readouts]
    assert figure.title == "T"


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
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="twice"):
        panels_from_figure(SVG, path, simulation=SIMULATION, title="T")


def _grouped(key: str, readout: Mapping[str, str]) -> tuple[str, str]:
    substance, _, compartment = readout["label"].partition(", ")
    return compartment, substance


def test_a_group_opens_one_panel_per_compartment(tmp_path: Path) -> None:
    path = _sidecar(
        tmp_path,
        {
            "li": {
                "label": "Liver",
                "readouts": [
                    _readout("LI__Cpl_gli", label="glimepiride, plasma"),
                    _readout("LI__Cpl_m1", label="M1, plasma"),
                    _readout("LI__Cic_gli", label="glimepiride, cells"),
                ],
            }
        },
    )
    panels, figure = panels_from_figure(
        SVG, path, simulation=SIMULATION, title="T", group=_grouped
    )
    assert list(panels) == ["li_plasma", "li_cells"]
    assert [c.label for c in panels["li_plasma"].curves] == ["glimepiride", "M1"]
    assert panels["li_plasma"].title == "Liver, plasma"
    assert figure.panels["li"] == ("li_plasma", "li_cells")


def test_a_key_of_one_group_keeps_the_id_of_its_key(tmp_path: Path) -> None:
    path = _sidecar(
        tmp_path,
        {
            "ve": {
                "label": "Venous blood",
                "readouts": [_readout("Cve", label="gli, plasma")],
            }
        },
    )
    panels, _ = panels_from_figure(
        SVG, path, simulation=SIMULATION, title="T", group=_grouped
    )
    assert list(panels) == ["ve"]
    assert panels["ve"].title == "Venous blood"


def test_one_group_of_two_units_opens_a_panel_per_unit(tmp_path: Path) -> None:
    path = _sidecar(
        tmp_path,
        {
            "urine": {
                "label": "Urine",
                "readouts": [
                    _readout("Aurine", unit="umol", kind="amount", label="gli, urine"),
                    _readout("Furine", unit="umol/h", kind="rate", label="gli, urine"),
                ],
            }
        },
    )
    panels, _ = panels_from_figure(
        SVG, path, simulation=SIMULATION, title="T", group=_grouped
    )
    assert list(panels) == ["urine_amount", "urine_rate"]
    assert [c.unit for p in panels.values() for c in p.curves] == ["umol", "umol/h"]
    assert [p.title for p in panels.values()] == ["Urine", "Urine"]


def test_a_group_which_is_no_word_text_keeps_the_panel_id_an_sid(
    tmp_path: Path,
) -> None:
    path = _sidecar(
        tmp_path,
        {
            "li": {
                "label": "Liver",
                "readouts": [
                    _readout("LI__Cpl_gli", label="gli, plasma"),
                    _readout("LI__Cti_gli", label="gli, tissue (unbound)"),
                ],
            }
        },
    )
    panels, _ = panels_from_figure(
        SVG, path, simulation=SIMULATION, title="T", group=_grouped
    )
    assert list(panels) == ["li_plasma", "li_tissue__unbound_"]


def test_more_curves_than_the_limit_in_one_group_become_several_panels(
    tmp_path: Path,
) -> None:
    """A group of more readouts than a panel draws is split after its id (#591)."""
    readouts = [
        _readout(f"LI__C{i}", label=f"s{i}, plasma") for i in range(MAX_CURVES + 1)
    ]
    path = _sidecar(tmp_path, {"li": {"label": "Liver", "readouts": readouts}})

    panels, figure = panels_from_figure(
        SVG, path, simulation=SIMULATION, title="T", group=_grouped
    )

    # one group, so the id of the part is the id of the key
    assert figure.panels["li"] == ("li_1", "li_2")
    assert [len(panels[pid].curves) for pid in ("li_1", "li_2")] == [MAX_CURVES, 1]
    assert panels["li_1"].title.endswith("(1 of 2)")


def test_a_group_which_fits_keeps_the_id_and_title_of_its_part(
    tmp_path: Path,
) -> None:
    """A part of at most `MAX_CURVES` readouts is one panel, named as before (#591)."""
    readouts = [_readout(f"LI__C{i}", label=f"s{i}, plasma") for i in range(MAX_CURVES)]
    path = _sidecar(tmp_path, {"li": {"label": "Liver", "readouts": readouts}})

    panels, figure = panels_from_figure(
        SVG, path, simulation=SIMULATION, title="T", group=_grouped
    )

    assert figure.panels["li"] == ("li",)
    assert len(panels["li"].curves) == MAX_CURVES
    assert panels["li"].title == "Liver"


def test_a_group_which_is_no_pair_of_texts_is_refused(tmp_path: Path) -> None:
    path = _sidecar(tmp_path, {"li": {"label": "Liver", "readouts": [_readout("C")]}})
    with pytest.raises(ContractError, match="'li'"):
        panels_from_figure(
            SVG,
            path,
            simulation=SIMULATION,
            title="T",
            group=lambda key, readout: "x",  # ty: ignore[invalid-argument-type]
        )


def test_a_group_which_raises_names_the_key(tmp_path: Path) -> None:
    def boom(key: str, readout: Mapping[str, str]) -> tuple[str, str]:
        raise RuntimeError("no")

    path = _sidecar(tmp_path, {"li": {"label": "Liver", "readouts": [_readout("C")]}})
    with pytest.raises(ContractError, match="'li'"):
        panels_from_figure(SVG, path, simulation=SIMULATION, title="T", group=boom)


def _built() -> tuple[dict[str, Panel], AppFigure]:
    return panels_from_figure(SVG, READOUTS, simulation=SIMULATION, title="T")


def test_an_id_the_model_lacks_is_dropped_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    panels, figure = _built()
    with caplog.at_level(logging.WARNING, logger="sbmlsim.panel_figure"):
        kept, restricted, dropped = restrict_to_model(
            panels, figure, lambda i: i != "LI__Cis_gli"
        )
    assert dropped == ("LI__Cis_gli",)
    assert [c.id for c in kept["li_gli"].curves] == ["LI__Cpl_gli", "LI__Cic_gli"]
    assert "LI__Cis_gli" in caplog.text
    assert restricted.panels["li/sub:gli"] == ("li_gli",)


def test_a_group_with_every_id_missing_becomes_a_decoration() -> None:
    panels, figure = _built()
    gone = {c.id for c in panels["li_gli"].curves}
    kept, restricted, dropped = restrict_to_model(
        panels, figure, lambda i: i not in gone
    )
    assert "li_gli" not in kept
    assert restricted.panels["li/sub:gli"] == ()
    assert set(dropped) == gone
    assert restricted.panels["li"] == ("li",)


def test_a_complete_model_changes_nothing(caplog: pytest.LogCaptureFixture) -> None:
    panels, figure = _built()
    kept, restricted, dropped = restrict_to_model(panels, figure, lambda i: True)
    assert kept == panels
    assert restricted == figure
    assert dropped == ()
    assert not caplog.records


def test_a_model_with_no_readout_of_the_figure_is_refused() -> None:
    panels, figure = _built()
    with pytest.raises(ContractError, match="LI__Cpl_gli"):
        restrict_to_model(panels, figure, lambda i: False)


def test_a_blank_drawing_is_refused_naming_the_file(tmp_path: Path) -> None:
    blank = tmp_path / "blank.svg"
    blank.write_text("  \n", encoding="utf-8")
    with pytest.raises(ContractError, match=r"(?s)blank\.svg.*\nsvg\n"):
        panels_from_figure(blank, READOUTS, simulation=SIMULATION, title="T")


def test_a_blank_title_is_refused_naming_the_file() -> None:
    with pytest.raises(ContractError, match=r"(?s)model\.svg.*\ntitle\n"):
        panels_from_figure(SVG, READOUTS, simulation=SIMULATION, title=" ")


KNOBS = (
    Knob(
        id="dose",
        label="dose",
        target="PODOSE_gli",
        unit="mg",
        min=1,
        max=8,
        default=4,
    ),
    Knob(
        id="crcl",
        label="creatinine clearance",
        unit="ml/min",
        min=10,
        max=130,
        default=125,
    ),
)


def test_every_panel_takes_the_knobs_of_its_page() -> None:
    panels, _ = panels_from_figure(
        SVG, READOUTS, simulation=SIMULATION, title="T", knobs=KNOBS
    )
    assert len(panels) > 1
    assert all(panel.knobs == KNOBS for panel in panels.values())


def test_a_panel_without_given_knobs_has_none() -> None:
    panels, _ = panels_from_figure(SVG, READOUTS, simulation=SIMULATION, title="T")
    assert all(panel.knobs == () for panel in panels.values())
