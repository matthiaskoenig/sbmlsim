"""Test the console output of a fit."""

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

from sbmlsim.fit import FitParameter, FitSettings, MappingKind, display
from sbmlsim.fit.options import (
    LossFunctionType,
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)

PARAMETERS = [
    FitParameter("p1", start_value=1.0, lower_bound=0.1, upper_bound=10.0, unit="mM"),
    FitParameter("p2", lower_bound=1e-4, upper_bound=1.0, unit=None),
]

DATA = pd.DataFrame(
    [
        {"experiment": "A", "fm_key": "fm1", "yid": "y1", "kind": "training"},
        {"experiment": "A", "fm_key": "fm2", "yid": "y2", "kind": "training"},
        {"experiment": "A", "fm_key": "fm3", "yid": "y3", "kind": "outlier"},
        {"experiment": "B", "fm_key": "fm4", "yid": "y4", "kind": "validation"},
    ]
)


def render(renderable: object) -> str:
    """Render to text, as the console would."""
    buffer = StringIO()
    console = Console(file=buffer, width=120, no_color=True)
    console.print(renderable)
    return buffer.getvalue()


def test_parameters_table() -> None:
    """The parameters are a table of their bounds and units."""
    text = render(display.parameters_table(PARAMETERS))
    for token in ["parameter", "start", "lower", "upper", "unit", "p1", "p2", "mM"]:
        assert token in text
    # a parameter without a start value and without a unit
    assert "-" in text
    assert "model" in text


def test_settings_table() -> None:
    """The settings are a table of the options of the fit."""
    settings = FitSettings(
        residual=ResidualType.NORMALIZED,
        loss_function=LossFunctionType.SOFT_L1,
        weighting_curves=(WeightingCurvesType.POINTS,),
        weighting_points=WeightingPointsType.ERROR_WEIGHTING,
    )
    text = render(display.settings_table(settings))
    for token in ["NORMALIZED", "SOFT_L1", "POINTS", "ERROR_WEIGHTING", "1.0e-06"]:
        assert token in text


def test_settings_table_without_weighting() -> None:
    """Curves which are not weighted are reported as none."""
    assert "none" in render(display.settings_table(FitSettings()))


def test_data_summary_table() -> None:
    """The data is counted per experiment and kind."""
    text = render(display.data_summary_table(DATA))
    lines = [line for line in text.splitlines() if line.strip()]

    # every kind is a column, so a new one does not need a new test
    header = next(line for line in lines if "experiment" in line)
    for kind in MappingKind:
        assert kind.value in header

    counts = {"training": 2, "validation": 0, "outlier": 1}
    row_a = next(line for line in lines if line.strip().startswith("A"))
    # A: 2 training, 1 outlier, 3 mappings, and 0 of every other kind
    assert row_a.split() == [
        "A",
        *[str(counts.get(kind.value, 0)) for kind in MappingKind],
        "3",
    ]

    totals = {"training": 2, "validation": 1, "outlier": 1}
    total = next(line for line in lines if "total" in line)
    assert total.split() == [
        "total",
        *[str(totals.get(kind.value, 0)) for kind in MappingKind],
        "4",
    ]


def test_data_summary_table_without_kind() -> None:
    """A table which carries no kind still reports the number of mappings."""
    text = render(display.data_summary_table(pd.DataFrame({"fm_key": ["a", "b"]})))
    assert "unknown" in text
    assert "2" in text


def test_data_table() -> None:
    """Every fit mapping is a row with its observable and its kind."""
    text = render(display.data_table(DATA))
    for token in ["fm1", "fm4", "y3", "validation", "outlier"]:
        assert token in text


def test_key_values_and_sections(capsys: pytest.CaptureFixture[str]) -> None:
    """The sections and the key/value blocks are printed."""
    display.section("Fit problem")
    display.key_values({"strategy": "ALL", "runs": 4})
    out = capsys.readouterr().out
    assert "Fit problem" in out
    assert "strategy" in out
    assert "ALL" in out
    assert "runs" in out


def test_link_is_a_single_line(capsys: pytest.CaptureFixture[str]) -> None:
    """A link is not wrapped, so that the terminal can open it."""
    display.link("report", "results/fit/index.html")
    out = capsys.readouterr().out.strip()
    assert len(out.splitlines()) == 1
    assert out.startswith("report")
    # a URI, so that it is a link on windows as well
    assert out.endswith(Path("results/fit/index.html").resolve().as_uri())
    assert out.split()[-1].startswith("file:///")


def test_print_sections(capsys: pytest.CaptureFixture[str]) -> None:
    """The sections of a fit are printed without a console of their own."""
    display.print_parameters(PARAMETERS)
    display.print_settings(FitSettings())
    display.print_data(DATA)
    out = capsys.readouterr().out
    assert "Parameters (2)" in out
    assert "Settings" in out
    assert "Data (4 fit mappings)" in out
    # the summary comes before the single mappings
    assert out.index("total") < out.index("fm1")


def test_print_data_without_detail(capsys: pytest.CaptureFixture[str]) -> None:
    """The single mappings are optional."""
    display.print_data(DATA, detail=False)
    out = capsys.readouterr().out
    assert "total" in out
    assert "fm1" not in out


def test_section_icon(capsys: pytest.CaptureFixture[str]) -> None:
    """A section carries an icon which tells it apart from the others."""
    display.section("Settings", icon=display.ICON_SETTINGS)
    out = capsys.readouterr().out
    assert "Settings" in out
    # the icon is rendered, not its name
    assert ":gear:" not in out

    display.section("No icon")
    assert "No icon" in capsys.readouterr().out


def test_icons_are_distinct() -> None:
    """Every section has its own icon."""
    icons = [
        display.ICON_FIT,
        display.ICON_PARAMETERS,
        display.ICON_SETTINGS,
        display.ICON_DATA,
        display.ICON_OPTIMIZATION,
        display.ICON_REPORT,
    ]
    assert len(set(icons)) == len(icons)
