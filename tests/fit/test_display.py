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
    for token in [
        "NORMALIZED",
        "SOFT_L1",
        "POINTS",
        "ERROR_WEIGHTING",
        "concentration 1.0e-10",
    ]:
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


def test_data_table_metadata() -> None:
    """The fields of the metadata are columns, a missing value is a dash."""
    df = DATA.copy()
    df["route"] = ["IV", "PO", None, "PO"]
    text = render(display.data_table(df))
    assert "route" in text
    assert "IV" in text
    assert "-" in text.splitlines()[4]


def test_data_table_wide(capsys: pytest.CaptureFixture[str]) -> None:
    """A table with many metadata columns is not truncated."""
    df = DATA.copy()
    for i in range(12):
        df[f"metadata_field_{i}"] = f"value_of_field_{i}"
    display.print_data(df)
    out = capsys.readouterr().out
    assert "…" not in out
    assert "value_of_field_11" in out


def test_data_table_excluded_row_is_grey() -> None:
    """The complete row of an excluded mapping is grey."""
    df = DATA.copy()
    df.loc[3, "kind"] = MappingKind.EXCLUDED.value
    buffer = StringIO()
    Console(file=buffer, width=120, force_terminal=True, color_system="standard").print(
        display.data_table(df)
    )
    lines = buffer.getvalue().splitlines()
    grey = next(line for line in lines if "fm4" in line)
    other = next(line for line in lines if "fm1" in line)
    # the row style is applied to the first cell, the experiment
    assert "\x1b" in grey.split("fm4")[0]
    assert "\x1b" not in other.split("fm1")[0].strip()


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


def test_the_parameter_table_shows_a_target_only_when_there_is_one() -> None:
    """An ordinary fit is not given a column of repeated names."""
    from sbmlsim.fit.display import parameters_table
    from sbmlsim.fit.objects import FitParameter

    plain = parameters_table([FitParameter("Ka", 1.0, 0.1, 10.0, "1/hr")])
    assert [c.header for c in plain.columns] == [
        "parameter",
        "start",
        "lower",
        "upper",
        "unit",
    ]

    versioned = parameters_table(
        [FitParameter("Ka_po", 1.0, 0.1, 10.0, "1/hr", target="Ka")]
    )
    assert "target" in [c.header for c in versioned.columns]


def test_the_coverage_table_names_the_uncovered_simulations() -> None:
    """The table says which simulations keep the value of the model."""
    from sbmlsim.fit.display import coverage_table
    from sbmlsim.fit.parameter_mapping import CoverageRow

    table = coverage_table(
        [
            CoverageRow(
                "Ka_po", "Ka", 6, 9, ["Beermann1976|iv1_5", "Beermann1976|iv35_4"]
            )
        ]
    )
    assert table.row_count == 1


def test_print_parameters_shows_the_coverage_of_an_uncovered_parameter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The coverage table is printed only when some simulation is uncovered."""
    from sbmlsim.fit.parameter_mapping import CoverageRow

    coverage = [CoverageRow("Ka_po", "Ka", 6, 9, ["Beermann1976|iv1_5"])]
    display.print_parameters(PARAMETERS, coverage=coverage)
    out = capsys.readouterr().out
    assert "Beermann1976|iv1_5" in out


def test_print_parameters_hides_the_coverage_table_when_everything_is_covered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fit whose parameters reach every simulation gets no coverage table."""
    from sbmlsim.fit.parameter_mapping import CoverageRow

    coverage = [CoverageRow("Ka", "Ka", 2, 2, [])]
    display.print_parameters(PARAMETERS, coverage=coverage)
    out = capsys.readouterr().out
    assert "not covered" not in out


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


def test_the_parameters_of_a_hook_are_one_row_per_array(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sbmlsim.fit.derived import HookSummary, ParameterGroup
    from sbmlsim.fit.display import groups_table, hooks_table, print_parameters

    # the elements of a network are dimensionless, as `network_fit_parameters` makes them
    elements = [
        FitParameter(f"net__l__w__{k}", float(k), -5.0, 5.0, unit="dimensionless")
        for k in range(4)
    ]
    ids = (*[p.pid for p in elements], "net__l__w__4")
    summary = HookSummary(
        name="net",
        kind="pre_initialization",
        description="l (Linear)",
        targets=("gamma",),
        groups=(ParameterGroup("net.l.w", ids),),
    )
    print_parameters(
        [FitParameter("alpha", 1.0, 0.0, 10.0, unit="1/min"), *elements],
        hooks=[summary],
    )
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "net.l.w" in out
    assert "net__l__w__0" not in out
    table = groups_table([(summary.groups[0], elements)])
    assert table.row_count == 1
    assert hooks_table([summary]).row_count == 1


def test_a_network_in_two_hooks_is_printed_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The arrays of a network which two hooks share are one row each, and the
    coverage of their elements is left out."""
    from sbmlsim.fit.derived import HookSummary, ParameterGroup
    from sbmlsim.fit.display import print_parameters
    from sbmlsim.fit.parameter_mapping import CoverageRow

    elements = [
        FitParameter(f"net__l__w__{k}", float(k), -5.0, 5.0, unit="dimensionless")
        for k in range(3)
    ]
    group = ParameterGroup("net.l.w", tuple(p.pid for p in elements))
    long_layers = ", ".join(f"layer{k} (Linear)" for k in range(8))
    hooks = [
        HookSummary("net", "observable", long_layers, ("gamma",), (group,)),
        HookSummary("net", "rhs", long_layers, ("delta",), (group,)),
    ]
    coverage = [
        CoverageRow("alpha", "alpha", 1, 2, ["other"]),
        *[CoverageRow(p.pid, p.pid, 1, 2, ["other"]) for p in elements],
    ]
    print_parameters(
        [FitParameter("alpha", 1.0, 0.0, 10.0, unit="mM"), *elements],
        coverage=coverage,
        hooks=hooks,
    )
    out = capsys.readouterr().out
    assert out.count("net.l.w") == 1
    assert "net__l__w__0" not in out
    # the coverage table lists the single parameter only
    assert "not covered" in out
    # the description of the layers is not cut at the width of the console
    assert "…" not in out
    assert long_layers in out.replace("\n", "")


def test_a_versioned_element_is_a_member_of_its_array() -> None:
    """The elements of an array are found by the entity they write."""
    from sbmlsim.fit.derived import HookSummary, ParameterGroup, group_parameters

    group = ParameterGroup("net.l.w", ("net__l__w__0", "net__l__w__1"))
    first = FitParameter("w0_a", 1.0, unit="dimensionless", target="sciml:net__l__w__0")
    second = FitParameter("net__l__w__1", 2.0, unit="dimensionless")
    other = FitParameter("alpha", 1.0, unit="mM")
    single, groups = group_parameters(
        [first, other, second],
        [
            HookSummary("net", "rhs", "l (Linear)", (), (group,)),
            HookSummary("net", "observable", "l (Linear)", (), (group,)),
        ],
    )
    assert single == [other]
    assert groups == [(group, [first, second])]
