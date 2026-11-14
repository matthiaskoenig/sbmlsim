"""The panel declaration of a model, its `panel/1` and its `app/1` contract."""

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
    APP_CONTRACT,
    CONTRACT,
    MAX_CURVES,
    MAX_END,
    MAX_GRID,
    MAX_KNOBS,
    MAX_POINTS,
    App,
    AppPage,
    AppVariant,
    Choice,
    ContractError,
    Knob,
    Panel,
    PanelCurve,
    PanelData,
    PanelSimulation,
    Scenario,
    app_from_payload,
    app_payload,
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

#: the activity of one allele of CYP2C9, an input of the formula of a derived
#: target and of no entity of its own, as a range with named stops
ALLELE1 = Knob(
    id="a1",
    label="activity of allele 1",
    unit="percent",
    kind="range",
    min=0.0,
    max=100.0,
    steps=3,
    choices=(Choice(label="*1", value=100.0), Choice(label="*3", value=10.0)),
    default=100.0,
)
ALLELE2 = ALLELE1.model_copy(update={"id": "a2", "label": "activity of allele 2"})


@pytest.fixture
def panel() -> Panel:
    return Panel(
        id="renal",
        title="Hydrochlorothiazide in renal impairment",
        subtitle="the plasma concentration over the renal function",
        x_label="time (hr)",
        y_min=0.001,
        y_max=1.0,
        x_max=24.0,
        knobs=(EGFR, DOSE, ALLELE1, ALLELE2),
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
            derived={"f_cyp2c9": "(a1 + a2) / 2 / 100"},
        ),
    )


@pytest.fixture
def app() -> App:
    return App(
        title="Hydrochlorothiazide",
        subtitle="the model of hydrochlorothiazide and its clinical questions",
        pages=(
            AppPage(
                id="renal",
                title="Renal impairment",
                panels=("renal", "tubular"),
                scenarios=(
                    Scenario(
                        id="healthy", label="healthy, 25 mg", values={"dose": 25.0}
                    ),
                    Scenario(
                        id="severe",
                        label="severe impairment, 12.5 mg",
                        values={"egfr": 0.2, "dose": 12.5},
                    ),
                ),
            ),
            AppPage(id="pk", title="Pharmacokinetics", panels=("pk",)),
        ),
        variants=(
            AppVariant(id="default", label="healthy"),
            AppVariant(id="impaired", label="renal impairment"),
        ),
        citation="König et al. (2026)",
        doi="10.5281/zenodo.0000000",
        licenses=("CC BY-SA 4.0", "MIT"),
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


def test_a_choice_knob_of_more_choices_than_a_grid_holds_is_refused() -> None:
    """A grid over a choice knob runs its choices, which `MAX_GRID` bounds."""

    def knob(count: int) -> Knob:
        return Knob(
            id="stage",
            label="stage",
            target="f_stage",
            kind="choice",
            choices=tuple(
                Choice(label=str(index), value=float(index)) for index in range(count)
            ),
            default=0.0,
        )

    assert len(knob(MAX_GRID).choices) == MAX_GRID
    with pytest.raises(ValidationError) as error:
        knob(MAX_GRID + 1)
    assert (
        f"The choice knob 'stage' has {MAX_GRID + 1} choices, more than the "
        f"{MAX_GRID} points of a grid over it (`MAX_GRID`)."
    ) in str(error.value)


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


def test_a_series_which_names_a_curve_the_panel_has_not_is_refused() -> None:
    """A curve which is no curve of the panel would be a unit of its own."""
    series = PanelData(
        id="study2000",
        label="Study 2000",
        curve="[S2]",
        dataset="Study2000_Fig1",
        x="t",
        y="mean",
    )
    with pytest.raises(ValidationError, match=r"\[S2\]") as error:
        Panel(
            id="p",
            title="p",
            curves=(PanelCurve(id="[S1]", label="S1"),),
            data=(series,),
            simulation=PanelSimulation(end=1.0),
        )
    message = str(error.value)
    assert "study2000" in message
    assert "[S1]" in message


def test_a_series_whose_own_id_is_no_curve_of_the_panel_is_accepted() -> None:
    """Only a `curve` which is set is a reference to a curve of the panel."""
    series = PanelData(
        id="study2000", label="Study 2000", dataset="Study2000_Fig1", x="t", y="mean"
    )
    panel = Panel(
        id="p",
        title="p",
        curves=(PanelCurve(id="[S1]", label="S1", unit="mM"),),
        data=(series,),
        simulation=PanelSimulation(end=1.0),
    )
    assert panel.series_unit(series) is None


def test_the_unit_of_a_series_is_the_one_of_its_curve_and_else_its_own() -> None:
    series = PanelData(
        id="study2000",
        label="Study 2000",
        curve="[S1]",
        dataset="Study2000_Fig1",
        x="t",
        y="mean",
    )
    without = PanelCurve(id="[S1]", label="S1")
    panel = Panel(
        id="p",
        title="p",
        curves=(PanelCurve(id="[S1]", label="S1", unit="mM"),),
        data=(series,),
        simulation=PanelSimulation(end=1.0),
    )
    assert panel.series_unit(series) == "mM"
    assert panel.series_unit(series.model_copy(update={"unit": "uM"})) == "uM"
    assert panel.model_copy(update={"curves": (without,)}).series_unit(series) is None


def test_the_state_a_series_was_measured_at_round_trips_through_the_payload() -> None:
    """A series names the state of two knobs it was measured under."""
    dose = Knob(
        id="dose",
        label="dose",
        target="PODOSE_hctz",
        unit="mg",
        kind="choice",
        choices=(Choice(label="25 mg", value=25.0), Choice(label="50 mg", value=50.0)),
        default=25.0,
    )
    impairment = Knob(
        id="impairment",
        label="renal function",
        target="f_renal_function",
        kind="choice",
        choices=(Choice(label="normal", value=1.0), Choice(label="severe", value=0.08)),
        default=1.0,
    )
    series = PanelData(
        id="study2000",
        label="Study 2000",
        curve="[S1]",
        dataset="Study2000_Fig1",
        x="t",
        y="mean",
        at={"impairment": 0.08, "dose": 50.0},
    )
    panel = Panel(
        id="p",
        title="p",
        knobs=(dose, impairment),
        curves=(PanelCurve(id="[S1]", label="S1"),),
        data=(series,),
        simulation=PanelSimulation(end=1.0),
    )
    payload = panel_payload(panel)
    assert payload["data"][0]["at"] == {"impairment": 0.08, "dose": 50.0}
    assert panel_from_payload(payload) == panel


def test_a_series_without_a_state_is_drawn_at_every_state() -> None:
    """Data of a condition the panel does not vary states no condition."""
    series = PanelData(
        id="study2000", label="Study 2000", dataset="Study2000_Fig1", x="t", y="mean"
    )
    assert series.at is None
    payload = panel_payload(
        Panel(
            id="p",
            title="p",
            knobs=(DOSE,),
            curves=(PanelCurve(id="study2000", label="S1"),),
            data=(series,),
            simulation=PanelSimulation(end=1.0),
        )
    )
    assert payload["data"][0]["at"] is None


def _measured_at(at: dict[str, float]) -> Panel:
    """A panel of a choice knob and a range knob whose series names `at`."""
    return Panel(
        id="p",
        title="p",
        knobs=(EGFR, DOSE),
        curves=(PanelCurve(id="[S1]", label="S1"),),
        data=(
            PanelData(
                id="study2000",
                label="Study 2000",
                curve="[S1]",
                dataset="Study2000_Fig1",
                x="t",
                y="mean",
                at=at,
            ),
        ),
        simulation=PanelSimulation(end=1.0),
    )


@pytest.mark.parametrize(
    ("at", "knob", "match"),
    [
        ({"weight": 75.0}, "weight", "does not declare"),
        ({"egfr": 0.2}, "egfr", "kind 'range' without stops"),
        ({"dose": 50.0}, "dose", "none of its choices"),
    ],
)
def test_a_state_which_is_no_choice_of_a_knob_is_refused(
    at: dict[str, float], knob: str, match: str
) -> None:
    """A series of such a state is a series which is drawn nowhere."""
    with pytest.raises(ValidationError, match=match) as error:
        _measured_at(at)
    message = str(error.value)
    assert "study2000" in message
    assert knob in message


def test_a_state_of_a_choice_of_a_knob_is_accepted() -> None:
    assert _measured_at({"dose": 12.5}).data[0].at == {"dose": 12.5}


#: the renal function as a range with named stops, e.g. the stages of an
#: impairment; the default 0.9 is no stop
STOPPED = EGFR.model_copy(
    update={
        "choices": (
            Choice(label="severe", value=0.2),
            Choice(label="moderate", value=0.5),
            Choice(label="normal", value=1.0),
        ),
        "default": 0.9,
    }
)


def test_a_range_knob_carries_named_stops_which_its_grid_ignores() -> None:
    knob = Knob.model_validate(STOPPED.model_dump())
    assert knob.kind == "range"
    assert [choice.label for choice in knob.choices] == ["severe", "moderate", "normal"]
    # the default need not be a stop and the grid is the one of `steps`
    assert knob.default == 0.9
    assert knob.values() == EGFR.values()
    assert knob.values(3) == EGFR.values(3)


@pytest.mark.parametrize("value", [0.1, 1.5])
def test_a_stop_outside_the_range_of_its_knob_is_refused(value: float) -> None:
    with pytest.raises(ValidationError, match="outside its bounds") as error:
        EGFR.model_validate(
            {**EGFR.model_dump(), "choices": [{"label": "off", "value": value}]}
        )
    assert f"[{value}]" in str(error.value)
    assert "'egfr'" in str(error.value)


def test_a_state_of_a_stop_of_a_range_knob_is_accepted() -> None:
    panel = Panel(
        id="p",
        title="p",
        knobs=(STOPPED, DOSE),
        curves=(PanelCurve(id="[S1]", label="S1"),),
        data=(
            PanelData(
                id="study2000",
                label="Study 2000",
                curve="[S1]",
                dataset="Study2000_Fig1",
                x="t",
                y="mean",
                at={"egfr": 0.5, "dose": 25.0},
            ),
        ),
        simulation=PanelSimulation(end=1.0),
    )
    assert panel.data[0].at == {"egfr": 0.5, "dose": 25.0}
    assert panel_from_payload(panel_payload(panel)) == panel


def test_a_state_which_is_no_stop_of_a_range_knob_is_refused() -> None:
    """A point of the range which is no stop is no condition, the default neither."""
    for value in (0.9, 0.6):
        with pytest.raises(ValidationError, match="none of its stops") as error:
            Panel(
                id="p",
                title="p",
                knobs=(STOPPED,),
                curves=(PanelCurve(id="[S1]", label="S1"),),
                data=(
                    PanelData(
                        id="study2000",
                        label="Study 2000",
                        curve="[S1]",
                        dataset="Study2000_Fig1",
                        x="t",
                        y="mean",
                        at={"egfr": value},
                    ),
                ),
                simulation=PanelSimulation(end=1.0),
            )
        assert "[0.2, 0.5, 1.0]" in str(error.value)


def test_a_knob_without_a_target_feeds_the_formulas_of_derived_targets(
    panel: Panel,
) -> None:
    assert ALLELE1.target is None
    assert panel.simulation.derived == {"f_cyp2c9": "(a1 + a2) / 2 / 100"}
    payload = panel_payload(panel)
    assert payload["axes"][2]["target"] is None
    assert payload["simulation"]["derived"] == {"f_cyp2c9": "(a1 + a2) / 2 / 100"}
    assert panel_from_payload(payload) == panel
    assert PanelSimulation(end=1.0).derived == {}


@pytest.mark.parametrize(
    ("derived", "preinit", "match"),
    [
        ({"f-cyp2c9": "a1"}, {}, "'f-cyp2c9' is no SId"),
        ({"2f": "a1"}, {}, "'2f' is no SId"),
        ({"f_cyp2c9": "a1"}, {"f_cyp2c9": 1.0}, "pre-initialization"),
        ({"f_cyp2c9": "  "}, {}, "needs its formula"),
    ],
)
def test_a_derived_target_which_is_set_otherwise_or_is_no_target_is_refused(
    derived: dict[str, str], preinit: dict[str, float], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        PanelSimulation(end=1.0, derived=derived, preinit=preinit)


def test_a_derived_target_which_a_knob_sets_is_refused(panel: Panel) -> None:
    simulation = panel.simulation.model_copy(
        update={"derived": {"f_renal_function": "a1 / 100"}}
    )
    with pytest.raises(ValidationError, match="target of its knob 'egfr'") as error:
        Panel.model_validate({**panel.model_dump(), "simulation": simulation})
    assert "'renal'" in str(error.value)
    assert "'f_renal_function'" in str(error.value)


def test_the_label_of_the_value_axis_is_empty_without_one(panel: Panel) -> None:
    """A panel which declares none leaves that label to its renderer."""
    assert panel.y_label == ""
    labelled = panel.model_copy(update={"y_label": "plasma concentration"})
    payload = panel_payload(labelled)
    assert payload["panel"]["y_label"] == "plasma concentration"
    assert panel_from_payload(payload) == labelled
    assert panel_payload(panel)["panel"]["y_label"] == ""
    assert panel_from_payload(panel_payload(panel)).y_label == ""


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


def test_curves_which_declare_different_scales_are_refused() -> None:
    curves = (
        PanelCurve(id="[S1]", label="S1", scale="log"),
        PanelCurve(id="[S2]", label="S2"),
        PanelCurve(id="[S3]", label="S3", scale="log"),
    )
    with pytest.raises(
        ValidationError, match=r"'p'.*linear: \['\[S2\]'\]; log: \['\[S1\]', '\[S3\]'\]"
    ):
        Panel(id="p", title="p", curves=curves, simulation=PanelSimulation(end=1.0))
    # one scale, either of them, is one axis
    for scale in ("linear", "log"):
        same = tuple(curve.model_copy(update={"scale": scale}) for curve in curves)
        assert Panel(
            id="p", title="p", curves=same, simulation=PanelSimulation(end=1.0)
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


def test_a_panel_of_eight_knobs_is_accepted_and_of_more_is_refused() -> None:
    """A knob which is not gridded is one value, the grid bounds the work."""
    assert MAX_KNOBS == 8

    def knobs(n: int) -> tuple[Knob, ...]:
        return tuple(
            EGFR.model_copy(update={"id": f"k{i}", "target": f"p{i}"}) for i in range(n)
        )

    def build(n: int) -> Panel:
        return Panel(
            id="p",
            title="p",
            knobs=knobs(n),
            curves=(PanelCurve(id="[S1]", label="S1"),),
            simulation=PanelSimulation(end=1.0),
        )

    assert len(build(MAX_KNOBS).knobs) == MAX_KNOBS
    with pytest.raises(ValidationError, match="at most 8"):
        build(MAX_KNOBS + 1)


def test_the_limits_of_the_axes_are_none_without_one(minimal: Panel) -> None:
    assert (minimal.y_min, minimal.y_max, minimal.x_max) == (None, None, None)
    payload = panel_payload(minimal)["panel"]
    assert (payload["y_min"], payload["y_max"], payload["x_max"]) == (None, None, None)


@pytest.mark.parametrize(
    ("limits", "match"),
    [
        ({"y_min": 2.0, "y_max": 1.0}, "y_min < y_max"),
        ({"y_min": 1.0, "y_max": 1.0}, "y_min < y_max"),
        ({"y_min": 0.0}, "y_min=0.0 .* not positive"),
        ({"y_min": None, "y_max": -1.0}, "y_max=-1.0 .* not positive"),
        ({"x_max": 0.0}, "x_max=0.0 .* not positive"),
        ({"y_max": float("inf")}, "finite"),
        ({"x_max": float("nan")}, "finite"),
    ],
)
def test_limits_of_the_axes_which_are_no_axis_are_refused(
    panel: Panel, limits: dict[str, float | None], match: str
) -> None:
    """The curves of `panel` declare the `log` scale."""
    with pytest.raises(ValidationError, match=match) as error:
        Panel.model_validate({**panel.model_dump(), **limits})
    if "finite" not in match:
        assert "'renal'" in str(error.value)


@pytest.mark.parametrize("limit", ["y_min", "y_max"])
def test_a_y_limit_of_a_panel_without_curves_is_refused(
    minimal: Panel, limit: str
) -> None:
    """Its value axis has no unit, so no number is a limit of it."""
    without = {**minimal.model_dump(), "curves": []}
    assert Panel.model_validate({**without, "x_max": 5.0}).x_max == 5.0
    with pytest.raises(ValidationError, match="'decay' limits its value axis") as error:
        Panel.model_validate({**without, limit: 1.0})
    assert "draws no curve" in str(error.value)


def test_a_y_limit_of_zero_is_a_limit_of_a_linear_axis(minimal: Panel) -> None:
    limited = Panel.model_validate(
        {**minimal.model_dump(), "y_min": 0.0, "y_max": 5.0, "x_max": 8.0}
    )
    assert (limited.y_min, limited.y_max, limited.x_max) == (0.0, 5.0, 8.0)
    # a lower limit alone and an upper limit alone are limits as well
    assert Panel.model_validate({**minimal.model_dump(), "y_min": -1.0}).y_min == -1.0
    assert Panel.model_validate({**minimal.model_dump(), "y_max": -1.0}).y_max == -1.0


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
    assert [axis["id"] for axis in payload["axes"]] == ["egfr", "dose", "a1", "a2"]
    assert payload["panel"] == {
        "id": "renal",
        "title": panel.title,
        "subtitle": panel.subtitle,
        "x_label": panel.x_label,
        "y_label": panel.y_label,
        "y_min": 0.001,
        "y_max": 1.0,
        "x_max": 24.0,
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


def test_the_contract_of_an_app_is_its_version() -> None:
    assert APP_CONTRACT == "app/1"


def test_the_app_payload_round_trips(app: App) -> None:
    payload = app_payload(app)
    assert payload["contract"] == "app/1"
    assert [page["id"] for page in payload["pages"]] == ["renal", "pk"]
    assert [page["panels"] for page in payload["pages"]] == [
        ["renal", "tubular"],
        ["pk"],
    ]
    assert [variant["label"] for variant in payload["variants"]] == [
        "healthy",
        "renal impairment",
    ]
    assert payload["licenses"] == ["CC BY-SA 4.0", "MIT"]
    assert payload["pages"][0]["scenarios"] == [
        {"id": "healthy", "label": "healthy, 25 mg", "values": {"dose": 25.0}},
        {
            "id": "severe",
            "label": "severe impairment, 12.5 mg",
            "values": {"egfr": 0.2, "dose": 12.5},
        },
    ]
    assert payload["pages"][1]["scenarios"] == []
    assert app_from_payload(payload) == app


@pytest.mark.parametrize(
    ("scenario", "match"),
    [
        ({"id": "1st", "label": "x", "values": {"dose": 1.0}}, "'1st' .* no SId"),
        ({"id": "s", "label": " ", "values": {"dose": 1.0}}, "needs its label"),
        ({"id": "s", "label": "x", "values": {}}, "sets no knob"),
        ({"id": "s", "label": "x", "values": {"dose": float("nan")}}, "finite"),
    ],
)
def test_a_scenario_which_is_no_state_of_knobs_is_refused(
    scenario: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        Scenario.model_validate(scenario)


def test_two_scenarios_of_one_id_on_a_page_are_refused() -> None:
    one = Scenario(id="healthy", label="healthy", values={"dose": 25.0})
    with pytest.raises(ValidationError, match=r"scenarios \['healthy'\] twice"):
        AppPage(id="renal", title="Renal", panels=("renal",), scenarios=(one, one))
    # the same id on two pages is two scenarios
    App(
        title="t",
        pages=(
            AppPage(id="a", title="A", panels=("renal",), scenarios=(one,)),
            AppPage(id="b", title="B", panels=("renal",), scenarios=(one,)),
        ),
    )


def test_the_app_payload_round_trips_a_minimal_app() -> None:
    """An app of the required fields only, so the defaults round trip too."""
    minimal = App(
        title="decay",
        pages=(AppPage(id="decay", title="decay", panels=("decay",)),),
    )
    assert (minimal.subtitle, minimal.citation, minimal.doi) == ("", "", "")
    assert (minimal.variants, minimal.licenses) == ((), ())
    assert app_from_payload(app_payload(minimal)) == minimal


def test_the_app_payload_is_json_and_carries_no_python_type(app: App) -> None:
    payload = app_payload(app)
    assert json.loads(json.dumps(payload)) == payload
    for value in _leaves(payload):
        assert value is None or isinstance(value, (str, int, float, bool)), value


def test_an_app_payload_of_another_contract_version_is_refused() -> None:
    with pytest.raises(ContractError, match="app/2"):
        app_from_payload({"contract": "app/2", "title": "decay", "pages": []})
    with pytest.raises(ContractError):
        app_from_payload({"title": "decay", "pages": []})


def test_an_unknown_field_of_an_app_payload_is_refused(app: App) -> None:
    payload = app_payload(app)
    payload["pages"][0]["unknown"] = 1
    with pytest.raises(ValidationError, match="unknown"):
        app_from_payload(payload)


def test_an_app_without_a_page_is_refused() -> None:
    with pytest.raises(ValidationError):
        App(title="decay", pages=())


def test_a_page_without_a_panel_is_refused() -> None:
    with pytest.raises(ValidationError, match="panel"):
        App(title="decay", pages=(AppPage(id="decay", title="decay", panels=()),))


def test_a_page_which_lists_one_panel_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="renal"):
        AppPage(id="kidney", title="Kidney", panels=("renal", "pk", "renal"))


def test_two_pages_of_one_id_are_refused() -> None:
    page = AppPage(id="renal", title="Renal impairment", panels=("renal",))
    with pytest.raises(ValidationError, match="renal"):
        App(title="hctz", pages=(page, page))


def test_two_variants_of_one_id_are_refused() -> None:
    variant = AppVariant(id="default", label="healthy")
    with pytest.raises(ValidationError, match="default"):
        App(
            title="hctz",
            pages=(AppPage(id="renal", title="Renal", panels=("renal",)),),
            variants=(variant, variant),
        )


@pytest.mark.parametrize(
    "id_",
    ["", "1page", "renal pk", "renal/pk", "../pk", "renal.pk", "renal-pk"],
)
def test_an_id_of_a_page_or_a_variant_which_is_no_sid_is_refused(id_: str) -> None:
    """The id of a page is the one a link of the app names, so it is no path."""
    with pytest.raises(ValidationError, match="SId"):
        AppPage(id=id_, title="Renal impairment", panels=("renal",))
    with pytest.raises(ValidationError, match="SId"):
        AppVariant(id=id_, label="healthy")


@pytest.mark.parametrize(
    "build",
    [
        lambda: App(
            title="  ",
            pages=(AppPage(id="renal", title="Renal", panels=("renal",)),),
        ),
        lambda: AppPage(id="renal", title="", panels=("renal",)),
        lambda: AppVariant(id="default", label=""),
    ],
)
def test_a_title_which_is_empty_is_refused(build: Callable[[], Any]) -> None:
    """A title and a label are shown, so they are no empty text."""
    with pytest.raises(ValidationError):
        build()


def test_an_app_is_frozen_and_forbids_unknown_fields(app: App) -> None:
    with pytest.raises(ValidationError):
        app.title = "another title"
    with pytest.raises(ValidationError, match="footer"):
        App(
            title="hctz",
            pages=(AppPage(id="renal", title="Renal", panels=("renal",)),),
            footer="a footer",
        )


def test_an_app_is_picklable(app: App) -> None:
    assert pickle.loads(pickle.dumps(app)) == app


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


def test_a_project_declares_the_app_of_its_model(tmp_path: Path, app: App) -> None:
    """The pages of the app are the project's, they are no model semantics."""
    from sbmlsim.project import Project

    project = Project(name="toy", root=tmp_path, app=app)
    assert project.app == app
    assert Project(name="toy", root=tmp_path).app is None
    with pytest.raises(ValidationError):
        Project(name="toy", root=tmp_path, app="an app")
