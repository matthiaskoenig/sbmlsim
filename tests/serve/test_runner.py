"""The registry of the served models and the runner of their panels.

The model, the panel and the registry of the tests are the fixtures of
`tests/serve/conftest.py`; the runner of them is built here, as the tests of
the API build the application instead.
"""

import math
import re
import threading
from pathlib import Path

import numpy as np
import pytest
from pint.errors import UndefinedUnitError

from examples.panels.hctz import DOSE, EGFR_STAGES, END, HCTZ_RENAL_URINE, URINE
from sbmlsim.panel import (
    App,
    AppPage,
    AppVariant,
    Choice,
    Knob,
    Panel,
    PanelCurve,
    Scenario,
)
from sbmlsim.project import Project
from sbmlsim.serve import runner as runner_module
from sbmlsim.serve.runner import (
    DeliveredUnit,
    ModelEntry,
    ModelRegistry,
    PanelResult,
    PanelRunner,
    UnknownModel,
    UnknownPanel,
)
from sbmlsim.units import ureg

#: the root of the repository, whose documentation quotes the refusals
ROOT = Path(__file__).parents[2]
#: the reference of the project of the HCTZ example
HCTZ_REFERENCE = "examples.hctz_fitting.project:PROJECT"
#: the seconds a wait which must succeed gives up after, i.e. a failed test
TIMEOUT = 20.0
#: the seconds a parked load of a model gives up after, so nothing hangs
PARKED = 120.0


@pytest.fixture(scope="module")
def runner(registry: ModelRegistry) -> PanelRunner:
    return PanelRunner(registry)


def test_a_point_gives_one_curve_per_declared_curve(
    runner: PanelRunner, panel: Panel
) -> None:
    result = runner.run("hctz", panel, {"egfr": 1.0, "dose": 25.0}, {})
    assert len(result.x) == panel.simulation.points + 1
    assert result.x[0] == 0.0
    assert result.x[-1] == END
    assert set(result.curves) == {curve.id for curve in panel.curves}
    assert result.dims == ()
    assert result.failed == ()
    row = result.curves["[Cve_hctz]"]
    assert len(row) == 1
    assert len(row[0]) == len(result.x)
    assert max(row[0]) > 0.0


def test_a_grid_gives_a_row_per_knob_value(runner: PanelRunner, panel: Panel) -> None:
    # a grid over a choice knob runs its choices, the three stages of the knob
    result = runner.run("hctz", panel, {"dose": 25.0}, {"egfr": 3})
    assert [axis.id for axis in result.dims] == ["egfr"]
    assert len(result.dims[0].values) == 3
    assert result.dims[0].values == EGFR_STAGES.values(None)
    assert result.dims[0].label == "renal function"
    assert result.dims[0].unit == "dimensionless"
    assert len(result.curves["[Cve_hctz]"]) == 3
    assert all(len(row) == len(result.x) for row in result.curves["[Cve_hctz]"])
    # a lower renal function clears less, so more of the dose is left in the
    # plasma at the end of the day
    ends = [row[-1] for row in result.curves["[Cve_hctz]"]]
    assert ends == sorted(ends, reverse=True)


def test_a_grid_of_two_knobs_gives_a_row_per_point_in_c_order(
    runner: PanelRunner, panel: Panel
) -> None:
    result = runner.run("hctz", panel, {}, {"egfr": 2, "dose": 3})
    assert [axis.id for axis in result.dims] == ["egfr", "dose"]
    assert result.dims[0].values == EGFR_STAGES.values(None)
    assert result.dims[1].values == DOSE.values(None)
    rows = result.curves["[Cve_hctz]"]
    assert len(rows) == len(EGFR_STAGES.choices) * len(DOSE.choices)
    # the rows of one renal function follow each other, the dose fastest
    assert max(rows[0]) < max(rows[1]) < max(rows[2]) < max(rows[3])


def test_the_curves_carry_the_units_of_the_model(
    runner: PanelRunner, panel: Panel
) -> None:
    result = runner.run("hctz", panel, {"egfr": 1.0, "dose": 25.0}, {})
    # the panel declares no x unit, so the x is in the time unit of the model,
    # which is the unit of the `end` of the panel
    assert result.units["x"] == "min"
    assert result.units["[Cve_hctz]"] == "uM"


def test_a_declared_x_unit_delivers_the_time_axis_in_it(
    runner: PanelRunner, panel: Panel
) -> None:
    in_hours = panel.model_copy(
        update={"simulation": panel.simulation.model_copy(update={"x_unit": "hr"})}
    )
    result = runner.run("hctz", in_hours, {"egfr": 1.0, "dose": 25.0}, {})
    assert result.units["x"] == "hr"
    # the `end` of the panel stays in the time unit of the model, a day of it
    assert len(result.x) == panel.simulation.points + 1
    assert result.x[0] == 0.0
    assert result.x[-1] == pytest.approx(END / 60.0)
    assert result.x[-1] == pytest.approx(24.0)
    # the curves are the ones of the same simulation
    assert result.curves == runner.run("hctz", panel, {"egfr": 1.0}, {}).curves


def test_the_units_of_a_panel_are_the_ones_of_its_result(
    runner: PanelRunner, registry: ModelRegistry
) -> None:
    """`units` answers without a run what a run delivers, see the module.

    The study data is declared under the curve of the amount in the urine, so
    the panel of this test is the one which carries it.
    """
    with_data = HCTZ_RENAL_URINE
    units = runner.units(registry.entry("hctz"), with_data)
    assert units.x is not None
    assert units.x.text == "min"
    urine = units.series[URINE.id]
    assert urine is not None
    assert urine.text == "umol"
    result = runner.run("hctz", with_data, {"egfr": 1.0, "dose": 25.0}, {})
    assert units.x.text == result.units["x"]


def test_the_x_unit_a_panel_declares_is_the_unit_of_its_units(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    in_hours = panel.model_copy(
        update={"simulation": panel.simulation.model_copy(update={"x_unit": "hr"})}
    )
    units = runner.units(registry.entry("hctz"), in_hours)
    assert units.x is not None
    assert units.x.text == "hr"
    assert units.x.unit == ureg.Unit("hour")


def test_a_unit_id_of_the_model_is_resolved_for_the_x_and_for_the_data(
    runner: PanelRunner, registry: ModelRegistry
) -> None:
    """A panel may declare a unit id of its model, which pint does not know."""
    # `KI__min` is the minute and `substance` the mole of the HCTZ model
    with pytest.raises(UndefinedUnitError):
        ureg.Unit("KI__min")
    with_data = HCTZ_RENAL_URINE
    urine = PanelCurve(id="Aurine_hctz", label="urine", unit="substance")
    declared = with_data.model_copy(
        update={
            "curves": (urine,),
            "simulation": with_data.simulation.model_copy(update={"x_unit": "KI__min"}),
        }
    )
    units = runner.units(registry.entry("hctz"), declared)
    assert units.x is not None
    assert (units.x.text, units.x.unit) == ("KI__min", ureg.Unit("minute"))
    series = units.series[URINE.id]
    assert series is not None
    assert (series.text, series.unit) == ("substance", ureg.Unit("mole"))
    result = runner.run("hctz", declared, {"egfr": 1.0}, {})
    assert result.units["x"] == units.x.text
    assert result.units["Aurine_hctz"] == series.text


@pytest.mark.parametrize("x_unit", ["mg", "not_a_unit"])
def test_an_x_unit_which_is_no_time_of_the_model_is_refused(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel, x_unit: str
) -> None:
    """A unit which is no time and one which is no unit at all are refused."""
    bogus = panel.model_copy(
        update={"simulation": panel.simulation.model_copy(update={"x_unit": x_unit})}
    )
    for run in (
        lambda: runner.run("hctz", bogus, {"egfr": 1.0}, {}),
        lambda: runner.units(registry.entry("hctz"), bogus),
    ):
        with pytest.raises(ValueError) as error:
            run()
        assert x_unit in str(error.value)
        assert "min" in str(error.value)


@pytest.mark.parametrize("unit", ["mg", "not_a_unit"])
def test_a_unit_of_a_curve_which_is_no_unit_of_its_values_is_refused(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel, unit: str
) -> None:
    """The declaration is refused once, where a run and the study data meet it.

    `run` is the path of a simulation of the panel, which converts the values
    of the model, and `units` the path of its study data, which only asks for
    the unit the curve is drawn in: a declaration the model does not convert
    to is no unit of the panel on either, and both answer a 500.
    """
    bogus = panel.model_copy(
        update={
            "curves": (panel.curves[0].model_copy(update={"unit": unit}),),
            "data": (URINE.model_copy(update={"curve": "[Cve_hctz]"}),),
        }
    )
    for refuses in (
        lambda: runner.run("hctz", bogus, {"egfr": 1.0}, {}),
        lambda: runner.units(registry.entry("hctz"), bogus),
    ):
        with pytest.raises(ValueError) as error:
            refuses()
        assert unit in str(error.value)
        # the unit the model gives the selection of the curve
        assert "mmol" in str(error.value)


def test_y_limits_of_curves_of_one_unit_are_served(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    """The two curves of the panel are delivered in `uM`, so a number is a limit."""
    limited = panel.model_copy(update={"y_min": 0.0, "y_max": 2.0, "x_max": END / 60})
    limited = limited.model_copy(
        update={
            "simulation": limited.simulation.model_copy(update={"x_unit": "hr"}),
        }
    )
    units = runner.units(registry.entry("hctz"), limited)
    assert units.x.text == "hr"
    result = runner.run("hctz", limited, {"egfr": 1.0}, {})
    assert result.x[-1] == pytest.approx(END / 60)


def test_y_limits_of_curves_of_different_units_are_refused(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    """Curves of `uM` and `nM` share no number which could be a limit of their axis.

    It is refused where the units of the panel are resolved, for a run and for
    the study data alike, as the units of a curve are; a panel without limits
    is no such problem, since a renderer fits each curve as it is.
    """
    mixed = panel.model_copy(
        update={
            "curves": (
                panel.curves[0],
                panel.curves[1].model_copy(update={"unit": "nM"}),
            ),
        }
    )
    assert runner.units(registry.entry("hctz"), mixed).x.text == "min"
    bogus = [
        mixed.model_copy(update=limits) for limits in ({"y_max": 2.0}, {"y_min": 0.1})
    ]
    refusals = [
        *(lambda one=one: runner.run("hctz", one, {"egfr": 1.0}, {}) for one in bogus),
        *(lambda one=one: runner.units(registry.entry("hctz"), one) for one in bogus),
    ]
    for refuses in refusals:
        with pytest.raises(ValueError, match="do not share") as error:
            refuses()
        message = str(error.value)
        for named in ("'renal_plasma'", "'[Cve_hctz]' in 'uM'", "'nM'"):
            assert named in message


def test_y_limits_of_curves_of_one_unit_spelled_otherwise_are_served(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    """The unit is compared and not its text: `uM` and `umol/l` are one unit."""
    spelled = panel.model_copy(
        update={
            "y_max": 2.0,
            "curves": (
                panel.curves[0],
                panel.curves[1].model_copy(update={"unit": "umol/l"}),
            ),
        }
    )
    assert runner.units(registry.entry("hctz"), spelled).x.text == "min"


@pytest.mark.parametrize(
    ("x_unit", "x_max", "end"),
    [(None, END + 1.0, "1440.0 in 'min'"), ("hr", 25.0, "24.0 in 'hr'")],
)
def test_an_x_max_beyond_the_end_of_the_simulation_is_refused(
    runner: PanelRunner,
    registry: ModelRegistry,
    panel: Panel,
    x_unit: str | None,
    x_max: float,
    end: str,
) -> None:
    """The `x_max` is in the unit of the x, the `end` in the time unit of the model."""
    bogus = panel.model_copy(
        update={
            "x_max": x_max,
            "simulation": panel.simulation.model_copy(update={"x_unit": x_unit}),
        }
    )
    for refuses in (
        lambda: runner.run("hctz", bogus, {"egfr": 1.0}, {}),
        lambda: runner.units(registry.entry("hctz"), bogus),
    ):
        with pytest.raises(ValueError, match="beyond the end") as error:
            refuses()
        message = str(error.value)
        assert f"x_max={x_max}" in message
        assert end in message
        assert "'renal_plasma'" in message
    # the end itself is a limit, in either unit
    at_end = bogus.model_copy(update={"x_max": END if x_unit is None else 24.0})
    assert runner.units(registry.entry("hctz"), at_end).x.text == (x_unit or "min")


def test_a_unit_of_a_series_which_is_no_unit_is_refused(
    runner: PanelRunner, registry: ModelRegistry
) -> None:
    """It is refused where it is resolved, before a series is read."""
    bogus = HCTZ_RENAL_URINE.model_copy(
        update={"data": (URINE.model_copy(update={"unit": "not_a_unit"}),)}
    )
    with pytest.raises(ValueError) as error:
        runner.units(registry.entry("hctz"), bogus)
    assert "not_a_unit" in str(error.value)
    assert URINE.id in str(error.value)


def test_a_curve_without_a_unit_carries_the_one_of_the_model(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    curve = PanelCurve(id="[Cve_hctz]", label="hydrochlorothiazide plasma")
    without = panel.model_copy(update={"curves": (curve,), "data": ()})
    declared = runner.run("hctz", panel, {"egfr": 1.0, "dose": 25.0}, {})
    result = runner.run("hctz", without, {"egfr": 1.0, "dose": 25.0}, {})
    assert result.units["[Cve_hctz]"] == "mmol/litre"
    # the study data under such a curve is delivered in that unit as well
    under = without.model_copy(
        update={"data": (URINE.model_copy(update={"curve": "[Cve_hctz]"}),)}
    )
    series = runner.units(registry.entry("hctz"), under).series
    assert series[URINE.id] == DeliveredUnit(
        text="mmol/litre", unit=ureg.Unit("mmol/litre")
    )
    # the declared unit converts the values of the model, it does not relabel them
    assert result.curves["[Cve_hctz]"][0][-1] * 1000.0 == pytest.approx(
        declared.curves["[Cve_hctz]"][0][-1]
    )


def test_a_knob_value_overrides_the_preinit_of_the_panel(
    runner: PanelRunner, panel: Panel
) -> None:
    # the panel pre-initializes the dose to zero, the knob sets it
    low = runner.run("hctz", panel, {"egfr": 1.0, "dose": 12.5}, {})
    high = runner.run("hctz", panel, {"egfr": 1.0, "dose": 50.0}, {})
    assert 0.0 < max(low.curves["[Cve_hctz]"][0]) < max(high.curves["[Cve_hctz]"][0])


def test_a_knob_which_a_request_leaves_out_runs_at_its_default(
    runner: PanelRunner, panel: Panel
) -> None:
    given = runner.run("hctz", panel, {"egfr": 1.0, "dose": 25.0}, {})
    left_out = runner.run("hctz", panel, {}, {})
    assert left_out.curves == given.curves


def test_a_value_of_no_knob_of_the_panel_is_refused(
    runner: PanelRunner, panel: Panel
) -> None:
    with pytest.raises(ValueError, match="kidney"):
        runner.run("hctz", panel, {"kidney": 1.0}, {})
    with pytest.raises(ValueError, match="kidney"):
        runner.run("hctz", panel, {}, {"kidney": 3})


#: the creatinine clearance, a knob without a target which the formula of the
#: renal function reads, in its declared unit
CRCL = Knob(
    id="crcl",
    label="creatinine clearance",
    unit="ml/min",
    min=10.0,
    max=150.0,
    steps=3,
    default=110.0,
)
#: the renal function computed from the creatinine clearance
RENAL_FUNCTION = {"f_renal_function": "min(crcl / 110, 1)"}


def _derived(panel: Panel, knobs: tuple[Knob, ...], derived: dict[str, str]) -> Panel:
    """The panel with other knobs and derived targets."""
    return panel.model_copy(
        update={
            "knobs": knobs,
            "simulation": panel.simulation.model_copy(update={"derived": derived}),
        }
    )


def _renal_function(values: list[float]) -> Knob:
    """The renal function as a knob which sets its target, at these values."""
    return Knob(
        id="egfr",
        label="renal function",
        target="f_renal_function",
        kind="choice",
        choices=tuple(Choice(label=str(v), value=v) for v in values),
        default=values[0],
    )


def test_a_derived_target_over_the_knob_of_the_grid_is_the_target_set_directly(
    runner: PanelRunner, panel: Panel
) -> None:
    """The formula is a value of the dimension of the grid, at each of its points."""
    derived = _derived(panel, (CRCL, DOSE), RENAL_FUNCTION)
    result = runner.run("hctz", derived, {"dose": 25.0}, {"crcl": 3})
    assert [axis.id for axis in result.dims] == ["crcl"]
    assert result.dims[0].values == (10.0, 80.0, 150.0)
    assert result.dims[0].unit == "ml/min"
    direct = _derived(panel, (_renal_function([10 / 110, 80 / 110, 1.0]), DOSE), {})
    expected = runner.run("hctz", direct, {"dose": 25.0}, {"egfr": 3})
    assert result.failed == expected.failed == ()
    for curve in panel.curves:
        rows = np.asarray(result.curves[curve.id])
        assert rows.shape == (3, len(result.x))
        assert np.array_equal(rows, np.asarray(expected.curves[curve.id]))
    # the three rows are three renal functions and not one
    plasma = np.asarray(result.curves["[Cve_hctz]"])
    assert not np.array_equal(plasma[0], plasma[2])


def test_a_derived_target_over_a_knob_which_is_no_grid_is_a_preinit_value(
    runner: PanelRunner, panel: Panel
) -> None:
    derived = _derived(panel, (CRCL, DOSE), RENAL_FUNCTION)
    result = runner.run("hctz", derived, {"crcl": 55.0}, {"dose": 4})
    assert [axis.id for axis in result.dims] == ["dose"]
    direct = _derived(panel, (_renal_function([0.5]), DOSE), {})
    expected = runner.run("hctz", direct, {"egfr": 0.5}, {"dose": 4})
    for curve in panel.curves:
        assert np.array_equal(
            np.asarray(result.curves[curve.id]), np.asarray(expected.curves[curve.id])
        )
    # and the knob is the formula's: another clearance is another curve
    other = runner.run("hctz", derived, {"crcl": 110.0}, {"dose": 4})
    assert other.curves["[Cve_hctz]"] != result.curves["[Cve_hctz]"]


def test_a_formula_of_two_knobs_of_the_grid_has_its_value_at_every_point(
    runner: PanelRunner, panel: Panel
) -> None:
    """The product of the grid is one dimension, in C order of the knobs."""
    a1 = CRCL.model_copy(
        update={"id": "a1", "unit": None, "min": 0.5, "max": 1.0, "default": 1.0}
    )
    a2 = a1.model_copy(update={"id": "a2"})
    # a formula of exact numbers, whatever order sympy evaluates it in
    derived = _derived(panel, (a1, a2, DOSE), {"f_renal_function": "(2 * a1 + a2) / 4"})
    result = runner.run("hctz", derived, {"dose": 25.0}, {"a1": 2, "a2": 3})
    assert [axis.id for axis in result.dims] == ["a1", "a2"]
    assert [axis.values for axis in result.dims] == [(0.5, 1.0), (0.5, 0.75, 1.0)]
    # the points in C order: a1 is the outer axis and a2 the inner one
    values = [(2 * x1 + x2) / 4 for x1 in (0.5, 1.0) for x2 in (0.5, 0.75, 1.0)]
    assert values == [0.375, 0.4375, 0.5, 0.625, 0.6875, 0.75]
    direct = _derived(panel, (_renal_function(values), DOSE), {})
    expected = runner.run("hctz", direct, {"dose": 25.0}, {"egfr": 6})
    rows = np.asarray(result.curves["[Cve_hctz]"])
    assert rows.shape == (6, len(result.x))
    assert np.array_equal(rows, np.asarray(expected.curves["[Cve_hctz]"]))
    assert not np.array_equal(rows[0], rows[-1])


@pytest.mark.parametrize(
    ("knobs", "derived", "named"),
    [
        (
            (CRCL, DOSE),
            {"f_renal_function": "min(clearance / 110, 1)"},
            ["'f_renal_function'", "['clearance']", "'crcl'"],
        ),
        (
            (CRCL, DOSE),
            {"f_renal_function": "min(crcl / 110, "},
            ["'f_renal_function'", "no math", "'min(crcl / 110, '"],
        ),
        ((CRCL, DOSE), {}, ["'crcl'", "sets no target"]),
        (
            (CRCL, DOSE, CRCL.model_copy(update={"id": "unused"})),
            RENAL_FUNCTION,
            ["'unused'", "sets no target"],
        ),
    ],
)
def test_a_derived_target_the_service_cannot_run_is_refused_at_startup(
    project: Project,
    panel: Panel,
    monkeypatch: pytest.MonkeyPatch,
    knobs: tuple[Knob, ...],
    derived: dict[str, str],
    named: list[str],
) -> None:
    """A panel which is in no app is checked as well, in every variant."""
    bogus = _derived(panel, knobs, derived).model_copy(update={"id": "derived"})
    broken = project.replace(panels={**project.panels, bogus.id: bogus})
    monkeypatch.setattr(runner_module, "load_project", lambda reference: broken)
    with pytest.raises(ValueError) as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    message = str(error.value)
    for one in ("'hctz'", "variant 'default'", "panel 'derived'", *named):
        assert one in message


@pytest.mark.parametrize(
    ("knobs", "derived", "preinit", "named"),
    [
        (
            (CRCL, DOSE),
            {"f_renal_fucntion": "min(crcl / 110, 1)"},
            None,
            ["'f_renal_fucntion', a derived target"],
        ),
        (
            (_renal_function([1.0]).model_copy(update={"target": "f_renal"}), DOSE),
            {},
            None,
            ["'f_renal', the target of the knob 'egfr'"],
        ),
        (
            (CRCL, DOSE),
            RENAL_FUNCTION,
            {"PODOSE_hctz": 0.0, "body_weight": 75.0},
            ["'body_weight', a target of the preinit"],
        ),
    ],
)
def test_a_target_which_is_no_entity_of_the_model_is_refused_at_startup(
    project: Project,
    panel: Panel,
    monkeypatch: pytest.MonkeyPatch,
    knobs: tuple[Knob, ...],
    derived: dict[str, str],
    preinit: dict[str, float] | None,
    named: list[str],
) -> None:
    """A typo of a target is refused once, from the SBML read for the panels."""
    bogus = _derived(panel, knobs, derived).model_copy(update={"id": "typo"})
    if preinit is not None:
        bogus = bogus.model_copy(
            update={
                "simulation": bogus.simulation.model_copy(update={"preinit": preinit})
            }
        )
    broken = project.replace(panels={**project.panels, bogus.id: bogus})
    monkeypatch.setattr(runner_module, "load_project", lambda reference: broken)
    with pytest.raises(ValueError, match="no entity of the model") as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    message = str(error.value)
    for one in ("'hctz'", "variant 'default'", "panel 'typo'", *named):
        assert one in message


def test_a_concentration_of_a_species_is_a_target_of_a_knob(
    project: Project, panel: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[S]` is the concentration of a species `S`, which a change may set."""
    concentration = _renal_function([0.0]).model_copy(
        update={"id": "initial", "target": "[Cve_hctz]"}
    )
    served = _derived(panel, (concentration, DOSE), {}).model_copy(
        update={"id": "concentration"}
    )
    fine = project.replace(panels={**project.panels, served.id: served})
    monkeypatch.setattr(runner_module, "load_project", lambda reference: fine)
    assert ModelRegistry.of({"hctz": HCTZ_REFERENCE}).panel("hctz", served.id) == served
    entities = fine.model_entities()
    assert entities is not None
    assert {"[Cve_hctz]", "Cve_hctz", "BW", "f_renal_function", "Vve"} <= entities
    assert "[BW]" not in entities


def test_a_panel_of_derived_targets_which_read_its_knobs_is_served(
    project: Project, panel: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = _derived(panel, (CRCL, DOSE), RENAL_FUNCTION)
    served = served.model_copy(update={"id": "derived"})
    fine = project.replace(panels={**project.panels, served.id: served})
    monkeypatch.setattr(runner_module, "load_project", lambda reference: fine)
    registry = ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    assert registry.panel("hctz", "derived") == served


def test_the_registry_reads_the_panels_of_the_annotation(
    registry: ModelRegistry,
) -> None:
    assert registry.panel("hctz", "renal_plasma").knobs[0].id == "egfr"


def test_the_registry_loads_the_project_of_a_reference() -> None:
    registry = ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    assert registry.ids() == ("hctz",)
    entry = registry.entry("hctz")
    assert entry.id == "hctz"
    assert entry.project.name == "hctz"
    # the project of the example has no title, so its name is the one
    assert entry.title == "hctz"


def test_an_app_page_names_panels_of_the_model(project: Project) -> None:
    """Every panel id of every page of the app is a panel the model declares."""
    app = project.app
    assert app is not None
    declared = set(project.panels_of())
    assert {panel for page in app.pages for panel in page.panels} <= declared


def test_a_page_naming_an_unknown_panel_is_refused_when_the_registry_is_built(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panels of a model are read from its SBML, so the check is the registry's.

    A project is imported without its SBML, which is why the app itself cannot
    check it: the registry of a service reads it once, before any request.
    """
    broken = project.replace(
        app=App(
            title="Hydrochlorothiazide",
            pages=(AppPage(id="renal", title="Renal", panels=("tubular",)),),
        )
    )
    monkeypatch.setattr(runner_module, "load_project", lambda reference: broken)
    with pytest.raises(ValueError, match="tubular") as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    message = str(error.value)
    assert "hctz" in message
    assert "renal" in message


def test_a_variant_of_an_app_which_the_project_has_not_is_refused(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = project.replace(
        app=App(
            title="Hydrochlorothiazide",
            pages=(AppPage(id="renal", title="Renal", panels=("renal_plasma",)),),
            variants=(AppVariant(id="impaired", label="renal impairment"),),
        )
    )
    monkeypatch.setattr(runner_module, "load_project", lambda reference: broken)
    with pytest.raises(ValueError, match="impaired"):
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})


def test_a_page_whose_panels_declare_one_knob_differently_is_refused(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page shows one control for a knob id its panels share, so it is one knob.

    The same id on another page is another control and may be declared
    otherwise, which the HCTZ app does with the dose.
    """
    panels = project.panels_of()
    plasma = panels["renal_plasma"]
    urine = panels["renal_urine"]
    stepped = urine.model_copy(
        update={"knobs": (EGFR_STAGES, DOSE.model_copy(update={"default": 50.0}))}
    )
    broken = project.replace(
        panels={**project.panels, "renal_urine": stepped},
        app=App(
            title="Hydrochlorothiazide",
            pages=(AppPage(id="renal", title="Renal", panels=(plasma.id, urine.id)),),
        ),
    )
    monkeypatch.setattr(runner_module, "load_project", lambda reference: broken)
    with pytest.raises(ValueError, match="differently") as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    message = str(error.value)
    for named in ("'hctz'", "'renal'", "'dose'", "'renal_plasma'", "'renal_urine'"):
        assert named in message
    assert "(default)" in message
    # the documentation quotes this refusal as the code writes it
    assert f"`{message}`" in (ROOT / "docs" / "serve.md").read_text(encoding="utf-8")
    # a label which differs is a field which differs as well
    relabelled = urine.model_copy(
        update={"knobs": (EGFR_STAGES.model_copy(update={"label": "eGFR"}), DOSE)}
    )
    monkeypatch.setattr(
        runner_module,
        "load_project",
        lambda reference: broken.replace(
            panels={**project.panels, "renal_urine": relabelled}
        ),
    )
    with pytest.raises(ValueError, match=r"'egfr' differently \(label\)"):
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})


@pytest.mark.parametrize(
    ("values", "named"),
    [
        ({"weight": 75.0}, ["'weight'", "no panel of the page declares"]),
        ({"dose": 30.0}, ["'dose' to 30.0", "none of the choices", "'renal_plasma'"]),
        ({"egfr": 0.4, "dose": 5.0}, None),
    ],
)
def test_a_scenario_of_a_knob_no_panel_of_its_page_can_take_is_refused(
    project: Project,
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, float],
    named: list[str] | None,
) -> None:
    """A scenario sets knobs of its page to values of these knobs."""
    app = project.app
    assert app is not None
    scenario = Scenario(id="patient", label="a patient", values=values)
    with_scenario = project.replace(
        app=app.model_copy(
            update={
                "pages": (
                    app.pages[0].model_copy(update={"scenarios": (scenario,)}),
                    *app.pages[1:],
                )
            }
        )
    )
    monkeypatch.setattr(runner_module, "load_project", lambda reference: with_scenario)
    if named is None:
        registry = ModelRegistry.of({"hctz": HCTZ_REFERENCE})
        assert registry.entry("hctz").project.app == with_scenario.app
        return
    with pytest.raises(ValueError) as error:
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    message = str(error.value)
    for one in ("'hctz'", "page 'renal'", "scenario 'patient'", *named):
        assert one in message


def test_a_scenario_value_outside_the_bounds_of_a_range_knob_is_refused(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value of a knob is a value of its declaration on the page of the scenario.

    The dose of the page `dose` is a range of 12.5 to 50 mg: 60 mg is no value
    of it, and neither is 5 mg, which is a choice of the dose of the page
    `renal`, another knob of the same id.
    """
    app = project.app
    assert app is not None
    for value in (60.0, 5.0):
        problem = re.escape(f"to {value}, which is outside the bounds [12.5, 50.0]")
        scenario = Scenario(id="high", label="a dose", values={"dose": value})
        with_scenario = project.replace(
            app=app.model_copy(
                update={
                    "pages": (
                        app.pages[0],
                        app.pages[1].model_copy(update={"scenarios": (scenario,)}),
                    )
                }
            )
        )
        monkeypatch.setattr(
            runner_module, "load_project", lambda reference, p=with_scenario: p
        )
        with pytest.raises(ValueError, match=problem) as error:
            ModelRegistry.of({"hctz": HCTZ_REFERENCE})
        # the first panel of the page which declares the knob
        assert "panel 'dose_plasma'" in str(error.value)
    # a value within the bounds which is no point of the grid is a value
    scenario = Scenario(id="mid", label="a dose", values={"dose": 33.3})
    monkeypatch.setattr(
        runner_module,
        "load_project",
        lambda reference: project.replace(
            app=app.model_copy(
                update={
                    "pages": (
                        app.pages[0],
                        app.pages[1].model_copy(update={"scenarios": (scenario,)}),
                    )
                }
            )
        ),
    )
    assert ModelRegistry.of({"hctz": HCTZ_REFERENCE}).ids() == ("hctz",)


def test_a_knob_a_scenario_sets_is_shared_even_when_one_panel_declares_it(
    project: Project,
) -> None:
    app = project.app
    assert app is not None
    plasma = project.panels_of()["dose_plasma"]
    scenario = Scenario(id="low", label="12.5 mg", values={"dose": 12.5})
    page = app.pages[1].model_copy(update={"panels": ("dose_plasma",)})
    single = project.replace(
        app=app.model_copy(
            update={"pages": (page.model_copy(update={"scenarios": (scenario,)}),)}
        )
    )
    entry = ModelEntry(id="hctz", project=single, title="Hydrochlorothiazide")
    shared = entry.shared()["dose"]
    assert [one.knob.id for one in shared] == ["dose"]
    assert shared[0].knob == plasma.knobs[0]
    assert shared[0].panels == (plasma,)
    # without the scenario a page of one panel shares nothing
    lone = project.replace(app=app.model_copy(update={"pages": (page,)}))
    assert ModelEntry(id="hctz", project=lone, title="x").shared() == {"dose": ()}


def test_one_knob_id_declared_alike_on_a_page_and_otherwise_on_another_is_served(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the HCTZ app declares `dose` as a choice on `renal` and as a range on `dose`
    app = project.app
    assert app is not None
    panels = project.panels_of()
    kinds = {
        page.id: {
            knob.kind
            for one in page.panels
            for knob in panels[one].knobs
            if knob.id == "dose"
        }
        for page in app.pages
    }
    assert kinds == {"renal": {"choice"}, "dose": {"range"}, "patient": {"choice"}}
    monkeypatch.setattr(runner_module, "load_project", lambda reference: project)
    assert ModelRegistry.of({"hctz": HCTZ_REFERENCE}).ids() == ("hctz",)


def test_the_registry_takes_the_app_of_a_model_as_it_is_declared(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "load_project", lambda reference: project)
    registry = ModelRegistry.of({"hctz": HCTZ_REFERENCE})
    assert registry.entry("hctz").project.app == project.app


def test_an_unknown_model_raises_unknown_model(registry: ModelRegistry) -> None:
    with pytest.raises(UnknownModel) as error:
        registry.entry("pkdb_models.models.hctz:PROJECT")
    assert "pkdb_models.models.hctz:PROJECT" in str(error.value)


def test_an_unknown_panel_raises_unknown_panel(registry: ModelRegistry) -> None:
    with pytest.raises(UnknownPanel) as error:
        registry.panel("hctz", "tubular")
    assert "tubular" in str(error.value)
    with pytest.raises(UnknownModel):
        registry.panel("renal", "renal")


def test_a_point_which_does_not_integrate_is_flagged_and_not_raised(
    runner: PanelRunner, panel: Panel
) -> None:
    # a renal function below its bounds, for which the model does not integrate
    result = runner.run("hctz", panel, {"egfr": -1.0, "dose": 25.0}, {})
    assert result.failed == (0,)
    assert math.isnan(result.curves["[Cve_hctz]"][0][-1])


def test_a_failing_point_of_a_grid_is_flagged_by_its_flat_index(
    runner: PanelRunner, panel: Panel
) -> None:
    # a renal function below its bounds as the first choice of the knob, for
    # which the model does not integrate
    stages = (Choice(label="broken", value=-1.0), *EGFR_STAGES.choices)
    broken = panel.model_copy(
        update={"knobs": (EGFR_STAGES.model_copy(update={"choices": stages}), DOSE)}
    )
    result = runner.run("hctz", broken, {"dose": 25.0}, {"egfr": 4})
    assert result.failed == (0,)
    assert all(math.isnan(value) for value in result.curves["[Cve_hctz]"][0])
    assert not any(math.isnan(value) for value in result.curves["[Cve_hctz]"][2])


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path]]:
    """Record every load of a model, which the runner still really loads."""
    real = runner_module._load
    recorded: list[tuple[str, Path]] = []

    def record(entry: ModelEntry, path: Path) -> runner_module._Warm:
        recorded.append((entry.id, path))
        return real(entry, path)

    monkeypatch.setattr(runner_module, "_load", record)
    return recorded


def test_the_default_variant_and_its_name_are_one_warm_model(
    registry: ModelRegistry, panel: Panel, loads: list[tuple[str, Path]]
) -> None:
    """A variant which is not named and its name are one model, not two."""
    named = panel.model_copy(
        update={
            "simulation": panel.simulation.model_copy(update={"variant": "default"})
        }
    )
    runner = PanelRunner(registry)
    first = runner.run("hctz", panel, {"egfr": 1.0}, {})
    second = runner.run("hctz", named, {"egfr": 1.0}, {})
    assert len(loads) == 1
    assert second.curves == first.curves


def test_two_threads_which_ask_for_a_cold_model_load_it_once(
    registry: ModelRegistry, panel: Panel, loads: list[tuple[str, Path]]
) -> None:
    runner = PanelRunner(registry)
    start = threading.Barrier(2)
    results: list[PanelResult] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            start.wait(TIMEOUT)
            results.append(runner.run("hctz", panel, {"egfr": 1.0}, {}))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert errors == []
    assert len(loads) == 1
    assert results[0].curves == results[1].curves


def test_a_warm_model_does_not_wait_for_a_model_which_is_loading(
    registry: ModelRegistry, panel: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A load holds the lock of its own key, not one of the whole runner.

    The project is served under a second id as well, so the runner has two
    warm models of one project. Parking the load of the second one is the
    stimulus of the test and no stand in for what is tested: the model is
    really loaded once the test releases it.
    """
    served = registry.entry("hctz")
    both = ModelRegistry(
        {
            "hctz": served,
            "slow": ModelEntry(
                id="slow", project=served.project, title="the same model"
            ),
        }
    )
    runner = PanelRunner(both)
    expected = runner.run("hctz", panel, {"egfr": 1.0}, {})
    real = runner_module._load
    loading, release, read = (threading.Event() for _ in range(3))

    def parked(entry: ModelEntry, path: Path) -> runner_module._Warm:
        loading.set()
        release.wait(PARKED)
        return real(entry, path)

    monkeypatch.setattr(runner_module, "_load", parked)
    results: list[PanelResult] = []
    errors: list[BaseException] = []

    def load_the_cold_model() -> None:
        try:
            results.append(runner.run("slow", panel, {"egfr": 1.0}, {}))
        except BaseException as error:
            errors.append(error)

    def read_the_warm_model() -> None:
        try:
            runner.run("hctz", panel, {"egfr": 1.0}, {})
            read.set()
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=load_the_cold_model),
        threading.Thread(target=read_the_warm_model),
    ]
    try:
        threads[0].start()
        assert loading.wait(TIMEOUT), "the load of the cold model did not start"
        threads[1].start()
        assert read.wait(TIMEOUT), "the warm model waited for the one which loads"
    finally:
        release.set()
        for thread in threads:
            thread.join(TIMEOUT)
    assert errors == []
    assert [thread.is_alive() for thread in threads] == [False, False]
    assert results[0].curves == expected.curves


def test_concurrent_runs_of_one_model_do_not_interleave(
    registry: ModelRegistry, panel: Panel
) -> None:
    # a runner which shares no run, so both threads run the scan
    runner = PanelRunner(registry, max_run_bytes=0)
    serial = runner.run("hctz", panel, {"dose": 25.0}, {"egfr": 5})
    results: list[PanelResult] = []
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def work() -> None:
        try:
            start.wait()
            results.append(runner.run("hctz", panel, {"dose": 25.0}, {"egfr": 5}))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(results) == 2
    for result in results:
        assert result.failed == serial.failed
        assert result.curves == serial.curves
