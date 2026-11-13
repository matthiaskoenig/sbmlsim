"""The registry of the served models and the runner of their panels.

The model, the panel and the registry of the tests are the fixtures of
`tests/serve/conftest.py`; the runner of them is built here, as the tests of
the API build the application instead.
"""

import math
import threading
from pathlib import Path

import pytest
from pint.errors import UndefinedUnitError

from examples.panels.hctz import DOSE, EGFR_STAGES, END, URINE
from sbmlsim.panel import App, AppPage, AppVariant, Choice, Panel, PanelCurve
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
    row = result.curves["Cve_hctz"]
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
    assert len(result.curves["Cve_hctz"]) == 3
    assert all(len(row) == len(result.x) for row in result.curves["Cve_hctz"])
    # a lower renal function clears less, so more of the dose is left in the
    # plasma at the end of the day
    ends = [row[-1] for row in result.curves["Cve_hctz"]]
    assert ends == sorted(ends, reverse=True)


def test_a_grid_of_two_knobs_gives_a_row_per_point_in_c_order(
    runner: PanelRunner, panel: Panel
) -> None:
    result = runner.run("hctz", panel, {}, {"egfr": 2, "dose": 3})
    assert [axis.id for axis in result.dims] == ["egfr", "dose"]
    assert result.dims[0].values == EGFR_STAGES.values(None)
    assert result.dims[1].values == DOSE.values(None)
    rows = result.curves["Cve_hctz"]
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
    assert result.units["Cve_hctz"] == "umol"


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
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    """`units` answers without a run what a run delivers, see the module."""
    units = runner.units(registry.entry("hctz"), panel)
    assert units.x is not None
    assert units.x.text == "min"
    urine = units.series[URINE.id]
    assert urine is not None
    assert urine.text == "umol"
    result = runner.run("hctz", panel, {"egfr": 1.0, "dose": 25.0}, {})
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
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    """A panel may declare a unit id of its model, which pint does not know."""
    # `KI__min` is the minute and `substance` the mole of the HCTZ model
    with pytest.raises(UndefinedUnitError):
        ureg.Unit("KI__min")
    urine = PanelCurve(id="Aurine_hctz", label="urine", unit="substance")
    declared = panel.model_copy(
        update={
            "curves": (panel.curves[0], urine),
            "simulation": panel.simulation.model_copy(update={"x_unit": "KI__min"}),
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
            "data": (URINE.model_copy(update={"curve": "Cve_hctz"}),),
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


def test_a_unit_of_a_series_which_is_no_unit_is_refused(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    """It is refused where it is resolved, before a series is read."""
    bogus = panel.model_copy(
        update={"data": (URINE.model_copy(update={"unit": "not_a_unit"}),)}
    )
    with pytest.raises(ValueError) as error:
        runner.units(registry.entry("hctz"), bogus)
    assert "not_a_unit" in str(error.value)
    assert URINE.id in str(error.value)


def test_a_curve_without_a_unit_carries_the_one_of_the_model(
    runner: PanelRunner, registry: ModelRegistry, panel: Panel
) -> None:
    curve = PanelCurve(id="Cve_hctz", label="hydrochlorothiazide plasma")
    without = panel.model_copy(update={"curves": (curve,), "data": ()})
    declared = runner.run("hctz", panel, {"egfr": 1.0, "dose": 25.0}, {})
    result = runner.run("hctz", without, {"egfr": 1.0, "dose": 25.0}, {})
    assert result.units["Cve_hctz"] == "mmol"
    # the study data under such a curve is delivered in that unit as well
    under = without.model_copy(
        update={"data": (URINE.model_copy(update={"curve": "Cve_hctz"}),)}
    )
    series = runner.units(registry.entry("hctz"), under).series
    assert series[URINE.id] == DeliveredUnit(text="mmol", unit=ureg.Unit("mmol"))
    # the declared unit converts the values of the model, it does not relabel them
    assert result.curves["Cve_hctz"][0][-1] * 1000.0 == pytest.approx(
        declared.curves["Cve_hctz"][0][-1]
    )


def test_a_knob_value_overrides_the_preinit_of_the_panel(
    runner: PanelRunner, panel: Panel
) -> None:
    # the panel pre-initializes the dose to zero, the knob sets it
    low = runner.run("hctz", panel, {"egfr": 1.0, "dose": 12.5}, {})
    high = runner.run("hctz", panel, {"egfr": 1.0, "dose": 50.0}, {})
    assert 0.0 < max(low.curves["Cve_hctz"][0]) < max(high.curves["Cve_hctz"][0])


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


def test_the_registry_reads_the_panels_of_the_annotation(
    registry: ModelRegistry,
) -> None:
    assert registry.panel("hctz", "renal").knobs[0].id == "egfr"


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
            pages=(AppPage(id="renal", title="Renal", panels=("renal",)),),
            variants=(AppVariant(id="impaired", label="renal impairment"),),
        )
    )
    monkeypatch.setattr(runner_module, "load_project", lambda reference: broken)
    with pytest.raises(ValueError, match="impaired"):
        ModelRegistry.of({"hctz": HCTZ_REFERENCE})


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
    assert math.isnan(result.curves["Cve_hctz"][0][-1])


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
    assert all(math.isnan(value) for value in result.curves["Cve_hctz"][0])
    assert not any(math.isnan(value) for value in result.curves["Cve_hctz"][2])


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
    runner: PanelRunner, panel: Panel
) -> None:
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
