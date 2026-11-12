"""The registry of the served models and the runner of their panels.

The panel of the tests is built here and bound to the project of the HCTZ
example with `Project.replace(panels=...)`, as a model whose SBML carries the
annotation of its panels is not part of this task.

The time unit of the HCTZ model is minutes, which is the unit the `end` of a
`PanelSimulation` and the x of a result are in.
"""

import math
import threading

import pytest

from examples.hctz_fitting import MODEL_PATH
from examples.hctz_fitting.project import PROJECT
from sbmlsim.panel import Choice, Knob, Panel, PanelCurve, PanelSimulation
from sbmlsim.project import Project
from sbmlsim.serve.runner import (
    ModelEntry,
    ModelRegistry,
    PanelResult,
    PanelRunner,
    UnknownModel,
    UnknownPanel,
)

#: the reference of the project of the HCTZ example
HCTZ_REFERENCE = "examples.hctz_fitting.project:PROJECT"
#: one day in the time unit of the model
END = 24 * 60.0
#: the output points of the panel, fewer than a real panel so the tests are quick
POINTS = 50

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
    choices=(
        Choice(label="12.5 mg", value=12.5),
        Choice(label="25 mg", value=25.0),
        Choice(label="50 mg", value=50.0),
    ),
    default=25.0,
)


@pytest.fixture(scope="module")
def panel() -> Panel:
    return Panel(
        id="renal",
        title="Hydrochlorothiazide in renal impairment",
        subtitle="the plasma concentration over the renal function",
        x_label="time (min)",
        knobs=(EGFR, DOSE),
        curves=(
            PanelCurve(id="Cve_hctz", label="hydrochlorothiazide plasma", unit="umol"),
        ),
        # the panel pre-initializes the dose to zero, which the knob overrides
        simulation=PanelSimulation(
            end=END, points=POINTS, preinit={"PODOSE_hctz": 0.0, "BW": 75.0}
        ),
    )


@pytest.fixture(scope="module")
def project(panel: Panel) -> Project:
    """The project of the HCTZ example with the panel of the tests."""
    return PROJECT.replace(variants={"default": MODEL_PATH}, panels={"renal": panel})


@pytest.fixture(scope="module")
def registry(project: Project) -> ModelRegistry:
    return ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=project, title="Hydrochlorothiazide")}
    )


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
    result = runner.run("hctz", panel, {"dose": 25.0}, {"egfr": 5})
    assert [axis.id for axis in result.dims] == ["egfr"]
    assert len(result.dims[0].values) == 5
    assert result.dims[0].values == EGFR.values(5)
    assert result.dims[0].label == "renal function"
    assert result.dims[0].unit == "dimensionless"
    assert len(result.curves["Cve_hctz"]) == 5
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
    assert result.dims[1].values == DOSE.values(None)
    rows = result.curves["Cve_hctz"]
    assert len(rows) == 6
    # the rows of one renal function follow each other, the dose fastest
    assert max(rows[0]) < max(rows[1]) < max(rows[2])


def test_the_curves_carry_the_units_of_the_model(
    runner: PanelRunner, panel: Panel
) -> None:
    result = runner.run("hctz", panel, {"egfr": 1.0, "dose": 25.0}, {})
    # the time unit of the HCTZ model, the unit of the `end` of the panel
    assert result.units["x"] == "min"
    assert result.units["Cve_hctz"] == "umol"


def test_a_curve_without_a_unit_carries_the_one_of_the_model(
    runner: PanelRunner, panel: Panel
) -> None:
    curve = PanelCurve(id="Cve_hctz", label="hydrochlorothiazide plasma")
    without = panel.model_copy(update={"curves": (curve,)})
    declared = runner.run("hctz", panel, {"egfr": 1.0, "dose": 25.0}, {})
    result = runner.run("hctz", without, {"egfr": 1.0, "dose": 25.0}, {})
    assert result.units["Cve_hctz"] == "mmol"
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
    broken = panel.model_copy(
        update={"knobs": (EGFR.model_copy(update={"min": -1.0}), DOSE)}
    )
    result = runner.run("hctz", broken, {"dose": 25.0}, {"egfr": 3})
    assert result.failed == (0,)
    assert all(math.isnan(value) for value in result.curves["Cve_hctz"][0])
    assert not any(math.isnan(value) for value in result.curves["Cve_hctz"][2])


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
