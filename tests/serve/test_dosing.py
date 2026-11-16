"""The dosing protocols of a panel: a dose repeated at an interval, as changes."""

import numpy as np
import pytest
from pydantic import ValidationError

from examples.panels.hctz import PROJECT
from sbmlsim.engine import Engine
from sbmlsim.model.model_sbml import load_model
from sbmlsim.panel import (
    MAX_DOSES,
    Choice,
    Knob,
    Panel,
    PanelCurve,
    PanelDosing,
    PanelSimulation,
    dose_times,
    panel_from_payload,
    panel_payload,
)
from sbmlsim.panel_sbml import panels_from_sbml, panels_to_annotation
from sbmlsim.serve.runner import ModelEntry, ModelRegistry, PanelRunner, curve_factor
from sbmlsim.simulation import Change, Simulation
from sbmlsim.simulator import Simulator

#: three days in the minutes of the model
END = 3 * 24 * 60.0

DOSE = Knob(
    id="dose",
    label="dose",
    target="PODOSE_hctz",
    unit="mg",
    kind="range",
    min=12.5,
    max=50.0,
    steps=4,
    default=25.0,
)
INTERVAL = Knob(
    id="interval",
    label="dosing",
    unit="hr",
    kind="choice",
    choices=(
        Choice(label="single dose", value=0.0),
        Choice(label="once daily", value=24.0),
        Choice(label="twice daily", value=12.0),
    ),
    default=24.0,
)
DAYS = Knob(
    id="days",
    label="days of dosing",
    unit="day",
    kind="range",
    min=1.0,
    max=3.0,
    steps=3,
    default=2.0,
)
PANEL = Panel(
    id="dosing_test",
    title="Repeated oral doses",
    knobs=(DOSE, INTERVAL, DAYS),
    curves=(PanelCurve(id="[Cve_hctz]", label="plasma", unit="uM"),),
    simulation=PanelSimulation(
        end=END,
        points=72,
        preinit={"BW": 75.0},
        dosing=(
            PanelDosing(
                target="PODOSE_hctz", dose="dose", interval="interval", duration="days"
            ),
        ),
    ),
)


@pytest.fixture(scope="module")
def runner() -> PanelRunner:
    registry = ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=PROJECT, title="Hydrochlorothiazide")}
    )
    return PanelRunner(registry)


def test_the_times_of_the_doses_after_the_first_one() -> None:
    assert dose_times(0.0, None, 100.0) == []
    assert dose_times(24.0, None, 72.0) == [24.0, 48.0]
    assert dose_times(24.0, 48.0, 100.0) == [24.0]
    assert dose_times(12.0, 48.0, 30.0) == [12.0, 24.0]
    assert len(dose_times(1e-9, None, 1.0)) == MAX_DOSES


def test_a_dosing_reads_knobs_of_the_panel() -> None:
    with pytest.raises(ValidationError, match="does not declare"):
        Panel.model_validate(
            {**PANEL.model_dump(), "knobs": [DOSE.model_dump(), DAYS.model_dump()]}
        )
    with pytest.raises(ValidationError, match="sets 'BW'"):
        Panel.model_validate(
            {
                **PANEL.model_dump(),
                "knobs": [
                    DOSE.model_dump(),
                    {**INTERVAL.model_dump(), "target": "BW"},
                    DAYS.model_dump(),
                ],
            }
        )
    with pytest.raises(ValidationError, match="first dose"):
        Panel.model_validate(
            {
                **PANEL.model_dump(),
                "knobs": [
                    {**DOSE.model_dump(), "target": "IVDOSE_hctz"},
                    INTERVAL.model_dump(),
                    DAYS.model_dump(),
                ],
            }
        )


def test_a_dosing_travels_in_the_payload_and_the_annotation() -> None:
    assert panel_from_payload(panel_payload(PANEL)) == PANEL
    sbml = PROJECT.variant_path(None).read_text(encoding="utf-8")
    import libsbml

    doc = libsbml.readSBMLFromString(sbml)
    doc.getModel().appendAnnotation(panels_to_annotation([PANEL]))
    panels, problems = panels_from_sbml(libsbml.writeSBMLToString(doc))
    assert problems == ()
    assert panels[PANEL.id] == PANEL


def test_the_doses_are_changes_of_the_simulation(runner: PanelRunner) -> None:
    result = runner.run(
        "hctz", PANEL, {"dose": 25.0, "interval": 12.0, "days": 2.0}, {}
    )
    # by hand: the first dose at the start, then 25 mg at 12 h, 24 h and 36 h
    model = load_model(PROJECT.variant_path(None), Engine.ROADRUNNER)
    simulation = Simulation(
        end=END,
        preinit_changes={"BW": 75.0, "PODOSE_hctz": model.Q(25.0, "mg")},
        changes=[
            Change(
                times=[720.0, 1440.0, 2160.0],
                values={"PODOSE_hctz": "PODOSE_hctz + 25.0"},
            )
        ],
        times=np.linspace(0.0, END, 73).tolist(),
    )
    expected = Simulator(**PROJECT.simulator.integrator_settings()).run(
        model, simulation, time=np.linspace(0.0, END, 73), keep=["[Cve_hctz]"]
    ).ds["[Cve_hctz]"].to_numpy() * curve_factor(model, PANEL.curves[0])
    curve = np.asarray(result.curves["[Cve_hctz]"][0])
    assert curve.shape == expected.shape
    assert np.array_equal(curve, expected)
    single = runner.run("hctz", PANEL, {"dose": 25.0, "interval": 0.0}, {})
    assert max(single.curves["[Cve_hctz]"][0]) < curve.max()


def test_a_grid_over_the_interval_has_the_doses_of_each_point(
    runner: PanelRunner,
) -> None:
    grid = runner.run("hctz", PANEL, {"dose": 25.0, "days": 2.0}, {"interval": 3})
    assert [axis.id for axis in grid.dims] == ["interval"]
    for i, interval in enumerate((0.0, 24.0, 12.0)):
        point = runner.run(
            "hctz", PANEL, {"dose": 25.0, "days": 2.0, "interval": interval}, {}
        )
        assert grid.curves["[Cve_hctz]"][i] == point.curves["[Cve_hctz]"][0]


def test_a_grid_over_the_dose_doses_the_dose_of_each_point(runner: PanelRunner) -> None:
    grid = runner.run("hctz", PANEL, {"interval": 24.0, "days": 3.0}, {"dose": 4})
    for i, dose in enumerate(DOSE.values(4)):
        point = runner.run(
            "hctz", PANEL, {"dose": dose, "interval": 24.0, "days": 3.0}, {}
        )
        assert grid.curves["[Cve_hctz]"][i] == point.curves["[Cve_hctz]"][0]
