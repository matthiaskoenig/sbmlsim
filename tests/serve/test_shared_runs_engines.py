"""The runs which the panels of one simulation share.

A page of several panels of one model asks for every panel at one state of its
knobs at once, and the runner runs the scan of that state once for every panel
which shares the simulation (`sbmlsim.serve.runner`). The tests compare every
answer with the one of a runner which shares nothing (`max_runs=0`), bitwise,
on the page `patient` of the HCTZ demo, whose two panels share a simulation,
and on a page whose two panels do not, and count the runs of the simulator.
The name ends in `_engines` so that the job `amici engine` of the continuous
integration runs the case of AMICI, which runs alone and shares nothing.
"""

import functools
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from examples.panels.hctz import (
    CRCL,
    HCTZ_APP,
    HCTZ_PATIENT_PLASMA,
    HCTZ_PATIENT_URINE,
    HCTZ_RENAL_PLASMA,
    HCTZ_RENAL_URINE,
    PATIENT_PAGE,
    PATIENT_SIMULATION,
    POINTS,
    PROJECT,
    SIMULATION,
)
from sbmlsim.engine import Engine
from sbmlsim.panel import AppPage, Panel, PanelCurve
from sbmlsim.project import Project, SimulatorSettings
from sbmlsim.serve import app as app_module
from sbmlsim.serve.app import create_app
from sbmlsim.serve.runner import (
    SHARED_ENGINES,
    ModelEntry,
    ModelRegistry,
    PanelResult,
    PanelRunner,
)
from sbmlsim.simulator import Simulator
from tests.engines import OTHER_ENGINES, require

#: the seconds a wait which must succeed gives up after, i.e. a failed test
TIMEOUT = 60.0
#: the states of the page `patient` the shell asks for: the default, gridded
#: over the creatinine clearance, and the patient of a severe impairment,
#: gridded over the dose, and a point of a moderate one
STATES: tuple[tuple[dict[str, float], dict[str, int]], ...] = (
    ({"dose": 25.0}, {"crcl": 8}),
    ({"crcl": 20.0}, {"dose": 4}),
    ({"crcl": 45.0, "dose": 12.5}, {}),
)
#: a renal function below its bounds, for which the model does not integrate
BROKEN: dict[str, float] = {"crcl": -110.0, "dose": 25.0}

#: the urine of a patient at half the time points: a simulation of its own, so
#: the page `apart` has two panels which share none
URINE_APART = HCTZ_PATIENT_URINE.model_copy(
    update={
        "id": "patient_urine_apart",
        "simulation": PATIENT_SIMULATION.model_copy(update={"points": POINTS // 2}),
    }
)
#: the urine of a patient whose clearance knob has another id and whose
#: formula reads it: another panel, another simulation, the same scan
URINE_RENAMED = HCTZ_PATIENT_URINE.model_copy(
    update={
        "id": "patient_urine_renamed",
        "knobs": (
            CRCL.model_copy(update={"id": "clearance"}),
            *HCTZ_PATIENT_URINE.knobs[1:],
        ),
        "data": (),
        "simulation": PATIENT_SIMULATION.model_copy(
            update={"derived": {"f_renal_function": "min(clearance / 110, 1)"}}
        ),
    }
)
#: a panel of the simulation of the page `patient` whose curve no model has
UNSELECTABLE = HCTZ_PATIENT_PLASMA.model_copy(
    update={
        "id": "patient_unselectable",
        "curves": (PanelCurve(id="no_such_selection", label="nothing"),),
        "y_min": None,
        "y_max": None,
    }
)
#: the simulation of the renal panels without a pre-initialization of its own,
#: so the order of the values of a state is the order of the knobs which set it
BARE = SIMULATION.model_copy(update={"preinit": {}})
#: the renal plasma and urine of that simulation, the urine with its knobs
#: declared in the other order
ORDERED = HCTZ_RENAL_PLASMA.model_copy(
    update={"id": "renal_plasma_bare", "simulation": BARE}
)
URINE_REORDERED = HCTZ_RENAL_URINE.model_copy(
    update={
        "id": "renal_urine_reordered",
        "knobs": tuple(reversed(HCTZ_RENAL_URINE.knobs)),
        "data": (),
        "simulation": BARE,
    }
)
#: the body weight of a patient on the axis of minutes: another simulation, the
#: scan of the page `patient`, and a curve none of its panels draws
WEIGHT = HCTZ_PATIENT_PLASMA.model_copy(
    update={
        "id": "patient_weight",
        "curves": (PanelCurve(id="BW", label="body weight"),),
        "y_min": None,
        "y_max": None,
        "x_max": None,
        "simulation": PATIENT_SIMULATION.model_copy(update={"x_unit": "min"}),
    }
)
#: a page of two panels of one model which share no simulation
APART_PAGE = AppPage(
    id="apart",
    title="Apart",
    panels=("patient_plasma", "patient_urine_apart"),
)
#: the demo with the panels above and the page `apart`
SHARED_PROJECT: Project = PROJECT.replace(
    panels={
        **PROJECT.panels,
        URINE_APART.id: URINE_APART,
        URINE_RENAMED.id: URINE_RENAMED,
        UNSELECTABLE.id: UNSELECTABLE,
        ORDERED.id: ORDERED,
        URINE_REORDERED.id: URINE_REORDERED,
        WEIGHT.id: WEIGHT,
    },
    app=HCTZ_APP.model_copy(update={"pages": (*HCTZ_APP.pages, APART_PAGE)}),
)


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=SHARED_PROJECT, title="HCTZ")}
    )


@pytest.fixture(scope="module")
def alone(registry: ModelRegistry) -> PanelRunner:
    """A runner which shares no run, the answers of today."""
    return PanelRunner(registry, max_runs=0)


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record the selections of every run of the simulator, which still runs."""
    recorded: list[list[str]] = []
    real = Simulator.run

    def record(self: Simulator, *args: Any, **kwargs: Any) -> Any:
        recorded.append(list(kwargs["keep"]))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Simulator, "run", record)
    return recorded


def _panel(registry: ModelRegistry, panel_id: str) -> Panel:
    return registry.panel("hctz", panel_id)


def _assert_bitwise(one: PanelResult, other: PanelResult) -> None:
    """Assert that two results are one, every number to the bit."""
    assert one.x == other.x
    assert one.dims == other.dims
    assert one.units == other.units
    assert one.failed == other.failed
    assert one.curves.keys() == other.curves.keys()
    for curve, values in one.curves.items():
        assert (
            np.asarray(values).tobytes() == np.asarray(other.curves[curve]).tobytes()
        ), curve


@pytest.mark.parametrize(("values", "grid"), STATES)
def test_the_panels_of_a_page_which_share_a_simulation_run_once(
    registry: ModelRegistry,
    alone: PanelRunner,
    runs: list[list[str]],
    values: dict[str, float],
    grid: dict[str, int],
) -> None:
    expected = {
        panel_id: alone.run("hctz", _panel(registry, panel_id), values, grid)
        for panel_id in PATIENT_PAGE.panels
    }
    del runs[:]
    shared = PanelRunner(registry)
    for panel_id in PATIENT_PAGE.panels:
        _assert_bitwise(
            shared.run("hctz", _panel(registry, panel_id), values, grid),
            expected[panel_id],
        )
    # one run of the curves of both panels, and of no curve the model lacks
    assert runs == [["[Cve_hctz]", "[Cki_plasma_hctz]", "Aurine_hctz"]]


def test_a_point_which_does_not_integrate_is_flagged_in_every_panel_of_the_run(
    registry: ModelRegistry, alone: PanelRunner, runs: list[list[str]]
) -> None:
    shared = PanelRunner(registry)
    for panel_id in PATIENT_PAGE.panels:
        panel = _panel(registry, panel_id)
        result = shared.run("hctz", panel, BROKEN, {})
        assert result.failed == (0,)
        _assert_bitwise(result, alone.run("hctz", panel, BROKEN, {}))
    assert len(runs) == 1 + len(PATIENT_PAGE.panels)


@pytest.mark.parametrize(("values", "grid"), STATES)
def test_the_panels_of_a_page_which_share_no_simulation_run_each(
    registry: ModelRegistry,
    alone: PanelRunner,
    runs: list[list[str]],
    values: dict[str, float],
    grid: dict[str, int],
) -> None:
    shared = PanelRunner(registry)
    for panel_id in APART_PAGE.panels:
        panel = _panel(registry, panel_id)
        _assert_bitwise(
            shared.run("hctz", panel, values, grid),
            alone.run("hctz", panel, values, grid),
        )
    # the first panel shares its run with the page `patient`, the second one
    # shares no simulation and runs a scan of its own, the times of which differ
    assert runs[0] == ["[Cve_hctz]", "[Cki_plasma_hctz]", "Aurine_hctz"]
    assert runs[2] == ["Aurine_hctz"]
    assert len(runs) == 4


def test_a_run_is_keyed_on_its_scan_and_not_on_the_ids_of_the_knobs(
    registry: ModelRegistry, alone: PanelRunner, runs: list[list[str]]
) -> None:
    shared = PanelRunner(registry)
    shared.run("hctz", _panel(registry, "patient_plasma"), {"dose": 25.0}, {"crcl": 8})
    renamed = _panel(registry, URINE_RENAMED.id)
    result = shared.run("hctz", renamed, {"dose": 25.0}, {"clearance": 8})
    assert len(runs) == 1
    _assert_bitwise(
        result, alone.run("hctz", renamed, {"dose": 25.0}, {"clearance": 8})
    )


def test_a_panel_whose_curves_a_stored_run_lacks_runs_its_own(
    registry: ModelRegistry, runs: list[list[str]]
) -> None:
    """A stored run of the scan of a panel of another simulation is not enough."""
    shared = PanelRunner(registry)
    # the clearance knob of the renamed panel shares the scan and not the
    # simulation of the page `patient`, whose curves its run does not keep
    shared.run("hctz", _panel(registry, URINE_RENAMED.id), {"dose": 25.0}, {})
    shared.run("hctz", _panel(registry, "patient_plasma"), {"dose": 25.0}, {})
    assert runs == [
        ["Aurine_hctz"],
        ["[Cve_hctz]", "[Cki_plasma_hctz]", "Aurine_hctz"],
    ]


def test_another_state_is_another_run(
    registry: ModelRegistry, runs: list[list[str]]
) -> None:
    shared = PanelRunner(registry)
    plasma = _panel(registry, "patient_plasma")
    shared.run("hctz", plasma, {"dose": 25.0}, {"crcl": 8})
    shared.run("hctz", plasma, {"dose": 12.5}, {"crcl": 8})
    shared.run("hctz", plasma, {"dose": 25.0}, {"crcl": 4})
    shared.run("hctz", plasma, {"dose": 25.0}, {"crcl": 8})
    assert len(runs) == 3


def test_the_runs_a_runner_keeps_are_bounded(
    registry: ModelRegistry, runs: list[list[str]]
) -> None:
    shared = PanelRunner(registry, max_runs=1)
    plasma, urine = (_panel(registry, one) for one in PATIENT_PAGE.panels)
    shared.run("hctz", plasma, {"dose": 25.0}, {})
    shared.run("hctz", plasma, {"dose": 12.5}, {})
    # the first state was evicted by the second
    shared.run("hctz", urine, {"dose": 25.0}, {})
    assert len(runs) == 3


def test_a_state_is_one_run_whatever_order_a_panel_declares_its_knobs_in(
    registry: ModelRegistry, alone: PanelRunner, runs: list[list[str]]
) -> None:
    shared = PanelRunner(registry)
    # a point sets both targets, the renal function and the dose, in the order
    # of the knobs of each panel
    for values, grid in (
        ({"egfr": 0.5, "dose": 12.5}, {}),
        ({"dose": 5.0}, {"egfr": 3}),
    ):
        del runs[:]
        shared.run("hctz", _panel(registry, ORDERED.id), values, grid)
        reordered = _panel(registry, URINE_REORDERED.id)
        result = shared.run("hctz", reordered, values, grid)
        assert len(runs) == 1
        _assert_bitwise(result, alone.run("hctz", reordered, values, grid))


def test_two_groups_of_one_state_share_its_run_and_do_not_replace_it(
    registry: ModelRegistry, alone: PanelRunner, runs: list[list[str]]
) -> None:
    """A run again for the curves of a group keeps the curves of the run before."""
    shared = PanelRunner(registry)
    weight, plasma = _panel(registry, WEIGHT.id), _panel(registry, "patient_plasma")
    state: tuple[dict[str, float], dict[str, int]] = ({"dose": 25.0}, {"crcl": 8})
    for _ in range(2):
        for panel in (weight, plasma):
            shared.run("hctz", panel, *state)
    assert runs == [
        ["BW"],
        ["[Cve_hctz]", "[Cki_plasma_hctz]", "Aurine_hctz", "BW"],
    ]
    del runs[:]
    for panel in (weight, plasma):
        _assert_bitwise(
            shared.run("hctz", panel, *state), alone.run("hctz", panel, *state)
        )


def test_a_run_is_kept_by_its_last_use(
    registry: ModelRegistry, runs: list[list[str]]
) -> None:
    shared = PanelRunner(registry, max_runs=2)
    plasma = _panel(registry, "patient_plasma")
    first, second, third = (({"dose": dose}, {}) for dose in (5.0, 12.5, 25.0))
    shared.run("hctz", plasma, *first)
    shared.run("hctz", plasma, *second)
    # a use of the first run makes the second one the least recently used
    shared.run("hctz", plasma, *first)
    shared.run("hctz", plasma, *third)
    assert len(runs) == 3
    shared.run("hctz", plasma, *first)
    assert len(runs) == 3
    shared.run("hctz", plasma, *second)
    assert len(runs) == 4


def test_only_roadrunner_shares_a_run() -> None:
    """A run is shared only where a curve does not change with the selections."""
    assert frozenset({Engine.ROADRUNNER}) == SHARED_ENGINES
    assert not SHARED_ENGINES.intersection(OTHER_ENGINES)


@pytest.mark.parametrize("engine", OTHER_ENGINES, ids=lambda one: one.value)
def test_a_panel_of_another_engine_runs_alone(
    engine: Engine, runs: list[list[str]]
) -> None:
    require(engine)
    project = SHARED_PROJECT.replace(simulator=SimulatorSettings(engine=engine))
    registry = ModelRegistry(
        {"hctz": ModelEntry(id="hctz", project=project, title="HCTZ")}
    )
    shared = PanelRunner(registry)
    values, grid = STATES[2]
    for panel_id in PATIENT_PAGE.panels:
        result = shared.run("hctz", registry.panel("hctz", panel_id), values, grid)
        assert result.failed == ()
    # one run per panel, each of the curves of its own panel
    assert runs == [["[Cve_hctz]", "[Cki_plasma_hctz]"], ["Aurine_hctz"]]


def test_a_runner_of_a_negative_number_of_runs_is_refused(
    registry: ModelRegistry,
) -> None:
    with pytest.raises(ValueError, match="negative"):
        PanelRunner(registry, max_runs=-1)


def test_concurrent_requests_of_one_state_run_it_once(
    registry: ModelRegistry, alone: PanelRunner, runs: list[list[str]]
) -> None:
    """The panels of a page ask at once: the second one waits for the run."""
    values, grid = STATES[0]
    expected = {
        panel_id: alone.run("hctz", _panel(registry, panel_id), values, grid)
        for panel_id in PATIENT_PAGE.panels
    }
    del runs[:]
    shared = PanelRunner(registry)
    # the model is warm, so the requests meet at the lock of its runs
    shared.run("hctz", _panel(registry, "patient_plasma"), {"dose": 5.0}, {})
    start = threading.Barrier(2 * len(PATIENT_PAGE.panels))
    results: dict[int, PanelResult] = {}
    errors: list[BaseException] = []

    def work(index: int, panel_id: str) -> None:
        try:
            start.wait(TIMEOUT)
            panel = _panel(registry, panel_id)
            results[index] = shared.run("hctz", panel, values, grid)
        except BaseException as error:
            errors.append(error)

    panel_ids = [*PATIENT_PAGE.panels, *PATIENT_PAGE.panels]
    threads = [
        threading.Thread(target=work, args=(index, panel_id))
        for index, panel_id in enumerate(panel_ids)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT)
    assert errors == []
    assert not [thread for thread in threads if thread.is_alive()]
    assert len(runs) == 2
    for index, panel_id in enumerate(panel_ids):
        _assert_bitwise(results[index], expected[panel_id])


@pytest.fixture
def clients(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, TestClient]]:
    """A client of an application which shares runs and one which shares none."""
    with TestClient(create_app(registry)) as shared:
        monkeypatch.setattr(
            app_module, "PanelRunner", functools.partial(PanelRunner, max_runs=0)
        )
        with TestClient(create_app(registry)) as alone:
            yield shared, alone


def _without_timing(body: dict[str, Any]) -> dict[str, Any]:
    """Get a body of a simulation without the seconds and the time of its run."""
    provenance = {
        key: value
        for key, value in body["provenance"].items()
        if key not in {"seconds", "created"}
    }
    return {**body, "provenance": provenance}


@pytest.mark.parametrize("page", [PATIENT_PAGE, APART_PAGE], ids=lambda p: p.id)
@pytest.mark.parametrize(("values", "grid"), STATES)
def test_the_answers_of_a_page_are_the_ones_without_sharing(
    clients: tuple[TestClient, TestClient],
    page: AppPage,
    values: dict[str, float],
    grid: dict[str, int],
) -> None:
    shared, alone = clients
    for panel_id in page.panels:
        route = f"/api/models/hctz/panels/{panel_id}/simulate"
        body = {"values": values, "grid": grid}
        one = shared.post(route, json=body)
        other = alone.post(route, json=body)
        assert one.status_code == other.status_code == 200
        assert _without_timing(one.json()) == _without_timing(other.json())
