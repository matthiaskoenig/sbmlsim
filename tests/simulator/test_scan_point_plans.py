"""The plans of the points of a scan, built on a compiled plan."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.executor import execute
from sbmlsim.simulator.simulator import scan_point_plans
from tests.simulator.models import sbml_pk


def test_the_plans_of_points_equal_the_points_of_a_run() -> None:
    simulator = Simulator()
    model = simulator.load(sbml_pk())
    simulation = Simulation(
        end=24, steps=48, changes=[Change(0, {"PODOSE": Q(100, "mg")})]
    )
    scan = Scan(
        simulation,
        [
            Dimension("dose", values={"PODOSE": Q([50.0, 200.0], "mg")}),
            Dimension("rate", values={"ke": np.array([0.1, 0.3])}),
        ],
    )
    plan = simulator.compile(model, simulation)
    positions = np.array([[1, 0], [0, 1]])
    plans = scan_point_plans(scan, model, plan, positions)
    result = simulator.run(model, scan)
    assert len(plans) == 2
    for (i, j), point in zip(positions, plans, strict=True):
        native = execute(point, model, ["time", "[C]"])
        np.testing.assert_allclose(
            native["[C]"], result.ds["[C]"].values[i, j], rtol=1e-10
        )


def test_a_dimension_of_simulations_raises() -> None:
    simulator = Simulator()
    model = simulator.load(sbml_pk())
    simulation = Simulation(end=1, steps=2)
    scan = Scan(
        simulation,
        [
            Dimension(
                "s",
                simulations={"a": simulation, "b": Simulation(end=2, steps=2)},
            )
        ],
    )
    with pytest.raises(ValueError, match="all set values"):
        scan_point_plans(
            scan, model, simulator.compile(model, simulation), np.array([[0]])
        )


def test_the_point_of_a_scan_without_dimensions_is_its_simulation() -> None:
    simulator = Simulator()
    model = simulator.load(sbml_pk())
    simulation = Simulation(
        end=24, steps=48, changes=[Change(0, {"PODOSE": Q(100, "mg")})]
    )
    plan = simulator.compile(model, simulation)
    [point] = scan_point_plans(
        Scan(simulation, []), model, plan, np.zeros((1, 0), dtype=int)
    )
    np.testing.assert_allclose(
        execute(point, model, ["time", "[C]"])["[C]"],
        execute(plan, model, ["time", "[C]"])["[C]"],
    )
