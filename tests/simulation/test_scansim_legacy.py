"""The legacy ScanSim replaces the value of a target wherever the simulation sets it. Removed with ScanSim."""

import numpy as np

from sbmlsim import Q
from sbmlsim.simulation import Change, Dimension, ScanSim, Simulation


def test_scan_replaces_the_dose_at_every_time() -> None:
    """A scanned dose replaces the dose of every change, nothing is added."""
    sim = Simulation(end=72, changes=[Change([0, 24, 48], {"PODOSE": Q(10, "mg")})])
    scan = ScanSim(
        sim, [Dimension("dose", values={"PODOSE": Q(np.array([5.0, 20.0]), "mg")})]
    )
    _, sims = scan.to_simulations()
    assert [s.changes[0].values["PODOSE"] for s in sims] == [
        Q(5.0, "mg"),
        Q(20.0, "mg"),
    ]
    assert all(s.preinit_changes == {} for s in sims)
    assert all(s.changes[0].times == (0, 24, 48) for s in sims)


def test_scan_at_a_time_adds_a_change() -> None:
    """A dimension at a time is a change at that time."""
    scan = ScanSim(
        Simulation(end=10),
        [Dimension("k", values={"k1": np.array([1.0, 2.0])}, at=5)],
    )
    _, sims = scan.to_simulations()
    assert [s.changes[-1].times for s in sims] == [(5,), (5,)]
    assert [s.changes[-1].values["k1"] for s in sims] == [1.0, 2.0]
    assert all(s.preinit_changes == {} for s in sims)


def test_scan_of_two_dimensions() -> None:
    """Every combination of the dimensions is a simulation."""
    scan = ScanSim(
        Simulation(end=1),
        [
            Dimension("a", values={"k1": np.array([1.0, 2.0])}),
            Dimension("b", values={"k2": np.array([3.0, 4.0, 5.0])}),
        ],
    )
    indices, sims = scan.to_simulations()
    assert len(sims) == 6
    assert indices[5] == (1, 2)
    assert sims[5].preinit_changes == {"k1": 2.0, "k2": 5.0}


def test_the_scanned_simulation_is_not_changed() -> None:
    """The simulation of a scan stays as it is."""
    sim = Simulation(end=1, preinit_changes={"k1": 0.5})
    ScanSim(sim, [Dimension("a", values={"k1": np.array([1.0, 2.0])})]).to_simulations()
    assert sim.preinit_changes == {"k1": 0.5}
