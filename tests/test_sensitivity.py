"""Test sensitivity simulations."""

import pytest

from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation.sensitivity import ModelSensitivity, SensitivityType


def test_sensitivity() -> None:
    """Test sensitivity."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)

    p_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.PARAMETER_SENSITIVITY
    )
    assert len(p_ref) == 7
    p_keys = ["KM", "eff", "n", "ps_0", "ps_a", "tau_mRNA", "tau_prot"]
    for key in p_keys:
        assert key in p_ref

    s_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.SPECIES_SENSITIVITY
    )
    assert len(s_ref) == 1
    s_keys = ["Y"]
    for key in s_keys:
        assert key in s_ref

    all_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.All_SENSITIVITY
    )
    print(all_ref)
    assert len(all_ref) == 8
    all_keys = s_keys + p_keys
    for key in all_keys:
        assert key in all_ref


def test_sensitivity_change() -> None:
    """Test sensitivity change."""
    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    p_ref = ModelSensitivity.reference_dict(
        model=model, stype=SensitivityType.PARAMETER_SENSITIVITY
    )
    plus = ModelSensitivity.apply_change_to_dict(p_ref, change=0.1)
    minus = ModelSensitivity.apply_change_to_dict(p_ref, change=-0.1)
    for key in ["KM", "eff", "n", "ps_0", "ps_a", "tau_mRNA", "tau_prot"]:
        assert pytest.approx(1.1 * p_ref[key]) == plus[key]
        assert pytest.approx(0.9 * p_ref[key]) == minus[key]


def test_difference_scan_of_a_simulation() -> None:
    """The reference values are the ones of the model with the simulation's changes."""
    from sbmlsim.simulation import Scan, Simulation
    from sbmlsim.simulator import Simulator

    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulation = Simulation(end=10, steps=10, preinit_changes={"n": 3.0})
    scan = ModelSensitivity.difference_sensitivity_scan(
        model=model, simulation=simulation, difference=0.1
    )
    assert isinstance(scan, Scan)
    assert scan.simulation is simulation
    values = scan.dimensions[0].values["n"].magnitude
    assert any(v == pytest.approx(3.0 * 1.1) for v in values)
    model.set_selections(["time", "PX"])
    res = Simulator(n_workers=1).run(model, scan)
    # the changed parameters are coordinates of the dimension
    assert res["n"].values.max() == pytest.approx(3.3)
    assert res["PX"].dims == ("dim_sens", "time")


def test_distribution_scan_of_a_simulation() -> None:
    """The distribution scan is a scan of one dimension of `size` samples."""
    from sbmlsim.simulation import Scan, Simulation
    from sbmlsim.simulator import Simulator

    model = RoadrunnerSBMLModel(REPRESSILATOR_SBML)
    simulation = Simulation(end=10, steps=10)
    scan = ModelSensitivity.distribution_sensitivity_scan(
        model=model, simulation=simulation, cv=0.05, size=4
    )
    assert isinstance(scan, Scan)
    assert scan.simulation is simulation
    assert scan.dims == ("dim_sens",)
    model.set_selections(["time", "PX"])
    res = Simulator(n_workers=1).run(model, scan)
    assert res["PX"].shape == (4, 11)
    assert res.units["n"] == "dimensionless"


def test_reference_dict_follows_initial_assignments() -> None:
    """A change of a parameter reaches the species whose initial assignment uses it."""
    from tests.simulator.models import sbml

    model = RoadrunnerSBMLModel(sbml())
    ref = ModelSensitivity.reference_dict(
        model=model,
        changes={"b0": 0.5},
        stype=SensitivityType.All_SENSITIVITY,
        exclude_zero=False,
    )
    # B is the amount of the species, b0 a concentration in C = 2
    assert ref["B"] == pytest.approx(1.0)
