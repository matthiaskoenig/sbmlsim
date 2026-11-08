"""The typical magnitudes of the states, which scale their absolute tolerances (#435)."""

import numpy as np
import pytest

from sbmlsim.model import AbstractModel, RoadrunnerSBMLModel
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.simulation import Change, Dimension, Scan, Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.simulator.scales import typical_scales
from tests.simulator.models import sbml

#: a dose of a drug in nM next to a large state in mM, in one compartment
DRUG = """
model nm_drug
  compartment C = 2
  species glc in C = 5
  species drug in C = 0
  species gone in C = 0
  k = 1
  J0: drug -> gone; k * drug * C
  J1: glc -> ; 1e-3 * glc * C
end
"""


def test_the_scale_of_a_state_is_its_largest_amount() -> None:
    """The scales are the largest amounts of the states, a state at 0 has none."""
    simulator = Simulator(n_workers=1)
    model = AbstractModel(source=sbml(DRUG))
    dose = Simulation(end=5, changes=[Change(0, {"[drug]": 1e-6})])
    scales = typical_scales(simulator, model, dose)
    assert set(scales) == {"glc", "drug", "gone"}
    assert scales["glc"] == pytest.approx(10.0)
    assert scales["drug"] == pytest.approx(2e-6)
    # the default tolerance limits the accuracy of the small states
    assert scales["gone"] == pytest.approx(2e-6 * (1 - np.exp(-5)), rel=1e-3)
    # a scan and several simulations take the largest of all their points
    scan = Scan(dose, [Dimension("d", values={"[drug]": np.array([1e-6, 3e-6])}, at=0)])
    scales = typical_scales(simulator, model, [dose, scan])
    assert scales["drug"] == pytest.approx(6e-6)
    # without a dose the drug is 0 in every simulation and has no scale
    assert set(typical_scales(simulator, model, Simulation(end=5))) == {"glc"}
    with pytest.raises(ValueError, match="at least one simulation"):
        typical_scales(simulator, model, [])


def test_scaled_tolerances_integrate_a_small_state_as_accurately_as_a_large_one() -> (
    None
):
    """The drug in nM is accurate to its relative tolerance with scaled tolerances only."""
    model = AbstractModel(source=sbml(DRUG))
    dose = Simulation(
        end=5,
        changes=[Change(0, {"[drug]": 1e-6})],
        times=np.linspace(0, 5, 51).tolist(),
    )
    scales = typical_scales(Simulator(n_workers=1), model, dose)
    exact = 2e-6 * np.exp(-np.linspace(0, 5, 51))

    def error(tolerance: AbsoluteTolerance) -> float:
        simulator = Simulator(
            n_workers=1, absolute_tolerance=tolerance, relative_tolerance=1e-8
        )
        drug = simulator.run(simulator.load(model), dose, keep=["drug"]).ds["drug"]
        return float(np.max(np.abs(drug.values.ravel() - exact)) / 2e-6)

    loose = AbsoluteTolerance.of(1e-10)
    scaled = AbsoluteTolerance(relative=1e-8, scales=scales)
    assert error(loose) > 1e-6
    assert error(scaled) < 1e-6
    loaded = Simulator(n_workers=1, absolute_tolerance=scaled).load(model)
    assert isinstance(loaded, RoadrunnerSBMLModel)
    tolerances = loaded.tolerances()
    by_id = tolerances.set_index("sid")
    assert by_id.loc["drug", "absolute_tolerance"] == pytest.approx(1e-8 * 2e-6)
    assert by_id.loc["drug", "scale"] == pytest.approx(2e-6)
