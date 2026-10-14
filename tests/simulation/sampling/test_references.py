"""The references of the targets of a design."""

import numpy as np
import pytest

from sbmlsim import Q
from sbmlsim.model import RoadrunnerSBMLModel
from sbmlsim.simulation import Scan, Simulation, sampling
from sbmlsim.simulation.sampling import parameters_of, references
from sbmlsim.simulator import Simulator
from tests.simulator.models import (
    SPECIES_KINDS,
    VOLUME_OF_THE_REST,
    sbml,
    sbml_minutes,
)


def test_the_references_are_the_values_after_the_preinitialization() -> None:
    model = Simulator().load(sbml())
    refs = references(model, ["k1", "pinit", "X"])
    assert refs == {"k1": 0.8, "pinit": 4.0, "X": 12.0}
    changed = references(
        model, ["k1", "pinit"], Simulation(end=1, preinit_changes={"f": 5.0, "k1": 2.0})
    )
    # pinit = 2 f is an initial assignment of the changed f
    assert changed == {"k1": 2.0, "pinit": 10.0}


def test_the_changes_of_the_model_are_defaults() -> None:
    model = RoadrunnerSBMLModel(source=sbml(), changes={"f": 3.0})
    assert references(model, ["pinit"])["pinit"] == pytest.approx(6.0)


def test_a_reference_has_the_unit_of_its_target() -> None:
    refs = references(Simulator().load(sbml_minutes()), ["f"])
    assert refs["f"] == Q(2.0, "mg")


def test_an_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="'nope'"):
        references(Simulator().load(sbml()), ["nope"])


def test_the_parameters_of_a_model() -> None:
    model = Simulator().load(sbml())
    parameters = parameters_of(model)
    assert {"a0", "b0", "k1", "k2", "f"} <= set(parameters)
    assert not any(p.endswith("__initial") for p in parameters)
    assert "k1" not in parameters_of(model, exclude={"k1"})
    assert "k1" not in parameters_of(model, exclude="k1")
    assert "k2" in parameters_of(model, exclude="k1")
    assert "k1" not in parameters_of(model, exclude=lambda pid: pid.startswith("k"))
    assert {"A", "B", "X"} <= set(parameters_of(model, species=True))
    zero = RoadrunnerSBMLModel(source=sbml(), changes={"k2": 0.0})
    assert "k2" not in parameters_of(zero)
    assert "k2" in parameters_of(zero, exclude_zero=False)


def test_a_species_of_an_assignment_rule_is_no_parameter() -> None:
    """The species of an assignment rule are not listed, a change cannot set them (#300).

    A design of all parameters and species then runs.
    """
    simulator = Simulator(n_workers=1)
    model = simulator.load(sbml(SPECIES_KINDS))
    targets = parameters_of(model, species=True)
    assert targets == ["B", "BW", "F", "I", "N", "RR", "k"]
    design = sampling.local(targets, delta=0.1, model=model)
    result = simulator.run(model, Scan(Simulation(end=1, steps=2), [design]))
    assert result.ds.sizes["local"] == 2 * len(targets) + 1


def test_a_model_path_is_loaded() -> None:
    assert references(sbml(), ["k1"]) == {"k1": 0.8}


def test_the_references_leave_the_model_as_loaded() -> None:
    """The references of a design leave no pre-initialization in the model (#315).

    A fit reads the start values of its parameters from the model.
    """
    model = Simulator().load(sbml())
    refs = references(
        model, ["k1", "pinit"], Simulation(end=1, preinit_changes={"f": 5.0, "k1": 2.0})
    )
    assert refs == {"k1": 2.0, "pinit": 10.0}
    r = model.r_loaded
    assert (r["k1"], r["f"], r["pinit"]) == (0.8, 2.0, 4.0)
    parameters_of(model, simulation=Simulation(end=1, preinit_changes={"k1": 3.0}))
    assert r["k1"] == 0.8


@pytest.mark.parametrize(
    "fraction, concentration", [(0.0611, 3.67), (0.0144, 3.64), (0.04, 4.96)]
)
def test_a_simulation_after_the_references_is_the_simulation_alone(
    fraction: float, concentration: float
) -> None:
    """The references of a volume and a concentration leave no trace (#322).

    The references are read after the pre-initialization of a simulation
    which sets a parameter of the volume of `S` and the concentration of
    `S`; the simulation after them on the same model equals the simulation
    on a fresh model, bitwise.
    """
    simulation = Simulation(end=10, steps=20, preinit_changes={"FVpa": 0.005})

    def run(model: RoadrunnerSBMLModel) -> np.ndarray:
        model.set_selections(["time", "S", "[S]"])
        result = Simulator(n_workers=1).run(model, simulation)
        return np.stack([np.asarray(result[k].values) for k in ("S", "[S]")])

    model = RoadrunnerSBMLModel(source=sbml(VOLUME_OF_THE_REST))
    references(
        model,
        ["S"],
        Simulation(end=1, preinit_changes={"FVpa": fraction, "[S]": concentration}),
    )
    np.testing.assert_array_equal(
        run(model), run(RoadrunnerSBMLModel(source=sbml(VOLUME_OF_THE_REST)))
    )
