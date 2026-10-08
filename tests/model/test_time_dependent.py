"""A model is time dependent if its math reads the time."""

import pytest

from sbmlsim.model.symbols import ModelSymbols
from tests.simulator.models import PROBE, sbml

CASES = {
    "rule": "model m\n  A = 1; k := 1 + time\n  J: A -> ; k*A\nend",
    "kinetic law": "model m\n  A = 1\n  J: A -> ; time*A\nend",
    "event trigger": "model m\n  A = 1\n  E: at time > 2: A = 5\nend",
    "event assignment": "model m\n  A = 1\n  E: at A < 0.5: A = time\n  J: A -> ; A\nend",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_math_which_reads_the_time(case: str) -> None:
    """Every place a model reads the time makes it time dependent."""
    assert ModelSymbols.from_sbml(sbml(CASES[case])).time_dependent


def test_the_probe_is_not_time_dependent() -> None:
    """A model without the time is integrated in local time."""
    assert not ModelSymbols.from_sbml(sbml(PROBE)).time_dependent


def test_a_parameter_named_time_is_not_the_time() -> None:
    """The csymbol decides, not the identifier (case 01820)."""
    import libsbml

    doc = libsbml.readSBMLFromString(sbml(PROBE))
    model = doc.getModel()
    p = model.createParameter()
    p.setId("time_")
    p.setConstant(True)
    p.setValue(1.0)
    assert not ModelSymbols.from_sbml(libsbml.writeSBMLToString(doc)).time_dependent
