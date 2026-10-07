"""Probe models of the simulation engine, see the design of the engine.

`PROBE` covers every row of the table of roadrunner operations of the design:
initial assignments on species and parameters which depend on parameters, a
parameter with an assignment rule, a species with only substance units and a
compartment whose size is not one.
"""

import antimony

PROBE = """
model probe
  compartment C = 2;
  species A in C; species B in C;
  substanceOnly species X in C;
  a0 = 1; b0 = 1; k1 = 0.8; k2 = 0.6; f = 2
  kk := 3*f
  pinit = 2*f
  A = a0; B = b0; X = 3*pinit
  J1: A -> B; k1*A
  J2: B -> A; k2*B
end
"""


def sbml(model: str = PROBE) -> str:
    """Get the SBML of an antimony model.

    Args:
        model: the antimony model.

    Returns:
        The SBML of the model.

    Raises:
        ValueError: if antimony cannot read the model.
    """
    antimony.clearPreviousLoads()
    if antimony.loadAntimonyString(model) < 0:
        raise ValueError(antimony.getLastError())
    return antimony.getSBMLString(antimony.getMainModuleName())
