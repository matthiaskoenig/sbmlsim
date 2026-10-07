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


def sbml_minutes() -> str:
    """Get the probe model with the time unit minute and a dose in mg."""
    import libsbml

    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml())
    model: libsbml.Model = doc.getModel()
    minute = model.createUnitDefinition()
    minute.setId("minute")
    unit = minute.createUnit()
    unit.setKind(libsbml.UNIT_KIND_SECOND)
    unit.setMultiplier(60.0)
    unit.setScale(0)
    unit.setExponent(1)
    mg = model.createUnitDefinition()
    mg.setId("mg")
    unit = mg.createUnit()
    unit.setKind(libsbml.UNIT_KIND_GRAM)
    unit.setMultiplier(1.0)
    unit.setScale(-3)
    unit.setExponent(1)
    model.setTimeUnits("minute")
    model.getParameter("f").setUnits("mg")
    return libsbml.writeSBMLToString(doc)
