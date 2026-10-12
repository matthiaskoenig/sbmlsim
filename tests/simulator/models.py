"""Probe models of the simulation engine, see the design of the engine.

`PROBE` covers every row of the table of roadrunner operations of the design:
initial assignments on species and parameters which depend on parameters, a
parameter with an assignment rule, a species with only substance units and a
compartment whose size is not one.
"""

from typing import Any

import antimony
import numpy as np

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


#: states of every kind: concentration species in a normal and in a degenerate
#: compartment, an amount species, a parameter with a rate rule, a species
#: with an assignment rule and a boundary species, which are no states
TOLERANCE_PROBE = """
model tolerances
  compartment C = 2; compartment U = 1e-12;
  species A in C; species S in U; substanceOnly species X in C;
  species Y in C; $Bnd in C;
  A = 1; S = 0; X = 3; Bnd = 1; D = 5
  Y := 2*A
  D' = -0.1*D
  J1: A -> X; 0.5*A
  J2: A -> S; 0.1*A
end
"""


#: a concentration species in a compartment whose volume follows the
#: parameter `f` by an initial assignment, as the organs of a model follow
#: its body weight; the absolute tolerance of `A` is the concentration
#: tolerance times the volume
VOLUME = """
model volume
  compartment C
  f = 1
  C = 2*f
  species A in C = 1
  J: A -> ; A
end
"""

#: a concentration species in a compartment whose initial assignment reads an
#: assignment rule of the body weight, as the extracellular fluid of the HCTZ
#: model of pkdb_models: roadrunner's resetAll computes the amount of `S` in
#: the last bits from the body weight set before the reset (`BW = 70.46`:
#: 95.2875 instead of 95.28750000000002, #303)
VOLUME_OF_A_RULE = """
model volume_of_a_rule
  BW = 75
  f_TBW = 0.55
  f_ECF = 0.33
  ECF := f_ECF*f_TBW*BW
  compartment V
  V = 0.05*ECF
  species S in V = 140
  J: S -> ; 0.1*S
end
"""

#: a body weight after which roadrunner's resetAll misses the amount of `S`
#: of `VOLUME_OF_A_RULE` as loaded in the last bits
OTHER_BODY_WEIGHT = 70.45995681845807

#: every kind of species in a compartment whose size follows the body weight
#: `BW` by an initial assignment: a species of a reaction `F`, of an
#: assignment rule as a concentration `R` and as an amount `NR`, of a rate rule
#: `RR`, of an initial assignment `I`, a boundary species `B` and an amount
#: species `N`; roadrunner refuses to set a species of an assignment rule
#: (#300)
SPECIES_KINDS = """
model species_kinds
  BW = 75
  compartment V
  V = 0.2*BW
  species F in V = 2
  species R in V
  R := 3*F
  species RR in V = 1
  RR' = 0.1
  species $B in V = 4
  species I in V
  I = 2*F
  substanceOnly species N in V = 6
  substanceOnly species NR in V
  NR := 2*F
  k = 0.1
  J: F -> ; k*F
end
"""


#: a species which grows as `S' = k S^2` and goes to infinity at the time
#: `1 / (k S0)`: the integration up to the time 1 fails for `k = 2` and works
#: for `k = 0.1`
BLOWUP = """
model blowup
  compartment C = 1
  species S in C = 1
  k = 0.1
  J: -> S; k*S^2
end
"""


def auc_of_c(time: np.ndarray, values: dict[str, Any]) -> float:
    """A custom observable: the trapezoidal area under `[C]`."""
    return float(np.trapezoid(values["[C]"], time))


def doubled(time: np.ndarray, values: dict[str, Any]) -> np.ndarray:
    """A custom timecourse: twice `[C]`."""
    return 2.0 * values["[C]"]


def last_value(time: np.ndarray, values: dict[str, Any]) -> float:
    """A custom observable: the last value of `S`."""
    return float(values["S"][-1])


def fails_for_large_k1(time: np.ndarray, values: dict[str, Any]) -> float:
    """A custom observable which fails for a point whose `k1` is above one."""
    if values["k1"][0] > 1.0:
        raise ValueError("k1 is too large")
    return 0.0


#: a one-compartment model with a first-order absorption from the depot
#: `PODOSE`: for a dose D at 0, C(t) = D ka / (V (ka - ke)) (exp(-ke t) - exp(-ka t))
PK_MODEL = """
model onecomp
  compartment V = 10
  species C in V = 0
  ka = 1; ke = {ke}
  PODOSE = 0
  PODOSE' = -ka*PODOSE
  absorption: -> C; ka*PODOSE
  elimination: C -> ; ke*C*V
end
"""


#: a parameter of the PK model with `blowup=True` which grows as `B' = kb B^2`
#: and goes to infinity at the time `1 / kb`: the integration up to 48 hours
#: fails for `kb = 1` and works for `kb = 0`, which leaves the model unchanged
PK_BLOWUP = """
  kb = 0; B = 1
  B' = kb*B^2
end
"""


#: the amount of the PK model with `urine=True` which its elimination
#: collects, a cumulative amount which a change empties at the end of a
#: collection interval, and the amount eliminated, which increases
PK_URINE = """
  Aurine = 0
  Aurine' = ke*C*V
  Aelim = 0
  Aelim' = ke*C*V
end
"""


def sbml_pk(ke: float = 0.2, blowup: bool = False, urine: bool = False) -> str:
    """Get the one-compartment model in hours, mg and litres."""
    import libsbml

    text = PK_MODEL.format(ke=ke)
    if blowup:
        text = text.removesuffix("end\n") + PK_BLOWUP
    if urine:
        text = text.removesuffix("end\n") + PK_URINE
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml(text))
    model: libsbml.Model = doc.getModel()
    for uid, kind, scale, multiplier, exponent in (
        ("hr", libsbml.UNIT_KIND_SECOND, 0, 3600.0, 1),
        ("mg", libsbml.UNIT_KIND_GRAM, -3, 1.0, 1),
        ("per_hr", libsbml.UNIT_KIND_SECOND, 0, 3600.0, -1),
    ):
        definition = model.createUnitDefinition()
        definition.setId(uid)
        unit = definition.createUnit()
        unit.setKind(kind)
        unit.setScale(scale)
        unit.setMultiplier(multiplier)
        unit.setExponent(exponent)
    model.setTimeUnits("hr")
    model.setSubstanceUnits("mg")
    model.setExtentUnits("mg")
    model.setVolumeUnits("litre")
    model.getCompartment("V").setUnits("litre")
    model.getSpecies("C").setSubstanceUnits("mg")
    for pid, uid in (("ka", "per_hr"), ("ke", "per_hr"), ("PODOSE", "mg")):
        model.getParameter(pid).setUnits(uid)
    if urine:
        model.getParameter("Aurine").setUnits("mg")
        model.getParameter("Aelim").setUnits("mg")
    return libsbml.writeSBMLToString(doc)


def doubled_c(time: np.ndarray, values: dict[str, Any]) -> np.ndarray:
    """A custom timecourse: twice the concentration `c`."""
    return 2.0 * values["c"]


def clearance_of(covariates: dict[str, Any]) -> dict[str, Any]:
    """A population function: k1 scales with the body weight to the power 0.75."""
    bw = np.asarray(
        getattr(covariates["BW"], "magnitude", covariates["BW"]), dtype=float
    )
    return {"k1": 0.8 * (bw / 70.0) ** 0.75}


def no_mapping(covariates: dict[str, Any]) -> Any:
    """A wrong population function, which returns a list."""
    return [1.0]


def wrong_length(covariates: dict[str, Any]) -> dict[str, Any]:
    """A wrong population function, which returns one value too many."""
    return {"k1": np.ones(len(covariates["BW"]) + 1)}


def covariate_as_target(covariates: dict[str, Any]) -> dict[str, Any]:
    """A wrong population function, which returns a covariate."""
    return {"BW": covariates["BW"]}
