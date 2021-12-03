"""Dextromethorphan intestinal model."""
from pathlib import Path

from sbmlutils.cytoscape import visualize_sbml
from sbmlutils.factory import *
from sbmlutils.metadata import *
from sbmlutils.examples.templates import terms_of_use


class U(Units):
    mmole = UnitDefinition("mmole", "mmole")
    min = UnitDefinition("min", "min")
    mmole_per_min = UnitDefinition("mmole_per_min", "mmole/min")
    mM = UnitDefinition("mM", "mmole/liter")


_m = Model(
    "nan_species",
    name="NaN species example",
    notes=terms_of_use,
    units=U,
    model_units=ModelUnits(
        time=U.min,
        substance=U.mmole,
        extent=U.mmole,
        volume=U.liter,
    ),
)


_m.compartments = [
    Compartment(
        "Vext",
        value=NaN,
        name="plasma",
        sboTerm=SBO.PHYSICAL_COMPARTMENT,
        unit=U.liter,
    ),
]

_m.species = [
    Species(
        "dex_ext",
        initialConcentration=0.0,
        name="dextromethorphan (plasma)",
        compartment="Vext",
        substanceUnit=U.mmole,
        hasOnlySubstanceUnits=False,
        sboTerm=SBO.SIMPLE_CHEMICAL,
    ),
]

_m.reactions = [
    Reaction(
        "DEXEX",
        name="DEX export",
        equation="dex_ext ->",
        sboTerm=SBO.TRANSPORT_REACTION,
        rules=[
        ],
        pars=[
            Parameter(
                "DEXEX_Vmax",
                1,
                unit=U.mmole_per_min,
                sboTerm=SBO.MAXIMAL_VELOCITY,
            ),
            Parameter(
                "DEXEX_Km_dex",
                0.1,
                unit=U.mM,
                sboTerm=SBO.MICHAELIS_CONSTANT,
            ),
        ],
        formula=(
            "DEXEX_Vmax * dex_ext/(DEXEX_Km_dex + dex_ext)",
            U.mmole_per_min,
        ),
    ),
]

model_nan = _m

if __name__ == "__main__":

    result: FactoryResult = create_model(
        output_dir=Path(".").parent, models=model_nan
    )

