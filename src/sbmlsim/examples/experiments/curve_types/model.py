"""Create reaction example."""
from pathlib import Path

from sbmlutils.creator import create_model
from sbmlutils.examples import templates
from sbmlutils.factory import *
from sbmlutils.metadata.sbo import *
from sbmlutils.units import *


mid = "curve_types_model"
notes = Notes(
    [
        """
    <h1>Koenig sbmlutils example</h1>
    <h2>Description</h2>
    <p>Single reaction.
    </p>
    """,
        templates.terms_of_use,
    ]
)
creators = templates.creators
model_units = ModelUnits(
    time=UNIT_min,
    length=UNIT_KIND_METRE,
    extent=UNIT_mmole,
    substance=UNIT_mmole,
    volume=UNIT_KIND_LITRE,
)
units = [
    UNIT_min,
    UNIT_mmole,
    UNIT_mM,
    UNIT_mmole_per_min,
    UNIT_litre_per_min,
]

compartments = [
    Compartment(sid="cell", name="cell", value=1.0, unit="litre"),
]

species = [
    Species(
        sid="S1",
        compartment="cell",
        initialConcentration=10.0,
        sboTerm=SBO_SIMPLE_CHEMICAL,
        hasOnlySubstanceUnits=False,
        substanceUnit=UNIT_mmole,
    ),
    Species(
        sid="S2",
        compartment="cell",
        initialConcentration=5.0,
        sboTerm=SBO_SIMPLE_CHEMICAL,
        hasOnlySubstanceUnits=False,
        substanceUnit=UNIT_mmole,
    ),
    Species(
        sid="S3",
        compartment="cell",
        initialConcentration=0.0,
        sboTerm=SBO_SIMPLE_CHEMICAL,
        hasOnlySubstanceUnits=False,
        substanceUnit=UNIT_mmole,
    ),
]

parameters = [Parameter("k", 0.3, constant=True, unit=UNIT_litre_per_min)]

reactions = [
    Reaction(
        sid="_J0",
        equation="S1 -> S2",
        compartment="cell",
        formula=("k * S1", UNIT_mmole_per_min),
    ),
    Reaction(
        sid="_J1",
        equation="S2 -> S3",
        compartment="cell",
        formula=("k * S2", UNIT_mmole_per_min),
    ),
]


def create(tmp: bool = False) -> None:
    """Create model."""
    create_model(
        modules=["model"],
        output_dir=Path(__file__).parent / "results",
        tmp=tmp,
        units_consistency=True,
    )


if __name__ == "__main__":
    create()
