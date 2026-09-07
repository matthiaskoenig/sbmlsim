"""Model with a chain of two reactions for the curve type examples."""

from pathlib import Path

from sbmlutils.factory import *
from sbmlutils.metadata import SBO


class U(Units):
    """Units of the model."""

    min = UnitDefinition("min")
    mmole = UnitDefinition("mmole")
    mM = UnitDefinition("mM", "mmole/liter")
    mmole_per_min = UnitDefinition("mmole_per_min", "mmole/min")
    litre_per_min = UnitDefinition("litre_per_min", "liter/min")


model = Model(
    sid="curve_types_model",
    name="curve types model",
    notes="""
    # Curve types model
    Chain of two reactions, S1 -> S2 -> S3, for the curve type examples.
    """,
    creators=[
        Creator(
            familyName="König",
            givenName="Matthias",
            email="koenigmx@hu-berlin.de",
            organization="Humboldt-University Berlin, Institute for Theoretical Biology",
            site="https://livermetabolism.com",
            orcid="0000-0003-1725-179X",
        )
    ],
    units=U,
    model_units=ModelUnits(
        time=U.min,
        length=U.meter,
        extent=U.mmole,
        substance=U.mmole,
        volume=U.liter,
    ),
    compartments=[
        Compartment(sid="cell", name="cell", value=1.0, unit=U.liter),
    ],
    species=[
        Species(
            sid=sid,
            compartment="cell",
            initialConcentration=value,
            sboTerm=SBO.SIMPLE_CHEMICAL,
            hasOnlySubstanceUnits=False,
            substanceUnit=U.mmole,
        )
        for sid, value in [("S1", 10.0), ("S2", 5.0), ("S3", 0.0)]
    ],
    parameters=[Parameter("k", 0.3, constant=True, unit=U.litre_per_min)],
    reactions=[
        Reaction(
            sid="_J0",
            equation="S1 -> S2",
            compartment="cell",
            formula=("k * S1", U.mmole_per_min),
        ),
        Reaction(
            sid="_J1",
            equation="S2 -> S3",
            compartment="cell",
            formula=("k * S2", U.mmole_per_min),
        ),
    ],
)


def create(output_dir: Path) -> Path:
    """Create the model and return the path of the written SBML file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return create_model(model=model, filepath=output_dir / f"{model.sid}.xml").sbml_path


if __name__ == "__main__":
    create(output_dir=Path.cwd() / "results")
