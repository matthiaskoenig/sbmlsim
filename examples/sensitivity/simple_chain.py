"""Simple model for demonstration."""

from pathlib import Path

from sbmlutils.factory import *

_m = Model(
    sid="simple_chain",
    name="Model Simple Chain",
    notes="""Simple S1 ➞ S2 ➞ S3 conversion for testing.""",
    compartments=[
        Compartment(
            sid="Vli",
            name="liver volume",
            value=1.0,
        )
    ],
    species=[
        Species(
            sid="S1",
            name="S1",
            compartment="Vli",
            initialConcentration=1.0,
        ),
        Species(
            sid="S2",
            name="S2",
            compartment="Vli",
            initialConcentration=0.0,
        ),
        Species(
            sid="S3",
            name="S3",
            compartment="Vli",
            initialConcentration=0.0,
        ),
    ],
    parameters=[
        Parameter(sid="k1", value=1.0, name="rate S1 ➞ S2 conversion"),
        Parameter(sid="k2", value=1.0, name="rate S2 ➞ S3 conversion"),
    ],
    reactions=[
        Reaction(
            sid="R1",
            name="R1: S1 ➞ S2 conversion",
            equation="S1 -> S2",
            formula="k1 * S1",
        ),
        Reaction(
            sid="R2",
            name="R2: S2 ➞ S3 conversion",
            equation="S2 -> S3",
            formula="k2 * S2",
        ),
    ],
)

if __name__ == "__main__":
    results: FactoryResult = create_model(
        model=_m,
        filepath=Path(__file__).parent / f"{_m.sid}.xml",
        sbml_level=3,
        sbml_version=2,
        validation_options=ValidationOptions(units_consistency=False),
    )
