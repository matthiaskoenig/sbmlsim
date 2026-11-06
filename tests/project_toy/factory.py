"""The factory of the toy models: A decays with the rate k, a variant per rate."""

from pathlib import Path

from sbmlutils.factory import (
    Compartment,
    Model,
    ModelUnits,
    Parameter,
    Reaction,
    Species,
    UnitDefinition,
    Units,
    create_model,
)

#: the rate of the decay [1/min] of every variant, by the file of the variant
RATES: dict[str, float] = {"toy.xml": 0.2, "toy_slow.xml": 0.1}


class U(Units):
    """The units of the toy model."""

    mmole = UnitDefinition("mmole")
    min = UnitDefinition("min")
    m2 = UnitDefinition("m2", "meter^2")
    mM = UnitDefinition("mM", "mmole/liter")
    per_min = UnitDefinition("per_min", "1/min")
    mmole_per_min = UnitDefinition("mmole_per_min", "mmole/min")


def toy_model(rate: float) -> Model:
    """Get the toy model: A in the compartment Vc decays with the rate k."""
    return Model(
        sid="toy",
        name="toy decay model",
        units=U,
        model_units=ModelUnits(
            time=U.min,
            extent=U.mmole,
            substance=U.mmole,
            length=U.meter,
            area=U.m2,
            volume=U.liter,
        ),
        compartments=[Compartment("Vc", 1.0, name="central", unit=U.liter)],
        species=[
            Species(
                "A",
                initialConcentration=1.0,
                compartment="Vc",
                substanceUnit=U.mmole,
                hasOnlySubstanceUnits=False,
                name="A",
            )
        ],
        parameters=[Parameter("k", rate, unit=U.per_min, name="rate of the decay")],
        reactions=[
            Reaction(
                "DECAY",
                equation="A ->",
                formula=("k * A * Vc", U.mmole_per_min),
                compartment="Vc",
            )
        ],
    )


def create_toy_models(output_dir: Path) -> list[Path]:
    """Create the SBML of the variants of the toy model in a directory.

    Args:
        output_dir: directory of the models, created if it does not exist.

    Returns:
        The paths of the SBML files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        create_model(
            model=toy_model(rate),
            filepath=output_dir / name,
            sbml_level=3,
            sbml_version=2,
            validate=False,
            create_markdown=False,
            create_antimony=False,
        ).sbml_path
        for name, rate in RATES.items()
    ]


if __name__ == "__main__":
    create_toy_models(Path(__file__).parent / "models")
