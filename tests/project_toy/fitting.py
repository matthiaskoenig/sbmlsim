"""The fit problem of the toy model: the rate of the decay from the data of A."""

from sbmlsim.fit import FitMappingCollection, FitParameter
from sbmlsim.fit.cli import FitDefinition
from tests.project_toy import TOY_PATH
from tests.project_toy.experiments import ToyDecay


def collections_decay() -> dict[str, list[FitMappingCollection]]:
    """Get the fit mappings of the decay."""
    return {"decay": [FitMappingCollection(ToyDecay, mappings=["fm_decay_A"])]}


FIT_DEFINITIONS: dict[str, FitDefinition] = {
    "DECAY": FitDefinition(
        mapping_collections=collections_decay,
        parameters=[
            FitParameter(
                pid="k",
                start_value=0.5,
                lower_bound=0.01,
                upper_bound=1.0,
                unit="1/min",
            )
        ],
        base_path=TOY_PATH,
        data_path=TOY_PATH,
        description="the rate of the decay",
    ),
}
