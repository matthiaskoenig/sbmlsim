"""The fit of the toy credibility package: the rate of the decay, with its named sets."""

from sbmlsim.fit import FitMappingCollection, FitParameter
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.filters import NamedMappings, named_filter
from sbmlsim.fit.helpers import select_mappings
from tests.credibility_toy import CREDIBILITY_PATH, PROJECT
from tests.credibility_toy.experiments import CredA, CredB

#: the validation of the dose, by study
DOSE: NamedMappings = {"CredA": {"fm_dose"}}
#: the validation of the population
POPULATION: NamedMappings = {"CredA": {"fm_renal"}}
#: the data the model does not describe
LIMITATION: NamedMappings = {"CredA": {"fm_limit"}}
#: data of zeros, whose fold errors are not evaluable
ZERO: NamedMappings = {"CredB": {"fm_zero"}}
#: validation data no evidence claims
SUBJECTS: NamedMappings = {"CredA": {"fm_subject"}}

#: every named set of validation data
VALIDATION: NamedMappings = {}
for _set in (DOSE, POPULATION, LIMITATION, ZERO, SUBJECTS):
    for _study, _keys in _set.items():
        VALIDATION.setdefault(_study, set()).update(_keys)


def collections_decay() -> dict[str, list[FitMappingCollection]]:
    """Get the fit mappings of the decay, with the validation and the outlier."""
    return select_mappings(
        PROJECT,
        [CredA, CredB],
        filters=(),
        outliers=["fm_outlier"],
        validation=named_filter(VALIDATION),
    )


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
        base_path=CREDIBILITY_PATH,
        data_path=CREDIBILITY_PATH,
        description="the rate of the decay",
    ),
}
