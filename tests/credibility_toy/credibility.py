"""The contexts of use of the toy credibility package."""

import dataclasses

from sbmlsim.credibility import (
    ContextOfUse,
    CouLevel,
    Credibility,
    Domain,
    Evidence,
    EvidenceRole,
    Level,
    QuantityOfInterest,
    Unclaimed,
)
from sbmlsim.fit.filters import metadata_filter, named_filter
from sbmlsim.fit.variability import Variability
from sbmlsim.simulation.sampling import LogNormal
from tests.credibility_toy.fitting import DOSE, LIMITATION, POPULATION
from tests.credibility_toy.metadata import Health, Route, ToyMetaData

_A = (QuantityOfInterest("[A]", "the concentration of A"),)

CREDIBILITY = Credibility(
    contexts=(
        ContextOfUse(
            id="describe",
            title="Describe the decay",
            question="Does the model describe the intravenous decay of A?",
            level=CouLevel.C1,
            influence=Level.LOW,
            consequence=Level.LOW,
            quantities=_A,
            domain=Domain(),
            evidence=(
                Evidence(
                    "intravenous data",
                    EvidenceRole.CALIBRATION,
                    metadata_filter(ToyMetaData, route={Route.IV}),
                ),
            ),
        ),
        ContextOfUse(
            id="dose",
            title="Decay after a dose",
            question="Does the model predict the decay after a dose of the data?",
            level=CouLevel.C2,
            influence=Level.MEDIUM,
            consequence=Level.MEDIUM,
            quantities=_A,
            domain=Domain(
                metadata={"route": (Route.IV,), "health": (Health.HEALTHY,)},
                statement="the single doses of the data",
            ),
            evidence=(
                Evidence("doses", EvidenceRole.VALIDATION, named_filter(DOSE)),
                Evidence(
                    "other healthy data",
                    EvidenceRole.VALIDATION,
                    Unclaimed(
                        within=metadata_filter(
                            ToyMetaData, route={Route.IV}, health={Health.HEALTHY}
                        )
                    ),
                ),
            ),
        ),
        ContextOfUse(
            id="renal",
            title="Decay in renal impairment",
            question="Does the model predict the decay in renal impairment?",
            level=CouLevel.C3,
            influence=Level.MEDIUM,
            consequence=Level.HIGH,
            quantities=(
                *_A,
                QuantityOfInterest(
                    "A_ratio",
                    "the concentration in renal impairment over the healthy one",
                    ratio=True,
                ),
            ),
            domain=Domain(
                metadata={"health": (Health.HEALTHY, Health.RENAL_IMPAIRMENT)}
            ),
            evidence=(
                Evidence(
                    "renal impairment",
                    EvidenceRole.VALIDATION,
                    named_filter(POPULATION),
                ),
            ),
        ),
    ),
    limitations=(
        Evidence(
            "scaled data",
            EvidenceRole.LIMITATION,
            named_filter(LIMITATION),
            "data at three times the decay which the model does not describe",
        ),
    ),
)

#: a declaration which validates on fitted data, CredB `fm_dose` is training data
BROKEN = Credibility(
    contexts=(
        ContextOfUse(
            id="fitted",
            title="Validation on fitted data",
            question="Does the model predict the data it was fitted on?",
            level=CouLevel.C2,
            influence=Level.LOW,
            consequence=Level.LOW,
            quantities=_A,
            domain=Domain(metadata={"route": (Route.IV,)}),
            evidence=(
                Evidence(
                    "fitted dose",
                    EvidenceRole.VALIDATION,
                    named_filter({"CredB": {"fm_dose"}}),
                ),
            ),
        ),
    ),
)

#: a declaration whose domain claims a field which is no enum of the metadata
#: and a value of another enum
OUTSIDE = dataclasses.replace(
    BROKEN,
    contexts=(
        dataclasses.replace(
            BROKEN.contexts[0],
            domain=Domain(metadata={"label": (Route.IV,), "route": (Health.HEALTHY,)}),
        ),
    ),
)

#: the declaration with a variability of the rate of the decay between subjects
CREDIBILITY_VARIABILITY = dataclasses.replace(
    CREDIBILITY, variability=Variability({"k": LogNormal(cv=0.2)})
)
