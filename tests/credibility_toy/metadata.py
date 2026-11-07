"""The metadata of the fit mappings of the toy credibility package."""

from dataclasses import dataclass
from enum import StrEnum

from sbmlsim.fit.objects import MappingMetaData


class Route(StrEnum):
    """The route of a dose."""

    IV = "iv"
    PO = "po"


class Health(StrEnum):
    """The health of a group."""

    HEALTHY = "healthy"
    RENAL_IMPAIRMENT = "renal impairment"


@dataclass(kw_only=True)
class ToyMetaData(MappingMetaData):
    """The route and the health of the group of a mapping, and a free label."""

    route: Route
    health: Health = Health.HEALTHY
    #: a field which is no enum, which no domain can claim
    label: str = ""
