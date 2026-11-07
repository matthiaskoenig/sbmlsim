"""The model risk, the default acceptances and the default minimum of points.

The model risk follows ICH M15 and V&V 40 as a lookup of the influence of the
model on the decision and the consequence of a wrong decision. It is a table
and not a formula, so it can be changed in one place. The rules of an
evidence extension, e.g. of the ratio pairs, have their defaults in the
extension, see `sbmlsim.credibility.extensions`.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from sbmlsim.credibility.declaration import (
    Acceptance,
    AcceptanceRule,
    Level,
    MaxGmfe,
)
from sbmlsim.fit.evaluation import Fold

__all__ = [
    "DEFAULT_ACCEPTANCE",
    "DEFAULT_MIN_POINTS",
    "RISK_TABLE",
    "default_acceptance",
    "model_risk",
]

#: the model risk by influence and consequence
RISK_TABLE: Final[Mapping[tuple[Level, Level], Level]] = MappingProxyType(
    {
        (Level.LOW, Level.LOW): Level.LOW,
        (Level.LOW, Level.MEDIUM): Level.LOW,
        (Level.LOW, Level.HIGH): Level.MEDIUM,
        (Level.MEDIUM, Level.LOW): Level.LOW,
        (Level.MEDIUM, Level.MEDIUM): Level.MEDIUM,
        (Level.MEDIUM, Level.HIGH): Level.HIGH,
        (Level.HIGH, Level.LOW): Level.MEDIUM,
        (Level.HIGH, Level.MEDIUM): Level.HIGH,
        (Level.HIGH, Level.HIGH): Level.HIGH,
    }
)

#: the acceptance of a context of use which does not declare its own, by risk
DEFAULT_ACCEPTANCE: Final[Mapping[Level, tuple[AcceptanceRule, ...]]] = (
    MappingProxyType(
        {
            Level.LOW: (Acceptance(Fold(2.0), 70.0),),
            Level.MEDIUM: (Acceptance(Fold(2.0), 80.0), MaxGmfe(1.5)),
            Level.HIGH: (
                Acceptance(Fold(2.0), 90.0),
                Acceptance(Fold(1.5), 70.0),
                MaxGmfe(1.3),
            ),
        }
    )
)

#: the evaluable points a context of use which does not declare its own minimum
#: needs, by risk
DEFAULT_MIN_POINTS: Final[Mapping[Level, int]] = MappingProxyType(
    {Level.LOW: 5, Level.MEDIUM: 10, Level.HIGH: 20}
)


def model_risk(influence: Level, consequence: Level) -> Level:
    """Get the model risk of an influence and a consequence.

    Args:
        influence: the influence of the model on the decision.
        consequence: the consequence of a wrong decision.

    Returns:
        The model risk.
    """
    return RISK_TABLE[(influence, consequence)]


def default_acceptance(risk: Level) -> tuple[AcceptanceRule, ...]:
    """Get the acceptance a context of use of a risk takes without its own.

    Args:
        risk: the model risk.

    Returns:
        The rules, all of which must be met.
    """
    return DEFAULT_ACCEPTANCE[risk]
