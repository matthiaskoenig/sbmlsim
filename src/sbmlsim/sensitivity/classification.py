"""Classification of sensitivities and uncertainties.

The classes are the ones of the IPCS guidance for physiologically based
pharmacokinetic models; an undefined value (`NaN`) has no class (`None`), so
one undefined index does not stop the classification of the others.
"""

import math
from enum import StrEnum


class SensitivityClassification(StrEnum):
    """Sensitivity classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


def sensitivity_classification(s: float) -> SensitivityClassification | None:
    """Classification of local sensitivity as per WHO IPCS guidance.

    Classification based on absolute sensitivity:

    - High: `|Si,j| >= 0.5`
    - Medium: `0.2 <= |Si,j| < 0.5`
    - Low: `0.1 <= |Si,j| < 0.2`
    - Negligible: `|Si,j| < 0.1`

    Args:
        s: the normalized sensitivity.

    Returns:
        The class, `None` for an undefined value (`NaN`).

    References:
        International Programme on Chemical Safety (IPCS). Characterization and
        application of physiologically based pharmacokinetic models in risk assessment.
        World Health Organization; 2010. Contract No.: 9.
    """
    s_abs = abs(float(s))
    if math.isnan(s_abs):
        return None
    if s_abs >= 0.5:
        return SensitivityClassification.HIGH
    if s_abs >= 0.2:
        return SensitivityClassification.MEDIUM
    if s_abs >= 0.1:
        return SensitivityClassification.LOW
    return SensitivityClassification.NEGLIGIBLE


def sensitivity_classification_symbol(s: float) -> str:
    """Get the symbol of the class of a sensitivity.

    Args:
        s: the normalized sensitivity.

    Returns:
        The sign of the sensitivity once per class above negligible, `+++`
        for a high positive and `-` for a low negative sensitivity, empty for
        a negligible or an undefined one.
    """
    classification = sensitivity_classification(s)
    n = {
        SensitivityClassification.HIGH: 3,
        SensitivityClassification.MEDIUM: 2,
        SensitivityClassification.LOW: 1,
    }.get(classification, 0)
    return ("+" if s >= 0 else "-") * n


class UncertaintyClassification(StrEnum):
    """Uncertainty classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def uncertainty_classification(u: float) -> UncertaintyClassification | None:
    """Classification of uncertainty as per WHO IPCS guidance.

    Classification based on the absolute uncertainty, `U = (P97.5 - P2.5) / P50`
    is negative for an output with a negative median, e.g. a change of the
    body weight (#311):

    - High: `|Ui,j| >= 2`
    - Medium: `0.3 <= |Ui,j| < 2`
    - Low: `0 <= |Ui,j| < 0.3`

    Args:
        u: the uncertainty.

    Returns:
        The class, `None` for an undefined value (`NaN`).

    References:
        International Programme on Chemical Safety (IPCS). Characterization and application of physiologically based pharmacokinetic models in risk assessment.
        World Health Organization; 2010. Contract No.: 9.
    """
    u_abs = abs(float(u))
    if math.isnan(u_abs):
        return None
    if u_abs >= 2.0:
        return UncertaintyClassification.HIGH
    if u_abs >= 0.3:
        return UncertaintyClassification.MEDIUM
    return UncertaintyClassification.LOW


def uncertainty_classification_symbol(u: float) -> str:
    """Get the symbol of the class of an uncertainty.

    Args:
        u: the uncertainty.

    Returns:
        A star per class in angle brackets, `<***>` for a high and `<*>` for
        a low uncertainty, empty for an undefined one.
    """
    classification = uncertainty_classification(u)
    if classification is None:
        return ""
    n = {
        UncertaintyClassification.HIGH: 3,
        UncertaintyClassification.MEDIUM: 2,
        UncertaintyClassification.LOW: 1,
    }[classification]
    return f"<{'*' * n}>"
