"""Classification of sensitivities and uncertainties."""

from enum import Enum

import numpy as np


class SensitivityClassification(str, Enum):
    """Sensitivity classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


def sensitivity_classification(s: float) -> SensitivityClassification:
    """Classification of local sensitivity as per WHO IPCS guidance.

    Classification based on absolute sensitivity:

    - High: `|Si,j| >= 0.5`
    - Medium: `0.2 <= |Si,j| < 0.5`
    - Low: `0.1 <= |Si,j| < 0.2`
    - Negligible: `|Si,j| < 0.1`

    References:
        International Programme on Chemical Safety (IPCS). Characterization and
        application of physiologically based pharmacokinetic models in risk assessment.
        World Health Organization; 2010. Contract No.: 9.
    """
    s_abs = np.absolute(s)

    if np.greater_equal(s_abs, 0.5):
        return SensitivityClassification.HIGH
    elif np.greater_equal(s_abs, 0.2) and s_abs < 0.5:
        return SensitivityClassification.MEDIUM
    elif np.greater_equal(s_abs, 0.1) and s_abs < 0.2:
        return SensitivityClassification.LOW
    elif s_abs < 0.1:
        return SensitivityClassification.NEGLIGIBLE
    else:
        raise ValueError(f"Unsupported sensitivity classification for s={s}.")


def sensitivity_classification_symbol(s: float) -> str:
    """Calculates symbol for sensitivity classification."""
    classification = sensitivity_classification(s)
    n = 0
    if classification == SensitivityClassification.HIGH:
        n = 3
    elif classification == SensitivityClassification.MEDIUM:
        n = 2
    elif classification == SensitivityClassification.LOW:
        n = 1

    symbol = ""
    if s >= 0:
        symbol = "+" * n
    elif s < 0:
        symbol = "-" * n
    return symbol


class UncertaintyClassification(str, Enum):
    """Uncertainty classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def uncertainty_classification(u: float) -> UncertaintyClassification:
    """Classification of uncertainty as per WHO IPCS guidance.

    - High: `Ui,j >= 2`
    - Medium: `0.3 <= |Ui,j| < 2`
    - Low: `0 <= |Ui,j| < 0.3`

    References:
        International Programme on Chemical Safety (IPCS). Characterization and application of physiologically based pharmacokinetic models in risk assessment.
        World Health Organization; 2010. Contract No.: 9.
    """

    if np.greater_equal(u, 2.0):
        return UncertaintyClassification.HIGH
    elif np.greater_equal(u, 0.3) and u < 2.0:
        return UncertaintyClassification.MEDIUM
    elif np.greater_equal(u, 0.0) and u < 0.3:
        return UncertaintyClassification.LOW
    else:
        raise ValueError(f"Unsupported uncertainty classification for u={u}.")


def uncertainty_classification_symbol(u: float) -> str:
    """Calculates symbol for uncertainty classification."""
    classification = uncertainty_classification(u)
    n = 0
    if classification == UncertaintyClassification.HIGH:
        n = 3
    elif classification == UncertaintyClassification.MEDIUM:
        n = 2
    elif classification == UncertaintyClassification.LOW:
        n = 1

    return f"<{'*' * n}>"
