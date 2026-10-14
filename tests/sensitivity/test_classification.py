import math

import pytest

from sbmlsim.sensitivity.classification import (
    SensitivityClassification,
    UncertaintyClassification,
    sensitivity_classification,
    sensitivity_classification_symbol,
    uncertainty_classification,
    uncertainty_classification_symbol,
)

# -----------------------------------------------------------------------------
# Sensitivity classification
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.5, SensitivityClassification.HIGH),
        (1.2, SensitivityClassification.HIGH),
        (-0.5, SensitivityClassification.HIGH),
        (0.2, SensitivityClassification.MEDIUM),
        (0.49, SensitivityClassification.MEDIUM),
        (-0.3, SensitivityClassification.MEDIUM),
        (0.1, SensitivityClassification.LOW),
        (0.19, SensitivityClassification.LOW),
        (-0.15, SensitivityClassification.LOW),
        (0.0, SensitivityClassification.NEGLIGIBLE),
        (0.05, SensitivityClassification.NEGLIGIBLE),
        (-0.09, SensitivityClassification.NEGLIGIBLE),
    ],
)
def test_sensitivity_classification_thresholds(value, expected):
    """Test sensitivity classification thresholds including sign handling."""
    assert sensitivity_classification(value) is expected


def test_sensitivity_classification_returns_enum():
    """Ensure the function returns a SensitivityClassification enum."""
    result = sensitivity_classification(0.3)
    assert isinstance(result, SensitivityClassification)


# -----------------------------------------------------------------------------
# Uncertainty classification
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (2.0, UncertaintyClassification.HIGH),
        (10.0, UncertaintyClassification.HIGH),
        (0.3, UncertaintyClassification.MEDIUM),
        (1.99, UncertaintyClassification.MEDIUM),
        (0.0, UncertaintyClassification.LOW),
        (0.1, UncertaintyClassification.LOW),
        (0.29, UncertaintyClassification.LOW),
    ],
)
def test_uncertainty_classification_thresholds(value, expected):
    """Test uncertainty classification thresholds."""
    assert uncertainty_classification(value) is expected


def test_uncertainty_classification_returns_enum():
    """Ensure the function returns an UncertaintyClassification enum."""
    result = uncertainty_classification(0.5)
    assert isinstance(result, UncertaintyClassification)


@pytest.mark.parametrize(
    "value, expected",
    [
        (-0.1, UncertaintyClassification.LOW),
        (-0.3, UncertaintyClassification.MEDIUM),
        (-1.66, UncertaintyClassification.MEDIUM),
        (-2.0, UncertaintyClassification.HIGH),
        (-10.0, UncertaintyClassification.HIGH),
        (math.inf, UncertaintyClassification.HIGH),
        (-math.inf, UncertaintyClassification.HIGH),
    ],
)
def test_uncertainty_classification_of_a_negative_value(value, expected):
    """The magnitude is classified (#311).

    `U = (P97.5 - P2.5) / P50` is negative for an output with a negative
    median, e.g. a change of body weight.
    """
    assert uncertainty_classification(value) is expected


def test_an_undefined_value_has_no_class():
    """An undefined value gets no class instead of raising (#311)."""
    assert sensitivity_classification(math.nan) is None
    assert uncertainty_classification(math.nan) is None
    assert sensitivity_classification_symbol(math.nan) == ""
    assert uncertainty_classification_symbol(math.nan) == ""


@pytest.mark.parametrize(
    "value, symbol",
    [(0.6, "+++"), (-0.6, "---"), (0.3, "++"), (-0.15, "-"), (0.05, "")],
)
def test_sensitivity_classification_symbol(value, symbol):
    """The symbol of a sensitivity repeats its sign once per class above negligible."""
    assert sensitivity_classification_symbol(value) == symbol


@pytest.mark.parametrize(
    "value, symbol", [(2.5, "<***>"), (-1.66, "<**>"), (0.1, "<*>")]
)
def test_uncertainty_classification_symbol(value, symbol):
    """The symbol of an uncertainty has a star per class."""
    assert uncertainty_classification_symbol(value) == symbol
