import pytest

from sbmlsim.sensitivity.classification import (
    SensitivityClassification,
    UncertaintyClassification,
    sensitivity_classification,
    uncertainty_classification,
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
