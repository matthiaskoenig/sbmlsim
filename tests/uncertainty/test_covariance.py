"""Covariances of uncertain values."""

import numpy as np
import pytest

from sbmlsim.uncertainty import Covariance


def test_a_correlation_of_pairs() -> None:
    corr = Covariance.correlation({("CL", "V"): 0.6, ("V", "Ka"): -0.2})
    assert corr.targets == ("CL", "V", "Ka")
    assert corr.kind == "correlation"
    np.testing.assert_allclose(
        corr.matrix, [[1.0, 0.6, 0.0], [0.6, 1.0, -0.2], [0.0, -0.2, 1.0]]
    )
    assert corr.sds() is None
    assert not corr.matrix.flags.writeable


def test_a_covariance_has_sds_and_a_correlation() -> None:
    cov = Covariance(["CL", "V"], matrix=[[4.0, 1.2], [1.2, 9.0]], units=["l/hr", "l"])
    assert cov.kind == "covariance"
    sds = cov.sds()
    assert sds is not None
    np.testing.assert_allclose(sds, [2.0, 3.0])
    np.testing.assert_allclose(cov.correlation_matrix(), [[1.0, 0.2], [0.2, 1.0]])
    assert cov.units == ("l/hr", "l")


def test_a_matrix_which_is_not_positive_semi_definite_raises() -> None:
    matrix = [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]
    with pytest.raises(ValueError, match="smallest eigenvalue"):
        Covariance.correlation(matrix=matrix, targets=["a", "b", "c"])


def test_the_nearest_correlation_matrix() -> None:
    matrix = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    corr = Covariance.correlation(matrix=matrix, targets=["a", "b", "c"], nearest=True)
    result = corr.matrix
    assert np.linalg.eigvalsh(result).min() >= -1e-10
    np.testing.assert_allclose(np.diag(result), 1.0)
    np.testing.assert_allclose(result, result.T)
    # closer to the given matrix than the identity, which is a correlation
    assert np.linalg.norm(result - matrix) < np.linalg.norm(np.eye(3) - matrix)
    # the nearest of the nearest is itself
    again = Covariance.correlation(matrix=result, targets=["a", "b", "c"], nearest=True)
    np.testing.assert_allclose(again.matrix, result, atol=1e-8)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        ([[1.0, 0.5], [0.4, 1.0]], "symmetric"),
        ([[1.0, 1.5], [1.5, 1.0]], r"\|rho\|"),
        ([[0.9, 0.0], [0.0, 1.0]], "diagonal"),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "square"),
    ],
)
def test_an_invalid_correlation_raises(matrix: list[list[float]], match: str) -> None:
    targets = ["a", "b"] if len(matrix[0]) == 2 else ["a", "b", "c"]
    with pytest.raises(ValueError, match=match):
        Covariance.correlation(matrix=matrix, targets=targets)


def test_a_covariance_needs_a_positive_diagonal() -> None:
    with pytest.raises(ValueError, match="variance"):
        Covariance(["a", "b"], matrix=[[0.0, 0.0], [0.0, 1.0]])


def test_targets_are_unique() -> None:
    with pytest.raises(ValueError, match="twice"):
        Covariance(["a", "a"], matrix=[[1.0, 0.0], [0.0, 1.0]])


def test_a_round_trip_of_the_dict() -> None:
    cov = Covariance(
        ["CL", "V"], matrix=[[4.0, 1.2], [1.2, 9.0]], units=["l/hr", "l"], scale="log"
    )
    again = Covariance.from_dict(cov.to_dict())
    assert again == cov
    corr = Covariance.correlation({("CL", "V"): 0.6})
    assert Covariance.from_dict(corr.to_dict()) == corr
