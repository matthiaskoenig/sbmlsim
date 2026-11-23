import numpy as np
from scipy.linalg import solve_triangular

from sbmlsim.fit.objects import FitParameter, Prior, PriorDistribution
from sbmlsim.fit.priors import JointPrior
from sbmlsim.uncertainty import Covariance

PARAMETERS = [
    FitParameter(
        pid="p1",
        lower_bound=0.1,
        upper_bound=10.0,
        unit="dimensionless",
        prior=Prior(PriorDistribution.NORMAL, (1.3, 0.2)),
    ),
    FitParameter(
        pid="p2",
        lower_bound=0.1,
        upper_bound=10.0,
        unit="dimensionless",
        prior=Prior(PriorDistribution.LOG_NORMAL, (0.5, 0.3)),
    ),
]
MATRIX = np.array([[1.0, 0.5], [0.5, 1.0]])
CORRELATION = Covariance.correlation({("p1", "p2"): 0.5})


def test_the_whitened_scores_are_an_explicit_product_over_the_inverse_factor() -> None:
    prior = JointPrior(PARAMETERS, CORRELATION)
    c = prior.correlated
    assert c is not None
    identity = np.eye(len(c.indices))
    np.testing.assert_array_equal(
        c.inverse, solve_triangular(np.linalg.cholesky(MATRIX), identity, lower=True)
    )
    x = np.array([0.4, 1.7])
    z = c.scores(x)
    expected = np.array(
        [sum(c.inverse[i][j] * z[j] for j in range(z.size)) for i in range(z.size)]
    )
    np.testing.assert_array_equal(c.whitened(x), expected)


def test_a_score_which_is_not_finite_whitens_to_nan() -> None:
    prior = JointPrior(PARAMETERS, CORRELATION)
    c = prior.correlated
    assert c is not None
    assert np.all(np.isnan(c.whitened(np.array([1.7, -1.0]))))
