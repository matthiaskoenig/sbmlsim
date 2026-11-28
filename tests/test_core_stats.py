"""The statistics of sbmlsim are computed by sbmlsim-core, `sbmlsim_core.stats`."""

import pytest
from sbmlsim_core import stats

from sbmlsim.fit import evaluation, fisher
from sbmlsim.simulation.sampling import distributions
from sbmlsim.uncertainty import covariance, testing


@pytest.mark.parametrize(
    ("value", "core"),
    [
        (evaluation.MIN_POINTS_SUBSET_AGREEMENT, stats.MIN_POINTS_SUBSET_AGREEMENT),
        (fisher.DEFAULT_RANK_TOLERANCE, stats.DEFAULT_RANK_TOLERANCE),
        (covariance.PSD_RTOL, stats.PSD_RTOL),
        (distributions.EPS, stats.EPS),
        (testing.MIXTURE, stats.MIXTURE),
        (testing.MARGIN, stats.MARGIN),
        (testing.MAX_NEEDED, stats.MAX_NEEDED),
        (testing.BISECTIONS, stats.BISECTIONS),
    ],
)
def test_the_constants_of_sbmlsim_are_the_ones_of_sbmlsim_core(
    value: object, core: object
) -> None:
    """sbmlsim names the constants the kernels of sbmlsim-core compute with."""
    assert value == core
