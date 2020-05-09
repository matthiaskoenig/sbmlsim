import pytest

from sbmlsim.fit.optimization import OptimizerType, SamplingType
from sbmlsim.examples.experiments.midazolam import fitting


def test_fit1(tmp_path):
    fitting.mid1oh_iv_optimization(
        size=1,
        seed=1235,
        output_path=tmp_path,
        optimizer=OptimizerType.LEAST_SQUARE,
        sampling=SamplingType.LOGUNIFORM,
        diff_step=0.05,
        jac='3-point',
        gtol=1e-10,
        xtol=1e-12,
    )
