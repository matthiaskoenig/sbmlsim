import pytest

from sbmlsim.fit.optimization import OptimizerType, SamplingType
from sbmlsim.examples.experiments.midazolam import fitting_problems
from sbmlsim.examples.experiments.midazolam import fitting_serial
from sbmlsim.examples.experiments.midazolam import fitting_parallel


def test_fit1(tmp_path):
    fitting_serial.(
        size=1,
        seed=1235,
        output_path=tmp_path,
        optimizer=OptimizerType.LEAST_SQUARE,
        sampling=SamplingType.LOGUNIFORM_LHS,
        diff_step=0.05,
        jac='3-point',
        gtol=1e-10,
        xtol=1e-12,
    )
