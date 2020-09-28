import pytest

from sbmlsim.examples.experiments.midazolam import fitting_parallel, fitting_serial
from sbmlsim.examples.experiments.midazolam.fitting_problems import (
    op_kupferschmidt1995,
    op_mandema1992,
    op_mid1oh_iv,
)


def test_fit1(tmp_path):
    opt_res = fitting_serial.fit_lsq(problem_factory=op_mid1oh_iv)
    assert opt_res is not None


def test_fit2(tmp_path):
    opt_res = fitting_serial.fit_de(problem_factory=op_mid1oh_iv)
    assert opt_res is not None


def test_fit3(tmp_path):
    opt_res = fitting_parallel.fit_lsq(problem_factory=op_mid1oh_iv)
    assert opt_res is not None


def test_fit4(tmp_path):
    opt_res = fitting_parallel.fit_de(problem_factory=op_mid1oh_iv)
    assert opt_res is not None
