import pytest

from sbmlsim.examples.experiments.midazolam import fitting_serial
from sbmlsim.examples.experiments.midazolam import fitting_parallel


def test_fit1(tmp_path):
    opt_res = fitting_serial.fitlq_mid1ohiv()
    assert opt_res is not None


def test_fit2(tmp_path):
    opt_res = fitting_serial.fitde_mid1ohiv()
    assert opt_res is not None


def test_fit3(tmp_path):
    opt_res = fitting_parallel.fitlq_mid1ohiv()
    assert opt_res is not None


def test_fit4(tmp_path):
    opt_res = fitting_parallel.fitlq_mid1ohiv()
    assert opt_res is not None

