"""The problem of the PEtab v2 tutorial, written as the tables of the tutorial (#428)."""

from pathlib import Path

import numpy as np
import pytest

from examples.petab.tutorial.__main__ import compare_with_tutorial
from examples.petab.tutorial.experiment import SETTINGS, tutorial_problem
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.petab_v2 import from_petab, to_petab
from sbmlsim.fit.petab_v2.likelihood import log_likelihood

#: the estimate of the benchmark collection for the kinetic parameters, the
#: standard deviations of a fit of the tutorial
VALUES = {
    "Epo_degradation_BaF3": 0.0269825,
    "k_exp_hetero": 1.0e-5,
    "k_exp_homo": 0.00617,
    "k_imp_hetero": 0.0164,
    "k_imp_homo": 97749.4,
    "k_phos": 15766.5,
    "sd_pSTAT5A_rel": 48.95,
    "sd_pSTAT5B_rel": 4.80,
    "sd_rSTAT5A_rel": 30.70,
}


def test_the_export_has_the_tables_of_the_tutorial(tmp_path: Path) -> None:
    """The problem defined in python is written as the tables of the tutorial."""
    problem = tutorial_problem(base_path=tmp_path)
    problem.initialize(SETTINGS)
    yaml_file = to_petab(problem, tmp_path / "petab", settings=SETTINGS)
    checks = compare_with_tutorial(yaml_file.parent)
    assert all(checks.values()), checks

    # the problem which is read back has the likelihood of the problem
    read, settings = from_petab(yaml_file)
    assert settings.objective is SETTINGS.objective
    read.initialize(settings)
    pset = ParameterSet(sid="values", values=VALUES)
    llh = log_likelihood(problem, pset)
    assert np.isfinite(llh)
    assert log_likelihood(read, pset) == pytest.approx(llh, rel=1e-9)
    # the cost of the fit is the negative log-likelihood
    x = np.array([VALUES[pid] for pid in problem.pids])
    assert problem.cost_least_square(problem.to_scale(x)) == pytest.approx(
        -llh, rel=1e-9
    )
