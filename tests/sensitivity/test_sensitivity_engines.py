"""The sensitivity analyses of scans with the engines COPASI and AMICI.

An analysis is a function of a `ScanResult`, so it is the one of roadrunner
for the result of another engine. Every test runs with every engine besides
roadrunner, see `tests.engines`; the models of the algebraic tests have no
states.
"""

import numpy as np
import pytest

from sbmlsim import sensitivity
from sbmlsim.engine import Engine
from sbmlsim.simulation import PK, Change, Formula, Scan, Simulation, sampling
from sbmlsim.simulation.sampling import Uniform
from sbmlsim.simulator import Simulator
from tests.sensitivity.models import ISHIGAMI, POWER_LAW
from tests.simulator.models import sbml, sbml_pk

PI = np.pi


@pytest.mark.parametrize(
    "simulation",
    [Simulation(end=1, steps=2), Simulation(end=1, times=[0, 0.5, 1])],
    ids=["steps", "times"],
)
def test_local_sensitivities_of_a_model_without_states(
    other_engine: Engine, simulation: Simulation
) -> None:
    """The normalized sensitivities of the power law are its exponents."""
    model = Simulator().load(sbml(POWER_LAW))
    design = sampling.local(["a", "b", "c"], 0.001, model=model)
    scan = Scan(simulation, [design])
    observables = [Formula("y_max", "max(y)"), Formula("yt", "y")]
    r = sensitivity.local(Simulator(n_workers=1).run(model, scan, observables))
    e = sensitivity.local(
        Simulator(n_workers=1, engine=other_engine).run(model, scan, observables)
    )
    np.testing.assert_allclose(
        e["y_max.normalized"].values, [2.0, -1.0, 0.5], rtol=1e-5
    )
    for name in ("y_max.normalized", "y_max.raw", "yt.normalized"):
        np.testing.assert_allclose(e[name].values, r[name].values, rtol=1e-12)


def test_local_sensitivities_of_a_pk_model(other_engine: Engine) -> None:
    """The sensitivities of the cmax and the AUC are the ones of roadrunner.

    The normalized sensitivities are about 1 but the one of the AUC to `ka`,
    -1.5e-4, a central difference of two AUCs which nearly cancel, which the
    integrators give with an absolute difference of up to 3.0e-8 (COPASI)
    and 3.9e-10 (AMICI); the others differ by about 5e-8 relative.
    """
    simulation = Simulation(
        end=48,
        times=np.linspace(0, 48, 481).tolist(),
        changes=[Change(0, {"PODOSE": 100.0})],
    )
    model = Simulator().load(sbml_pk())
    design = sampling.local(["ka", "ke"], 0.01, model=model)
    scan = Scan(simulation, [design])
    observables = [PK("pk", "[C]", dose="PODOSE", route="oral")]
    r = sensitivity.local(Simulator(n_workers=1).run(model, scan, observables))
    e = sensitivity.local(
        Simulator(n_workers=1, engine=other_engine).run(model, scan, observables)
    )
    assert e.parameters == r.parameters == ["ka", "ke"]
    for name in ("pk.cmax.normalized", "pk.auc_last.normalized"):
        np.testing.assert_allclose(e[name].values, r[name].values, rtol=1e-6, atol=1e-7)


def test_sobol_indices_of_the_ishigami_function(other_engine: Engine) -> None:
    """The Sobol indices of a model without states are the ones of roadrunner."""
    bounds = {name: Uniform(-PI, PI) for name in ("x1", "x2", "x3")}
    model = Simulator().load(sbml(ISHIGAMI))
    scan = Scan(Simulation(end=1, steps=1), [sampling.sobol(bounds, 128, seed=1)])
    observables = [Formula("y_max", "max(y)")]
    r = sensitivity.sobol(Simulator(n_workers=1).run(model, scan, observables))
    e = sensitivity.sobol(
        Simulator(n_workers=1, engine=other_engine).run(model, scan, observables)
    )
    for name in ("y_max.S1", "y_max.ST"):
        np.testing.assert_allclose(e[name].values, r[name].values, atol=1e-10)
