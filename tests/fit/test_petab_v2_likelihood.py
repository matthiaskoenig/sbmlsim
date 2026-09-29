"""Tests of the log-likelihood of a problem."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sbmlsim.fit.objects import NoiseDistribution, NoiseModel, NoiseParameter
from sbmlsim.fit.petab_v2.likelihood import (
    default_noise_model,
    log_density,
    noise_values,
)

#: simulations and measurements of the case `sciml_problem_import/001` of the
#: PEtab SciML test suite, normal noise with the scale 0.05
SCIML_001 = Path(__file__).parents[1] / "data" / "petab" / "sciml_001_llh.tsv"
#: the `llh` of the `solutions.yaml` of the case and its `tol_llh`
SCIML_001_LLH = 33.02909543616689
SCIML_001_TOL = 1e-3


def test_log_density_normal() -> None:
    """The density of the normal distribution, calculated by hand."""
    # -0.5 * log(2 pi 0.25) - 0.5 * ((2 - 3) / 0.5)^2
    expected = -0.5 * np.log(2.0 * np.pi * 0.25) - 2.0
    assert expected == pytest.approx(-2.2257913526447273, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.NORMAL)
    assert density == pytest.approx([expected], rel=1e-14)


def test_log_density_log_normal() -> None:
    """The simulation is the median and the density is the one of `m`."""
    # -0.5 * log(2 pi 0.25 * 4) - 0.5 * ((log 2 - log 3) / 0.5)^2
    expected = -0.5 * np.log(2.0 * np.pi) - 2.0 * np.log(1.5) ** 2
    assert expected == pytest.approx(-1.2477424409910038, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LOG_NORMAL)
    assert density == pytest.approx([expected], rel=1e-14)


def test_log_density_laplace() -> None:
    """The density of the Laplace distribution, calculated by hand."""
    # -log(2 * 0.5) - |2 - 3| / 0.5
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LAPLACE)
    assert density == pytest.approx([-2.0], rel=1e-14)


def test_log_density_log_laplace() -> None:
    """The density of the log-Laplace distribution, calculated by hand."""
    # -log(2 * 0.5 * 2) - |log 2 - log 3| / 0.5
    expected = -np.log(2.0) - 2.0 * np.log(1.5)
    assert expected == pytest.approx(-1.5040773967762742, rel=1e-15)
    density = log_density([2.0], [3.0], 0.5, NoiseDistribution.LOG_LAPLACE)
    assert density == pytest.approx([expected], rel=1e-14)


@pytest.mark.parametrize("distribution", list(NoiseDistribution))
def test_log_density_is_a_density(distribution: NoiseDistribution) -> None:
    """The density of the measurement integrates to one.

    This is what tells the density of `m` from the density of `log m`, i.e.
    it pins the `1 / m` of the logarithmic distributions.
    """
    start = 1e-6 if distribution.is_log else -27.0
    m = np.linspace(start, start + 60.0, 600001)
    density = np.exp(log_density(m, np.full_like(m, 3.0), 0.25, distribution))
    assert np.trapezoid(density, m) == pytest.approx(1.0, abs=1e-4)


def test_log_density_has_its_maximum_at_the_measurement() -> None:
    """A simulation which hits the measurement is the most likely one."""
    for distribution in NoiseDistribution:
        at, off = log_density([2.0, 2.0], [2.0, 2.5], 0.5, distribution)
        assert at > off


def test_log_density_takes_a_scale_per_measurement() -> None:
    """The scale is one value or one per measurement."""
    density = log_density([1.0, 1.0], [1.0, 1.0], [0.5, 2.0])
    assert density == pytest.approx(
        [-0.5 * np.log(2.0 * np.pi * 0.25), -0.5 * np.log(2.0 * np.pi * 4.0)]
    )


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_log_density_requires_a_positive_scale(sigma: float) -> None:
    """A scale which is not a positive number is an error and not a `nan`."""
    with pytest.raises(ValueError, match="positive finite"):
        log_density([1.0], [1.0], sigma)


def test_log_density_requires_positive_values_for_a_log_distribution() -> None:
    """The logarithm of a measurement which is not positive does not exist."""
    with pytest.raises(ValueError, match="measurement"):
        log_density([0.0], [1.0], 0.5, NoiseDistribution.LOG_NORMAL)
    with pytest.raises(ValueError, match="simulation"):
        log_density([1.0], [-1.0], 0.5, NoiseDistribution.LOG_LAPLACE)
    # the normal distribution takes them
    assert np.isfinite(log_density([-1.0], [0.0], 0.5)).all()


def test_log_density_requires_one_shape() -> None:
    """Every measurement has a simulation."""
    with pytest.raises(ValueError, match="shape"):
        log_density([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(ValueError, match="scale"):
        log_density([1.0, 2.0], [1.0, 2.0], [0.5, 0.5, 0.5])


def test_log_likelihood_of_the_sciml_test_suite() -> None:
    """The `llh` of a case of the PEtab SciML test suite is reproduced.

    This pins the sign and the constant of the log-likelihood against a value
    which another tool calculated.
    """
    df = pd.read_csv(SCIML_001, sep="\t", float_precision="round_trip")
    assert len(df) == 20
    llh = float(np.sum(log_density(df.measurement, df.simulation, 0.05)))
    assert llh == pytest.approx(SCIML_001_LLH, abs=SCIML_001_TOL)
    assert llh == pytest.approx(SCIML_001_LLH, rel=1e-12)


def test_noise_values_of_a_number() -> None:
    """A noise formula which is a number is the scale of every measurement."""
    sigma = noise_values(NoiseModel(formula="0.05"), size=3)
    assert sigma.tolist() == [0.05, 0.05, 0.05]


def test_noise_values_of_a_placeholder() -> None:
    """A placeholder has a value per measurement."""
    noise = NoiseModel(
        formula="0.1 + 2 * sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), (1.0,)),
    )
    assert noise_values(noise, size=2) == pytest.approx([1.1, 2.1])


def test_noise_values_of_a_parameter() -> None:
    """A parameter is evaluated at the given value, the nominal one without."""
    noise = NoiseModel(
        formula="sigma_a ^ 2",
        parameters=(NoiseParameter(pid="sigma_a", value=3.0, estimate=True),),
    )
    assert noise_values(noise, size=2) == pytest.approx([9.0, 9.0])
    assert noise_values(noise, size=2, values={"sigma_a": 2.0}) == pytest.approx(
        [4.0, 4.0]
    )


def test_noise_values_of_a_placeholder_which_is_a_parameter() -> None:
    """The value of a placeholder is a number or a formula of parameters."""
    noise = NoiseModel(
        formula="sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), ("sigma_a",), ("2 * sigma_a",)),
        parameters=(NoiseParameter(pid="sigma_a", value=3.0),),
    )
    assert noise_values(noise, size=3) == pytest.approx([0.5, 3.0, 6.0])


def test_noise_values_of_the_observable() -> None:
    """A noise which is proportional to the simulation."""
    noise = NoiseModel(formula="0.1 * obs_a + 0.01", observable="obs_a")
    sigma = noise_values(noise, size=2, simulation=[1.0, 2.0])
    assert sigma == pytest.approx([0.11, 0.21])


def test_noise_values_of_a_placeholder_named_like_a_parameter() -> None:
    """A placeholder is the value of the measurement, whatever else has its name."""
    noise = NoiseModel(
        formula="sd",
        placeholders=("sd",),
        placeholder_values=((0.5,), (0.25,)),
        parameters=(NoiseParameter(pid="sd", value=3.0),),
    )
    sigma = noise_values(noise, size=2, values={"sd": 7.0})
    assert sigma.tolist() == [0.5, 0.25]


def test_noise_values_require_every_symbol() -> None:
    """A symbol without a value is named."""
    with pytest.raises(ValueError, match="k_unknown"):
        noise_values(NoiseModel(formula="2 * k_unknown"), size=2)


def test_noise_values_require_a_value_per_measurement() -> None:
    """The values of the placeholders are the ones of the measurements."""
    noise = NoiseModel(
        formula="sd", placeholders=("sd",), placeholder_values=((0.5,), (1.0,))
    )
    with pytest.raises(ValueError, match="'2' values"):
        noise_values(noise, size=3)


def test_noise_values_require_a_positive_scale() -> None:
    """A noise formula which is not positive is an error."""
    with pytest.raises(ValueError, match="positive finite"):
        noise_values(NoiseModel(formula="-0.05"), size=2)


def test_a_noise_model_requires_a_value_per_placeholder() -> None:
    """A measurement has a value for every placeholder."""
    with pytest.raises(ValueError, match="placeholders"):
        NoiseModel(
            formula="a + b", placeholders=("a", "b"), placeholder_values=((0.5,),)
        )


def test_default_noise_model() -> None:
    """Without a noise model the noise is the standard deviation of the data."""
    noise = default_noise_model(np.array([0.5, 0.25]))
    assert noise.distribution is NoiseDistribution.NORMAL
    assert noise_values(noise, size=2).tolist() == [0.5, 0.25]
    # and the scale is one for data without errors
    assert noise_values(default_noise_model(None), size=2).tolist() == [1.0, 1.0]
