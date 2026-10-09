"""The sampler: distributions and the designs of a scan.

A design is a dimension of a scan whose values are drawn from distributions,
see `sbmlsim.simulation.sampling.designs`; it carries the record of how it was
drawn, `Design`, which the result keeps as its provenance. A distribution
gives the values of a target at probabilities of the unit cube, so every
design works with every marginal; a distribution without a location is
relative to the reference of its target, see `references`.
"""

from sbmlsim.simulation.sampling.designs import (
    fast,
    lhs,
    local,
    morris,
    random,
    sobol,
)
from sbmlsim.simulation.sampling.distributions import (
    Distribution,
    Empirical,
    Fixed,
    LogNormal,
    LogUniform,
    Normal,
    Truncated,
    Uniform,
)
from sbmlsim.simulation.sampling.fit import (
    fit_parameters,
    fit_repeats,
    profile_parameters,
)
from sbmlsim.simulation.sampling.references import parameters_of, references
from sbmlsim.simulation.scan import Design

__all__ = [
    "Design",
    "Distribution",
    "Empirical",
    "Fixed",
    "LogNormal",
    "LogUniform",
    "Normal",
    "Truncated",
    "Uniform",
    "fast",
    "fit_parameters",
    "fit_repeats",
    "lhs",
    "local",
    "morris",
    "parameters_of",
    "profile_parameters",
    "random",
    "references",
    "sobol",
]
