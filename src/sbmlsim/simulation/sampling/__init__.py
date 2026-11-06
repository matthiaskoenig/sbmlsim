"""The sampler: distributions and the designs of a scan.

A design is a dimension of a scan whose values are drawn from distributions,
see `sbmlsim.simulation.sampling.designs`; it carries the record of how it was
drawn, `Design`, which the result keeps as its provenance. A distribution
gives the values of a target at probabilities of the unit cube, so every
design works with every marginal; a distribution without a location is
relative to the reference of its target, see `references`. The distributions
which a model states for its parameters with distrib are read by
`parameter_distributions` and drawn by `uncertainty_dimension`.
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
from sbmlsim.simulation.sampling.model_distributions import (
    UNCERTAINTY_DIMENSION,
    distribution_from_formula,
    parameter_distributions,
    uncertainty_dimension,
)
from sbmlsim.simulation.sampling.population import population
from sbmlsim.simulation.sampling.references import parameters_of, references
from sbmlsim.simulation.scan import Design

__all__ = [
    "UNCERTAINTY_DIMENSION",
    "Design",
    "Distribution",
    "Empirical",
    "Fixed",
    "LogNormal",
    "LogUniform",
    "Normal",
    "Truncated",
    "Uniform",
    "distribution_from_formula",
    "fast",
    "fit_parameters",
    "fit_repeats",
    "lhs",
    "local",
    "morris",
    "parameter_distributions",
    "parameters_of",
    "population",
    "profile_parameters",
    "random",
    "references",
    "sobol",
    "uncertainty_dimension",
]
