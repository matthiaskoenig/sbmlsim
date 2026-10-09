"""The sampler: distributions and the designs of a scan.

A design is a dimension of a scan whose values are drawn from distributions,
see `sbmlsim.simulation.sampling.designs`; it carries the record of how it was
drawn, `Design`, which the result keeps as its provenance. A distribution
gives the values of a target at probabilities of the unit cube, so every
design works with every marginal; a distribution without a location is
relative to the reference of its target, see `references`.
"""

from sbmlsim.simulation.sampling.designs import lhs, local, random
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
    "lhs",
    "local",
    "parameters_of",
    "random",
    "references",
]
