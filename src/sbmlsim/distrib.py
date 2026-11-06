"""The formulas of the distributions of the fitted parameters, without a simulator.

A fitted parameter carries the distribution of its draws as a formula of the
SBML package distrib: `normal(mean, sd[, lower, upper])`,
`lognormal(mu, sigma[, lower, upper])` (mu and sigma of the natural logarithm)
and `uniform(lower, upper)`, with numbers as arguments, truncated to the bounds
of the fit when they are given. This module is the one grammar of these
formulas: the draws of the fitted parameters parse them into the distributions
of `sbmlsim.simulation.sampling`, and a library which must not import a
simulator, e.g. the PBPK library of pkpdsim, reads their support to bound a
fitted fraction. It imports only the standard library, so its import loads
neither roadrunner nor the sampler.
"""

import math
import re
from dataclasses import dataclass

__all__ = ["ARGUMENTS", "DistribFormula", "parse", "support"]

ARGUMENTS: dict[str, tuple[int, ...]] = {
    "normal": (2, 4),
    "lognormal": (2, 4),
    "uniform": (2,),
}
"""The distributions and their numbers of arguments; the last two of four are
the bounds of a truncation."""

_CALL = re.compile(r"^\s*(\w+)\s*\((.*)\)\s*$")


@dataclass(frozen=True)
class DistribFormula:
    """A formula of a distribution of distrib.

    Attributes:
        name: the distribution, a key of `ARGUMENTS`.
        args: the arguments.
    """

    name: str
    args: tuple[float, ...]

    @property
    def bounds(self) -> tuple[float, float] | None:
        """The bounds of a truncated or uniform distribution, `None` without."""
        if self.name == "uniform" or len(self.args) == 4:
            return self.args[-2], self.args[-1]
        return None


def parse(formula: str) -> DistribFormula:
    """Parse a formula of a distribution.

    Args:
        formula: e.g. `normal(0.3, 0.05, 0.01, 0.6)`.

    Returns:
        The distribution and its arguments.

    Raises:
        ValueError: if the formula is no function of numbers, or no
            distribution of `ARGUMENTS` with a supported number of arguments.
    """
    match = _CALL.match(formula)
    if match is None:
        raise ValueError(f"The distribution '{formula}' is no function of numbers.")
    try:
        args = tuple(float(a) for a in match.group(2).split(","))
    except ValueError:
        raise ValueError(
            f"The distribution '{formula}' has arguments which are no numbers."
        ) from None
    name = match.group(1)
    if len(args) not in ARGUMENTS.get(name, ()):
        raise ValueError(
            f"The distribution '{formula}' is not supported: lognormal and normal "
            f"with two or four arguments, uniform with two."
        )
    return DistribFormula(name, args)


def support(formula: str) -> tuple[float, float]:
    """Get the support of a distribution: the values its draws can take.

    Args:
        formula: the formula of the distribution.

    Returns:
        The bounds of a truncated or uniform distribution; without bounds
        `normal` is unbounded and `lognormal` positive; a formula which
        `parse` refuses is unbounded (its values cannot be bounded).
    """
    try:
        distribution = parse(formula)
    except ValueError:
        return -math.inf, math.inf
    bounds = distribution.bounds
    if bounds is not None:
        return bounds
    if distribution.name == "lognormal":
        return 0.0, math.inf
    return -math.inf, math.inf
