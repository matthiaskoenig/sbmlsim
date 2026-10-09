"""Virtual populations: covariates drawn and mapped to the values of targets.

The covariates (e.g. the body weight, the age) are drawn by `random` or
`lhs`, and a function of a module maps them to the values of targets of the
model, `function(covariates) -> {target: values}`, vectorized over the points.
The covariates are coordinates of the dimension, which the result carries and
no model sees, so an observable can be plotted against them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.observables import _check_function
from sbmlsim.simulation.sampling.designs import lhs, random
from sbmlsim.simulation.sampling.distributions import Distribution
from sbmlsim.simulation.scan import Design, Dimension
from sbmlsim.simulator.simulator import ModelLike

#: the designs which draw the covariates of a population
METHODS = {"random": random, "lhs": lhs}


def population(
    function: Callable[[dict[str, Any]], Mapping[str, Any]],
    covariates: Mapping[str, Distribution],
    n: int,
    *,
    seed: int | None = None,
    method: str = "random",
    model: ModelLike | None = None,
    simulation: Simulation | None = None,
    id: str = "population",
) -> Dimension:
    """Get a virtual population, see the module.

    Args:
        function: a function of a module, `covariates -> {target: values}`, the
            values of every target for every point.
        covariates: covariate -> its distribution.
        n: the number of individuals.
        seed: the seed; `None` draws one, which the record keeps.
        method: `random` or `lhs`, the design of the covariates.
        model: the model, which a relative distribution of a covariate needs.
        simulation: the simulation whose pre-initialization gives the references.
        id: the id of the dimension.

    Returns:
        The dimension, the targets as its values and the covariates as its
        coordinates.

    Raises:
        ValueError: if the function is no function of a module or returns no
            mapping of targets to `n` values, a covariate as a target, or the
            method is unknown.
    """
    _check_function(function, id)
    if method not in METHODS:
        raise ValueError(
            f"The method of a population is one of {sorted(METHODS)}, not '{method}'."
        )
    drawn = METHODS[method](
        covariates, n, seed=seed, model=model, simulation=simulation, id=id
    )
    values = dict(drawn.values)
    targets = function(values)
    if not isinstance(targets, Mapping):
        raise ValueError(
            f"The function of the population '{id}' returns a mapping of targets to "
            f"values, not {targets!r}."
        )
    for target, column in targets.items():
        if target in values:
            raise ValueError(
                f"The function of the population '{id}' returns the covariate "
                f"'{target}' as a target; a covariate is never set on a model."
            )
        length = len(np.atleast_1d(getattr(column, "magnitude", column)))
        if length != len(drawn):
            raise ValueError(
                f"The function of the population '{id}' returns {length} values of "
                f"'{target}' for {len(drawn)} individuals; the values must have the "
                f"length of the population."
            )
    record = drawn.design
    if record is None:
        raise ValueError(f"The covariates of the population '{id}' have no record.")
    name = (
        f"{getattr(function, '__module__', '')}:{getattr(function, '__qualname__', '')}"
    )
    design = Design(
        method="population",
        distributions=record.distributions,
        options={
            **record.options,
            "method": method,
            "function": name,
            "covariates": list(covariates),
        },
        references=record.references,
    )
    return Dimension(id, values=dict(targets), coordinates=values, design=design)
