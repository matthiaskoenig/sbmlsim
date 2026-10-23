"""The sources of an uncertainty analysis: what is uncertain and how.

An `UncertaintyModel` collects the uncertain values of an analysis, apart
from the simulations it is applied to: the inputs, i.e. the `Uncertain`
values of the simulations (`inputs=True`), the parameters with their
distributions or uncertain values (`parameters`), and the dependence between
them (`covariance`, see `sbmlsim.uncertainty.Covariance`).

A `Variable` is one random variable of the analysis: an uncertain value used
for one or several targets (the same object, or objects of the same `name`),
with its distribution. A draw of the analysis gives every variable a value,
which replaces each of its targets wherever a simulation sets it (the rule of
`Simulation.with_values`). A target which a simulation sets to an uncertain
value and elsewhere to another value cannot be replaced in one place alone
and raises.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from sbmlsim.simulation.definition import Simulation
from sbmlsim.simulation.sampling.distributions import Distribution, LogNormal
from sbmlsim.uncertainty.covariance import Covariance
from sbmlsim.uncertainty.quantity import Uncertain
from sbmlsim.units import Quantity, ureg


@dataclass(frozen=True)
class Variable:
    """A random variable of an uncertainty analysis.

    Attributes:
        key: the key of the variable: the name of its uncertain value, else
            its first target.
        targets: the targets the value of the variable replaces.
        distribution: the distribution.
        kind: `uncertainty` or `variability`.
        source: `input` (an uncertain value of a simulation) or `parameter`.
    """

    key: str
    targets: tuple[str, ...]
    distribution: Distribution
    kind: str
    source: Literal["input", "parameter"]


def _key(target: str, value: Uncertain) -> str:
    """Get the key of a variable of an uncertain value."""
    return value.name if value.name else target


@dataclass(frozen=True)
class UncertaintyModel:
    """What is uncertain in an analysis, see the module.

    Attributes:
        parameters: target -> its uncertain value or distribution.
        covariance: the dependence between the variables, by their keys.
        inputs: whether the uncertain values of the simulations are
            variables.
    """

    parameters: Mapping[str, Uncertain | Distribution] = field(default_factory=dict)
    covariance: Sequence[Covariance] = ()
    inputs: bool = True

    def __post_init__(self) -> None:
        """Keep the covariances as a tuple."""
        object.__setattr__(self, "covariance", tuple(self.covariance))

    def variables(self, simulations: Mapping[str, Simulation]) -> list[Variable]:
        """Get the variables of the simulations of an analysis.

        Args:
            simulations: the simulations by their name, e.g. the scenarios
                of a comparison.

        Returns:
            The variables: the inputs in the order of their first use, then
            the parameters.

        Raises:
            ValueError: for a target set to an uncertain value and to another
                value, two uncertain values of one name and different
                distributions, or a parameter which is an input as well.
        """
        found: dict[str, tuple[list[str], Uncertain]] = {}
        # an object used for several targets is one variable, of its first key
        keys: dict[int, str] = {}
        if self.inputs:
            for name, simulation in simulations.items():
                _check_targets(name, simulation)
                for use in simulation.uncertain_values():
                    key = keys.setdefault(id(use.value), _key(use.target, use.value))
                    if key in found:
                        targets, known = found[key]
                        if known is not use.value and (
                            known.to_dict() != use.value.to_dict()
                        ):
                            raise ValueError(
                                f"The uncertain values of '{key}' differ: "
                                f"{known!r} and {use.value!r}; one random "
                                f"variable has one distribution."
                            )
                        if use.target not in targets:
                            targets.append(use.target)
                    else:
                        found[key] = ([use.target], use.value)
        variables = [
            Variable(key, tuple(targets), value.distribution, value.kind, "input")
            for key, (targets, value) in found.items()
        ]
        inputs = {t for v in variables for t in v.targets}
        for target, value in self.parameters.items():
            if target in inputs:
                raise ValueError(
                    f"The parameter '{target}' is an uncertain value of a "
                    f"simulation as well; declare it once."
                )
            if isinstance(value, Uncertain):
                variables.append(
                    Variable(
                        target, (target,), value.distribution, value.kind, "parameter"
                    )
                )
            else:
                variables.append(
                    Variable(target, (target,), value, "uncertainty", "parameter")
                )
        return variables

    @classmethod
    def from_covariance(
        cls,
        means: Mapping[str, Quantity | float],
        covariance: Covariance,
        *,
        shape: Literal["auto", "normal", "log-normal"] = "auto",
        **kwargs: Any,
    ) -> UncertaintyModel:
        """Create the uncertainty of parameters of their means and their covariance.

        A covariance on the log scale gives log-normal parameters with the
        means as medians and the variances of the logarithms of its diagonal;
        a covariance on the linear scale gives the uncertain values of the
        means and the standard deviations of its diagonal (`shape="auto"`:
        log-normal for a positive mean, see `Uncertain`).

        Args:
            means: target -> the mean (the median on the log scale).
            covariance: the covariance of the targets.
            shape: the shape of the marginals of a linear covariance.
            kwargs: the other fields of the model.

        Returns:
            The model with the parameters and the covariance.

        Raises:
            ValueError: for a correlation, which has no variances, or means
                which are not the targets of the covariance.
        """
        sds = covariance.sds()
        if sds is None:
            raise ValueError(
                "A correlation has no variances; give the parameters with their "
                "distributions and the correlation as covariance."
            )
        if set(means) != set(covariance.targets):
            raise ValueError(
                f"The means of {sorted(means)} are not the targets "
                f"{list(covariance.targets)} of the covariance."
            )
        parameters: dict[str, Distribution] = {}
        for k, target in enumerate(covariance.targets):
            mean = means[target]
            if covariance.scale == "log":
                cv = math.sqrt(math.expm1(float(sds[k]) ** 2))
                parameters[target] = LogNormal(median=mean, cv=cv)
                continue
            unit = (
                covariance.units[k]
                if covariance.units is not None and covariance.units[k]
                else None
            )
            sd: Quantity | float = (
                ureg.Quantity(float(sds[k]), unit) if unit else float(sds[k])
            )
            parameters[target] = Uncertain(mean, sd=sd, shape=shape).distribution
        return cls(parameters=parameters, covariance=(covariance,), **kwargs)


def _check_targets(name: str, simulation: Simulation) -> None:
    """Check that an uncertain target is set by its uncertain value alone.

    Raises:
        ValueError: if a target is set to an uncertain value and elsewhere to
            another value or another uncertain value.
    """
    uses: dict[str, list[tuple[Any, object]]] = {}
    for target, value in simulation.preinit_changes.items():
        uses.setdefault(target, []).append(
            (None, simulation.uncertain_preinit.get(target, value))
        )
    if simulation.presimulation is not None:
        for target, value in simulation.presimulation.preinit_changes.items():
            uses.setdefault(target, []).append(
                ("presimulation", simulation.presimulation.uncertain.get(target, value))
            )
    for change in simulation.changes:
        for time in change.times:
            for target, value in change.values.items():
                uses.setdefault(target, []).append(
                    (time, change.uncertain.get(target, value))
                )
    for target, values in uses.items():
        uncertain = [v for _, v in values if isinstance(v, Uncertain)]
        if not uncertain:
            continue
        first = uncertain[0]
        if any(v is not first for _, v in values):
            times = [t for t, _ in values]
            raise ValueError(
                f"The simulation '{name}' sets '{target}' to the uncertain value "
                f"{first!r} and to other values, at the times {times} (None is "
                f"before the initialization): a draw replaces the target "
                f"wherever it is set. Use one uncertain value for every use, or "
                f"a target of its own per use."
            )
