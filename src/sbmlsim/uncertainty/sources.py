"""The sources of an uncertainty analysis: what is uncertain and how.

An `UncertaintyModel` collects the uncertain values of an analysis, apart
from the simulations it is applied to: the inputs, i.e. the `Uncertain`
values of the simulations (`inputs=True`), the parameters with their
distributions or uncertain values (`parameters`), and the dependence between
them (`covariance`, see `sbmlsim.uncertainty.Covariance`).

A `Variable` is one random variable of the analysis: an uncertain value used
for one or several targets (the same object, or objects of the same `name`),
with its distribution. The uncertain values of one key in several scenarios
are one variable, also with distributions which differ between the scenarios,
e.g. a dose of 100 mg and of 200 mg: a draw takes the same quantile of the
distribution of every scenario, so the scenarios are paired (common random
numbers); within one scenario a key has one distribution. A draw of the analysis gives every variable a value,
which replaces each of its targets wherever a simulation sets it (the rule of
`Simulation.with_values`). A target which a simulation sets to an uncertain
value and elsewhere to another value cannot be replaced in one place alone
and raises, and so does a parameter which a simulation sets: its draw would
replace the value of the simulation, e.g. the inhibition of a scenario.
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
        by_scenario: scenario -> its distribution where it differs from
            `distribution`, the one of the first scenario: an input of one key
            whose distribution differs between scenarios, e.g. a dose of 100
            mg in one and of 200 mg in another, is drawn at the same quantile
            in every scenario (common random numbers) from its own marginal.
    """

    key: str
    targets: tuple[str, ...]
    distribution: Distribution
    kind: str
    source: Literal["input", "parameter"]
    by_scenario: Mapping[str, Distribution] = field(default_factory=dict)

    def marginal(self, scenario: str | None = None) -> Distribution:
        """Get the distribution of the variable in a scenario."""
        if scenario is None:
            return self.distribution
        return self.by_scenario.get(scenario, self.distribution)


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
                distributions, a parameter which is an input as well, or a
                parameter which a simulation sets.
        """
        # key -> its targets, its value in the first scenario and its
        # distributions in the scenarios which differ
        found: dict[str, tuple[list[str], Uncertain, dict[str, Distribution]]] = {}
        # an object used for several targets is one variable, of its first key
        keys: dict[int, str] = {}
        if self.inputs:
            for name, simulation in simulations.items():
                _check_targets(name, simulation)
                # key -> its value in this scenario
                here: dict[str, Uncertain] = {}
                for use in simulation.uncertain_values():
                    key = keys.setdefault(id(use.value), _key(use.target, use.value))
                    known = here.setdefault(key, use.value)
                    if not _same(known, use.value):
                        raise ValueError(
                            f"The uncertain values of '{key}' in the simulation "
                            f"'{name}' differ: {known!r} and {use.value!r}; one "
                            f"random variable has one distribution in a simulation."
                        )
                    if key not in found:
                        found[key] = ([use.target], use.value, {})
                        continue
                    targets, first, scenarios = found[key]
                    if use.target not in targets:
                        targets.append(use.target)
                    if use.value.kind != first.kind:
                        raise ValueError(
                            f"The uncertain values of '{key}' are of the kinds "
                            f"'{first.kind}' and '{use.value.kind}'; one random "
                            f"variable has one kind."
                        )
                    if use.value.distribution.to_dict() != (
                        first.distribution.to_dict()
                    ):
                        scenarios[name] = use.value.distribution
        variables = [
            Variable(
                key,
                tuple(targets),
                value.distribution,
                value.kind,
                "input",
                by_scenario=scenarios,
            )
            for key, (targets, value, scenarios) in found.items()
        ]
        inputs = {t for v in variables for t in v.targets}
        for target, value in self.parameters.items():
            if target in inputs:
                raise ValueError(
                    f"The parameter '{target}' is an uncertain value of a "
                    f"simulation as well; declare it once."
                )
            if target in found:
                raise ValueError(
                    f"The parameter '{target}' is the name of an uncertain value "
                    f"of a simulation as well; one key is one random variable, "
                    f"rename one of them."
                )
            for name, simulation in simulations.items():
                uses = _uses(simulation).get(target)
                if uses:
                    raise ValueError(
                        f"The simulation '{name}' sets the parameter '{target}' "
                        f"at the times {[t for t, _ in uses]} (None is before "
                        f"the initialization): a draw of the parameter would "
                        f"replace the value of the simulation. Set another "
                        f"target in the simulation, e.g. a factor or an "
                        f"activity of the parameter, or declare the value of "
                        f"the simulation uncertain."
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


def _same(a: object, b: object) -> bool:
    """Check whether two uses are of one random variable in a simulation.

    The same object, or two equal uncertain values of the same name; two
    objects without a name are two values, which one target cannot take.
    """
    return a is b or (
        isinstance(a, Uncertain)
        and isinstance(b, Uncertain)
        and a.name is not None
        and a.to_dict() == b.to_dict()
    )


def _uses(simulation: Simulation) -> dict[str, list[tuple[Any, object]]]:
    """Get the values a simulation sets its targets to, with their times.

    Returns:
        Target -> (time, value) of every use, an uncertain value as declared;
        the time is `None` before the initialization and `presimulation` for
        the pre-initialization of the steady state before it.
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
    return uses


def _check_targets(name: str, simulation: Simulation) -> None:
    """Check that an uncertain target is set by its uncertain value alone.

    Raises:
        ValueError: if a target is set to an uncertain value and elsewhere to
            another value or another uncertain value.
    """
    for target, values in _uses(simulation).items():
        uncertain = [v for _, v in values if isinstance(v, Uncertain)]
        if not uncertain:
            continue
        first = uncertain[0]
        if any(not _same(v, first) for _, v in values):
            times = [t for t, _ in values]
            raise ValueError(
                f"The simulation '{name}' sets '{target}' to the uncertain value "
                f"{first!r} and to other values, at the times {times} (None is "
                f"before the initialization): a draw replaces the target "
                f"wherever it is set. Use one uncertain value for every use, or "
                f"a target of its own per use."
            )
