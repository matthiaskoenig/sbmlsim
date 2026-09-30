"""A hook of the derived changes for the tests, which needs no network.

The hook scales an entity of the model by a parameter of the fit which is
not an entity of the model, i.e. what a network before the simulation does
without a network. It lives in a module because the workers of a parallel fit
import it when they unpickle the problem.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

import numpy as np

from sbmlsim.fit import FitParameter
from sbmlsim.fit.derived import HookSummary
from sbmlsim.fit.objects import EXTERNAL_PREFIX
from sbmlsim.fit.options import ParameterScaleType

#: the entity of the hctz model the hook writes
TARGET = "KI__HCTZEX_k"

#: the parameter of the fit the hook reads, which is not an entity
FACTOR = "factor_k"


@dataclass(frozen=True)
class Scaling:
    """The change `target = factor * value of the model`.

    Attributes:
        model: id of the model in the experiment.
        target: the entity which is set.
        factor: the parameter of the fit, the entity or the constant which is
            read.
        frozen: ids the fit must not write.
        constants: id -> value of the symbols which are neither entities of
            the model nor parameters of the fit.
        calls: the conditions the changes were calculated for.
    """

    model: str = "model"
    target: str = TARGET
    factor: str = FACTOR
    frozen: frozenset[str] = frozenset()
    constants: Mapping[str, float] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list, compare=False)

    def summary(self) -> HookSummary:
        return HookSummary(
            name="scaling",
            kind="scaling",
            description=f"{self.target} = {self.factor} * {self.target}",
            targets=(self.target,),
            groups=(),
        )

    def symbols(self) -> Collection[str]:
        return {self.factor, self.target}

    def targets(self) -> Collection[str]:
        return {self.target}

    def check_parameters(self, targets: Collection[str]) -> None:
        frozen = sorted(set(targets) & self.frozen)
        if frozen:
            raise ValueError(f"the parameters write {frozen}, which are frozen")

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        self.calls.append(condition)
        variables = {**self.constants, **values}
        return {self.target: variables[self.factor] * variables[self.target]}


def factor_parameter(start: float = 1.0, pid: str = FACTOR) -> FitParameter:
    """Get the parameter of the fit which is not an entity of the model."""
    return FitParameter(
        pid,
        start,
        lower_bound=0.0,
        upper_bound=np.inf,
        unit="dimensionless",
        target=f"{EXTERNAL_PREFIX}{pid}",
        scale=ParameterScaleType.LINEAR,
    )
