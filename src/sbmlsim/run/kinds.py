"""The kinds of analyses of the run layer: a spec and the functions which run it.

A `Kind` is an analysis the run layer runs: a pydantic `Spec` of its settings
and the functions of a run. `plan` completes a spec into what every unit of a
split run computes alike (a drawn seed, resolved ids, the number of `units`),
`run` runs the whole analysis in one process and returns its summary, and a
splittable kind computes the same as units, `compute_unit` one unit and
`merge` the result of all of them, so a split run and a whole run give the
same result.

The kinds of sbmlsim are registered in `KINDS` by the modules of
`_BUILTIN_MODULES`, which `get_kind` imports when it is first asked; a kind
outside of the registry is referenced as `module:attribute`, e.g. the kind of a
model package.

`import sbmlsim.run.kinds` imports pydantic and nothing of the analyses.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from sbmlsim.run.context import RunContext

__all__ = [
    "KINDS",
    "ConfigurationError",
    "Kind",
    "PlanMismatch",
    "Spec",
    "SplitSpec",
    "UnknownKind",
    "get_kind",
    "register",
]


class ConfigurationError(Exception):
    """A run is configured wrongly, e.g. an invalid spec or an output in use.

    The command line exits with 4 for it.
    """


class PlanMismatch(ConfigurationError):
    """A planned spec does not match what it resolves to, e.g. another scan."""


class UnknownKind(KeyError, ConfigurationError):
    """A kind which is not registered."""

    @override
    def __str__(self) -> str:
        """Get the message without the quotes of a `KeyError`."""
        return str(self.args[0]) if self.args else ""


class Spec(BaseModel):
    """The settings of an analysis, the base of the spec of every kind.

    A spec forbids unknown fields and is frozen, so its hash identifies a run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitSpec(Spec):
    """The spec of a kind which splits into units.

    Attributes:
        units: the number of units, set by the plan of the kind.
    """

    units: int | None = None


def _identity[S](spec: S) -> S:
    """Plan nothing: the spec as given."""
    return spec


@dataclass(frozen=True)
class Kind[S: Spec]:
    """A kind of analysis: its spec and the functions which run it.

    Attributes:
        name: the name of the kind, e.g. `fit`.
        spec: the class of its spec.
        run: runs the whole analysis into the context and returns its summary.
        plan: completes a spec into a planned spec (seeds, ids, `units`).
        count_units: the number of units of a planned spec, its `units`
            without.
        compute_unit: computes one unit of a planned spec into the context.
        merge: merges the units into the result and returns the summary.
        inputs: the fields of the spec, `Path | None`, which are inputs, i.e.
            the outputs of other runs.
        uses_cores: whether the kind runs on several CPUs.
        cost_setting: the field of the spec a scheduler estimates the cost by.
    """

    name: str
    spec: type[S]
    run: Callable[[S, RunContext], dict[str, Any]]
    plan: Callable[[S], S] = field(default=_identity)
    count_units: Callable[[S], int] | None = None
    compute_unit: Callable[[S, int, RunContext], None] | None = None
    merge: Callable[[S, RunContext], dict[str, Any]] | None = None
    inputs: tuple[str, ...] = ()
    uses_cores: bool = True
    cost_setting: str | None = None

    def __post_init__(self) -> None:
        """Check the kind.

        Raises:
            ValueError: for an input or a cost setting which is no field of the
                spec, a unit without a merge or a merge without a unit, and a
                splittable kind whose units are neither counted nor `units` of
                a `SplitSpec`.
        """
        fields = self.spec.model_fields
        unknown = [name for name in self.inputs if name not in fields]
        if unknown:
            raise ValueError(
                f"The inputs {unknown} of the kind '{self.name}' are no fields "
                f"of its spec '{self.spec.__name__}'."
            )
        if self.cost_setting is not None and self.cost_setting not in fields:
            raise ValueError(
                f"The cost setting '{self.cost_setting}' of the kind "
                f"'{self.name}' is no field of its spec."
            )
        if (self.compute_unit is None) != (self.merge is None):
            raise ValueError(
                f"The kind '{self.name}' needs both compute_unit and merge, or neither."
            )
        if (
            self.splittable
            and self.count_units is None
            and not issubclass(self.spec, SplitSpec)
        ):
            raise ValueError(
                f"The splittable kind '{self.name}' counts no units: give it "
                "count_units or a SplitSpec, whose plan sets units."
            )

    @property
    def splittable(self) -> bool:
        """Whether the kind runs as units and a merge."""
        return self.compute_unit is not None and self.merge is not None


#: the kinds by their name
KINDS: dict[str, Kind[Any]] = {}

#: the modules which register the kinds of sbmlsim, imported by `get_kind`
_BUILTIN_MODULES: tuple[str, ...] = ()

_lock = threading.Lock()
_builtins_loaded = False


def register(kind: Kind[Any]) -> None:
    """Register a kind by its name.

    Args:
        kind: the kind.

    Raises:
        ValueError: for a name which is registered or contains `:`, which
            references a kind.
    """
    if ":" in kind.name:
        raise ValueError(f"The name of a kind has no ':', '{kind.name}' has one.")
    if kind.name in KINDS:
        raise ValueError(f"The kind '{kind.name}' is registered already.")
    KINDS[kind.name] = kind


def _load_builtins() -> None:
    """Import the modules of the kinds of sbmlsim once, which register them."""
    global _builtins_loaded
    with _lock:
        if _builtins_loaded:
            return
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)
        _builtins_loaded = True


def _referenced(reference: str) -> Kind[Any]:
    """Get the kind a reference `module:attribute` names."""
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise ConfigurationError(f"A kind is 'module:attribute', not '{reference}'.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise ConfigurationError(
            f"The module '{module_name}' of the kind '{reference}' is not "
            f"imported: {err}"
        ) from err
    kind = getattr(module, attribute, None)
    if not isinstance(kind, Kind):
        raise ConfigurationError(
            f"'{reference}' is no Kind, it is a '{type(kind).__name__}'."
        )
    return kind


def get_kind(name: str) -> Kind[Any]:
    """Get a kind by its name or by a reference `module:attribute`.

    Args:
        name: the name of a registered kind, or `module:attribute` of a kind.

    Returns:
        The kind.

    Raises:
        UnknownKind: a `KeyError` and a `ConfigurationError` naming the
            kinds, for a name which is not registered.
        ConfigurationError: for a reference to no kind.
    """
    if ":" in name:
        return _referenced(name)
    _load_builtins()
    try:
        return KINDS[name]
    except KeyError:
        raise UnknownKind(
            f"There is no kind '{name}', the kinds are {sorted(KINDS)}."
        ) from None
