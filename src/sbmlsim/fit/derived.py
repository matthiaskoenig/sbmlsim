"""Changes of a simulation which are calculated from the parameters of a fit.

A fit writes the values of its parameters into the model as the changes of a
simulation. A derived change is a change which is not a parameter but a
function of them: before every simulation the problem hands the values of the
parameters, of the changes of the simulation and of the model to the objects
it was given as `hybridizations`, and adds the changes they answer with to
the simulation.

`DerivedChanges` is what such an object provides. The neural networks of a
hybrid problem are the implementation, see
`sbmlsim.sciml.hybridization.Hybridization`: a network which runs before the
simulation calculates parameters and initial values of the model from
parameters of the fit.

`resolve_derived_changes` binds the hooks to the simulation groups of a
problem when it is initialized and refuses what would make the objective flat
or wrong, `evaluate_derived_changes` calculates the changes of a group before
its simulation. `OptimizationProblem` calls both.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sbmlsim.units import Quantity

if TYPE_CHECKING:
    from sbmlsim.fit.objects import FitParameter
    from sbmlsim.fit.optimization import OptimizationProblem
    from sbmlsim.model import RoadrunnerSBMLModel
    from sbmlsim.simulation import TimecourseSim
    from sbmlsim.simulator import SimulatorSerial


@dataclass(frozen=True)
class ParameterGroup:
    """Parameters of a fit which a hook holds as one array.

    The console and the report show the group as one row, because a network
    adds hundreds of elements to a fit.

    Attributes:
        label: what the group is, e.g. `net1.layer1.weight`.
        ids: the ids of its elements, estimated or not, in the order of the
            array.
    """

    label: str
    ids: tuple[str, ...]


@dataclass(frozen=True)
class HookSummary:
    """What the console and the report say about a hook.

    Attributes:
        name: id of the hook, e.g. of the network.
        kind: where it sits, e.g. the pattern of a network.
        description: what it is made of, e.g. the layers of a network.
        targets: the entities it sets.
        groups: its parameters, as the groups the tables show.
    """

    name: str
    kind: str
    description: str
    targets: tuple[str, ...]
    groups: tuple[ParameterGroup, ...]


@runtime_checkable
class DerivedChanges(Protocol):
    """The changes of a simulation which follow from the values of a fit."""

    def summary(self) -> HookSummary:
        """Describe the hook for the console and the report."""
        ...

    @property
    def model(self) -> str:
        """Get the id of the model in the experiment the changes belong to."""
        ...

    @property
    def constants(self) -> Mapping[str, float]:
        """Get the values of the symbols the object provides itself.

        Returns:
            id -> value of the symbols which are neither entities of the
            model nor parameters of the fit.
        """
        ...

    def symbols(self) -> Collection[str]:
        """Get the ids whose values `derived_changes` reads.

        Returns:
            The ids of entities of the model, of parameters of the fit which
            are not entities of the model, and of the `constants`.
        """
        ...

    def targets(self) -> Collection[str]:
        """Get the entities of the model `derived_changes` sets.

        Returns:
            The ids of the entities, or the selections of their
            concentrations.
        """
        ...

    def check_parameters(self, targets: Collection[str]) -> None:
        """Check the targets of the parameters of a fit.

        Args:
            targets: the entities the parameters of the fit write, without the
                prefix of a target which is not an entity of the model.

        Raises:
            ValueError: if the fit writes what the derived changes set or
                hold constant.
        """
        ...

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        """Get the changes of a simulation for the values of the fit.

        Args:
            values: id -> value in the units of the model. The values are the
                ones of the parameters of the fit, of the changes of the
                simulation and of the model, in this order of precedence.
            condition: id of the simulation in its experiment.

        Returns:
            target -> value in the unit of the target in the model, for every
            target.

        Raises:
            ValueError: if a value is missing or a change is not a finite
                number.
        """
        ...


#: the derived changes of a simulation group: every hook of the model of the
#: group with the values of the model which its symbols name
GroupDerivedChanges = list[tuple[DerivedChanges, dict[str, float]]]


def describe(hybridization: DerivedChanges) -> dict[str, Any]:
    """Describe a hook for the serialization of a problem.

    Args:
        hybridization: the hook.

    Returns:
        The type, the model, the targets and the summary of the hook.
    """
    summary = hybridization.summary()
    return {
        "type": type(hybridization).__name__,
        "model": hybridization.model,
        "targets": sorted(hybridization.targets()),
        "summary": {
            "name": summary.name,
            "kind": summary.kind,
            "description": summary.description,
            "arrays": [group.label for group in summary.groups],
        },
    }


def hook_summaries(hooks: Iterable[DerivedChanges]) -> list[HookSummary]:
    """Get the summaries of the hooks of a problem, in their order."""
    return [hook.summary() for hook in hooks]


def group_parameters(
    parameters: Sequence[FitParameter], summaries: Iterable[HookSummary]
) -> tuple[list[FitParameter], list[tuple[ParameterGroup, list[FitParameter]]]]:
    """Split the parameters of a fit into single ones and the groups of the hooks.

    The elements of a group are found by the entity they write
    (`FitParameter.entity_id`), so a parameter of an element which is versioned
    under another id belongs to the group as well. A group which several hooks
    share, e.g. a network which is used twice, is listed once, at its first
    occurrence.

    Args:
        parameters: the parameters of the fit.
        summaries: the summaries of the hooks of the problem.

    Returns:
        The parameters which belong to no group, in their order, and every
        group with the parameters of the fit which are its elements. A group
        without a parameter of the fit is listed with none, i.e. an array
        which is frozen.
    """
    by_entity: dict[str, list[FitParameter]] = {}
    for parameter in parameters:
        by_entity.setdefault(parameter.entity_id, []).append(parameter)
    grouped: set[str] = set()
    seen: set[ParameterGroup] = set()
    groups: list[tuple[ParameterGroup, list[FitParameter]]] = []
    for summary in summaries:
        for group in summary.groups:
            if group in seen:
                continue
            seen.add(group)
            members = [p for sid in group.ids for p in by_entity.get(sid, [])]
            grouped.update(group.ids)
            groups.append((group, members))
    single = [p for p in parameters if p.entity_id not in grouped]
    return single, groups


def _some(ids: list[str], n: int = 20) -> str:
    """Get the first ids of a list and how many there are, for a message.

    The ids a hook reads are the elements of a network, which can be
    thousands.
    """
    if len(ids) <= n:
        return str(ids)
    return f"{ids[:n]} ... ({len(ids)} in total)"


def _entity(target: str) -> str:
    """Get the entity a target names, `S` for the concentration `[S]`."""
    if target.startswith("[") and target.endswith("]"):
        return target[1:-1]
    return target


def resolve_derived_changes(problem: OptimizationProblem) -> list[GroupDerivedChanges]:
    """Bind the hooks of a problem to its simulation groups.

    The derived changes of a group are the hooks of its model. The values of
    the model which they read are stored with them, from the model as it was
    loaded: a simulation changes the state of the model. The simulations of
    the problem must not have been simulated yet.

    Args:
        problem: the problem, with its fit mappings resolved into simulation
            groups and its `parameter_mapping`.

    Returns:
        The derived changes of every simulation group.

    Raises:
        ValueError: if a hook names a model no fit mapping is simulated with,
            if a parameter which is not an entity of the model and one which
            is reach a group with one id, if a parameter of the fit writes
            what a hook sets or holds constant, if a hook sets what is not an
            entity of the model or what the first timecourse of the
            simulation changes, if two hooks set one entity, if a hook reads
            what another one sets, if a hook reads a constant of its own
            which a parameter of the fit writes, if a hook reads a symbol
            which is neither an entity of the model, nor a parameter of the
            fit in the group, nor one of its constants, or if a parameter
            which is not an entity of the model is read by no hook.
    """
    model_keys = problem.model_keys
    unknown = sorted({h.model for h in problem.hybridizations} - set(model_keys))
    if unknown:
        raise ValueError(
            f"'{problem.opid}': the hybridizations name the models {unknown}, "
            f"but the fit mappings are simulated with the models "
            f"{sorted(set(model_keys))}."
        )
    mapping = problem.parameter_mapping_initialized
    group_derived: list[GroupDerivedChanges] = []
    read: set[str] = set()
    for k_group, group in enumerate(problem.mapping_groups):
        k0 = group[0]
        parameters = [
            problem.parameters[index] for index in mapping.indices_for(k_group).values()
        ]
        where = f"'{problem.opid}': in the simulation '{mapping.group_names[k_group]}'"
        _check_entity_ids(where, parameters)
        hooks = [h for h in problem.hybridizations if h.model == model_keys[k0]]
        derived = _group_derived_changes(
            where,
            hooks,
            parameters,
            problem.models[k0],
            problem.simulations[k0].timecourses[0].changes,
        )
        read.update(symbol for h, _ in derived for symbol in h.symbols())
        group_derived.append(derived)

    for parameter in problem.parameters:
        if parameter.is_external and parameter.entity_id not in read:
            raise ValueError(
                f"'{problem.opid}': FitParameter '{parameter.pid}' writes "
                f"'{parameter.target_id}', which is not an entity of a model "
                f"and which no hybridization reads. The objective does not "
                f"depend on the parameter."
            )
    return group_derived


def _check_entity_ids(where: str, parameters: Sequence[FitParameter]) -> None:
    """Refuse a parameter which is not an entity with the id of one which is.

    The hooks read the values of the parameters by `FitParameter.entity_id`.

    Args:
        where: the problem and the simulation group, for the message.
        parameters: the parameters of the fit in the group.

    Raises:
        ValueError: if an external parameter and a parameter of an entity of
            the model have one id.
    """
    external = {p.entity_id: p for p in parameters if p.is_external}
    for parameter in parameters:
        other = external.get(parameter.target_id)
        if other is not None and not parameter.is_external:
            raise ValueError(
                f"{where} the parameters '{other.pid}' (target "
                f"'{other.target_id}') and '{parameter.pid}' (target "
                f"'{parameter.target_id}') both reach the hybridizations as "
                f"'{parameter.target_id}'."
            )


def _group_derived_changes(
    where: str,
    hooks: Sequence[DerivedChanges],
    parameters: Sequence[FitParameter],
    model: RoadrunnerSBMLModel,
    changes: Mapping[str, Any],
) -> GroupDerivedChanges:
    """Check the hooks of a simulation group and read their values of the model.

    Args:
        where: the problem and the simulation group, for the message.
        hooks: the hooks of the model of the group.
        parameters: the parameters of the fit in the group.
        model: the model of the group, as it was loaded.
        changes: the changes of the first timecourse of the simulation, before
            it is simulated.

    Returns:
        The hooks with the values of the model which they read.

    Raises:
        ValueError: see `resolve_derived_changes`.
    """
    uinfo = model.uinfo
    entities = [p.entity_id for p in parameters]
    changed = {_entity(key) for key in changes}
    written: dict[str, DerivedChanges] = {}
    for hook in hooks:
        hook.check_parameters(entities)
        for target in sorted(hook.targets()):
            prefix = (
                f"{where} a hybridization of the model '{hook.model}' sets '{target}'"
            )
            entity = _entity(target)
            if target not in uinfo:
                raise ValueError(f"{prefix}, which is not an entity of the model.")
            writers = sorted(
                p.pid
                for p in parameters
                if not p.is_external and _entity(p.target_id) == entity
            )
            if writers:
                raise ValueError(
                    f"{prefix}, which the parameters {writers} of the fit write. "
                    f"An entity is estimated or derived."
                )
            if entity in changed:
                raise ValueError(
                    f"{prefix}, which the first timecourse of the simulation "
                    f"changes. The derived change would replace that change."
                )
            if entity in written:
                raise ValueError(
                    f"{where} two hybridizations of the model '{hook.model}' "
                    f"set '{entity}'."
                )
            written[entity] = hook

    derived: GroupDerivedChanges = []
    for hook in hooks:
        symbols = set(hook.symbols())
        prefix = f"{where} a hybridization of the model '{hook.model}' reads"
        # a hook reads the value of the model of what it sets
        chained = sorted(
            s for s in symbols if written.get(_entity(s), hook) is not hook
        )
        if chained:
            raise ValueError(
                f"{prefix} {chained}, which a hybridization sets. Derived "
                f"changes are calculated from the parameters of the fit and "
                f"the model, not from each other."
            )
        # the value of the fit would silently replace the constant
        pids = {p.entity_id: p.pid for p in parameters}
        both = sorted(set(hook.constants) & symbols & set(pids))
        if both:
            names = ", ".join(f"'{s}' (FitParameter '{pids[s]}')" for s in both)
            raise ValueError(
                f"{prefix} {names} as a constant of the hybridization and as a "
                f"parameter of the fit. A symbol is constant or estimated."
            )
        missing = sorted(
            s
            for s in symbols
            if s not in uinfo and s not in entities and s not in hook.constants
        )
        if missing:
            raise ValueError(
                f"{prefix} {_some(missing)}, which are neither entities of the model, "
                f"nor parameters of the fit in the simulation, nor constants "
                f"of the hybridization."
            )
        derived.append((hook, model_values(model, symbols)))
    return derived


def model_values(
    model: RoadrunnerSBMLModel, symbols: Collection[str]
) -> dict[str, float]:
    """Get the values of the entities of a model which are symbols.

    Args:
        model: the model as it was loaded, with its changes.
        symbols: ids, of which some are entities of the model.

    Returns:
        id -> value of the symbols which are entities of the model.

    Raises:
        ValueError: if the model is not loaded in roadrunner, or if
            roadrunner has no value for an entity.
    """
    if model.r is None:
        raise ValueError(f"Model '{model}' is not loaded in roadrunner.")
    values: dict[str, float] = {}
    for symbol in sorted(symbols):
        if symbol not in model.uinfo:
            # a parameter of the fit or a constant of the hook
            continue
        if symbol in model.changes:
            change = model.changes[symbol]
            values[symbol] = float(
                change.magnitude if isinstance(change, Quantity) else change
            )
            continue
        try:
            values[symbol] = float(model.r[symbol])
        except (RuntimeError, TypeError, ValueError) as err:
            raise ValueError(
                f"Model '{model}': the entity '{symbol}' has no value: {err}"
            ) from err
    return values


def evaluate_derived_changes(
    problem: OptimizationProblem,
    k_group: int,
    simulation: TimecourseSim,
    simulator: SimulatorSerial,
    quantities: Sequence[Quantity],
) -> dict[str, Quantity]:
    """Get the derived changes of the simulation of a group.

    The values the hooks calculate from are the values of the parameters of
    the fit, the changes of the first timecourse of the simulation and the
    values of the model, in this order of precedence and in the units of the
    model.

    Args:
        problem: the initialized problem.
        k_group: index of the simulation group.
        simulation: the simulation of the group with the changes of the
            parameters, normalized to the units of the model.
        simulator: simulator of the problem, with the model of the group.
        quantities: the quantity of every parameter, in the order of the
            parameter vector.

    Returns:
        The changes by entity of the model, in the unit of the entity.

    Raises:
        ValueError: if a hook cannot calculate its changes, or if it answers
            with other changes than the ones of its targets.
    """
    uinfo = simulator.uinfo
    Q_ = problem.runner_initialized.Q_
    mapping = problem.parameter_mapping_initialized
    k0 = problem.mapping_groups[k_group][0]
    derived = problem.group_derived[k_group]
    written = {target for h, _ in derived for target in h.targets()}

    changes: dict[str, float] = {
        key: float(value.magnitude if isinstance(value, Quantity) else value)
        for key, value in simulation.timecourses[0].changes.items()
        # a derived change of the last evaluation is not a value
        if key not in written
    }
    fitted: dict[str, float] = {}
    for index in mapping.indices_for(k_group).values():
        parameter = problem.parameters[index]
        quantity = quantities[index]
        if not parameter.is_external and parameter.target_id in uinfo:
            quantity = quantity.to(uinfo[parameter.target_id])
        fitted[parameter.entity_id] = float(quantity.magnitude)

    condition = problem.simulation_keys[k0]
    result: dict[str, Quantity] = {}
    for hook, model_values_ in derived:
        values: Mapping[str, float] = {**model_values_, **changes, **fitted}
        hook_changes = hook.derived_changes(values, condition=condition)
        targets = set(hook.targets())
        if set(hook_changes) != targets:
            extra = sorted(set(hook_changes) - targets)
            absent = sorted(targets - set(hook_changes))
            clauses = []
            if extra:
                clauses.append(f"with the changes {extra} which are not its targets")
            if absent:
                clauses.append(f"without its targets {absent}")
            raise ValueError(
                f"'{problem.opid}': in the simulation "
                f"'{mapping.group_names[k_group]}' a hybridization of the model "
                f"'{hook.model}' answers {' and '.join(clauses)}."
            )
        for target, value in hook_changes.items():
            result[target] = Q_(value, uinfo[target])
    return result
