"""Where a network sits in a hybrid problem.

A `Hybridization` connects a `Network` to a model: what its inputs are, which
entities of the model its outputs set, and in which of three places it runs.

| pattern | inputs | outputs | executed by |
| --- | --- | --- | --- |
| `PRE_INITIALIZATION` | constants: formulas of parameters, arrays | parameters and initial values, set before the simulation | numpy, once per simulation |
| `RHS` | formulas of species, parameters and time, arrays | parameters of the rate equations | roadrunner, as assignment rules |
| `OBSERVABLE` | formulas of species, parameters and time, arrays | symbols of an observable | roadrunner, as assignment rules |

A network before the simulation is evaluated by `derived_changes`, which a
fit calls for every simulation with the values of its parameters. The
networks of the other two patterns are compiled into the model by
`sbmlsim.sciml.compiler`; `derived_changes` sets only their inputs which are
arrays of a condition.

The condition of a simulation is the id of the simulation in its experiment.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
from numpy.typing import ArrayLike

from sbmlsim.fit.derived import HookSummary, ParameterGroup
from sbmlsim.fit.objects import FitParameter
from sbmlsim.mathml import TIME, evaluate_formula, formula_symbols
from sbmlsim.sciml.backend import BackendKind
from sbmlsim.sciml.errors import NetworkHybridizationError, NetworkImportError
from sbmlsim.sciml.interpreter import PLACEHOLDER
from sbmlsim.sciml.network import (
    Network,
    input_id,
    output_id,
    parse_io_id,
)
from sbmlsim.sciml.parameters import network_fit_parameters

#: the condition of the arrays and formulas which hold for every condition
ALL_CONDITIONS = "0"

#: relative tolerance of the values of the frozen elements in a compiled
#: model, libsbml writes 15 significant digits
FROZEN_RTOL = 1e-14


class NetworkPattern(StrEnum):
    """The place of a network in a hybrid problem."""

    PRE_INITIALIZATION = "pre_initialization"
    RHS = "rhs"
    OBSERVABLE = "observable"

    @property
    def is_compiled(self) -> bool:
        """Check whether a network of the pattern is compiled into the model."""
        return self is not NetworkPattern.PRE_INITIALIZATION


@dataclass(frozen=True, eq=False)
class NetworkInput:
    """An input of a network or an element of one.

    Exactly one of the attributes is given.

    Attributes:
        formula: an L3 formula of SBML which holds for every condition, e.g.
            `prey`, `alpha * prey` or `0.5`.
        formulas: id of the condition -> formula, for an input which differs
            between the conditions. `ALL_CONDITIONS` is the formula of the
            conditions which are not listed.
        arrays: id of the condition -> values, `ALL_CONDITIONS` for the values
            of the conditions which are not listed. The arrays of an input
            have one shape.
    """

    formula: str | None = None
    formulas: Mapping[str, str] | None = None
    arrays: Mapping[str, ArrayLike] | None = None

    def __post_init__(self) -> None:
        """Validate the input and copy its arrays.

        Raises:
            ValueError: if not exactly one attribute is given, if a formula
                is not valid math, if no condition is given, or if the arrays
                differ in their shape or hold a value which is not finite.
        """
        given = [
            name
            for name in ("formula", "formulas", "arrays")
            if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise ValueError(
                f"An input is a formula, formulas of conditions or arrays of "
                f"conditions, but {given or 'nothing'} is given"
            )
        if self.formulas is not None:
            if not self.formulas:
                raise ValueError("The formulas of an input name no condition")
            object.__setattr__(self, "formulas", dict(self.formulas))
        if self.arrays is not None:
            if not self.arrays:
                raise ValueError("The arrays of an input name no condition")
            arrays = {}
            for condition, values in self.arrays.items():
                array = np.array(values, dtype=float)
                if not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"The array of the condition '{condition}' holds values "
                        f"which are not finite"
                    )
                array.setflags(write=False)
                arrays[condition] = array
            shapes = {condition: a.shape for condition, a in arrays.items()}
            if len(set(shapes.values())) > 1:
                raise ValueError(
                    f"The arrays of an input have one shape, but the shapes of "
                    f"the conditions are {shapes}"
                )
            object.__setattr__(self, "arrays", arrays)
        for formula in self.all_formulas():
            formula_symbols(formula)

    def all_formulas(self) -> list[str]:
        """Get the formulas of the input, empty for an input of arrays."""
        if self.formula is not None:
            return [self.formula]
        return list((self.formulas or {}).values())

    @property
    def is_conditional(self) -> bool:
        """Check whether the input differs between the conditions."""
        values = self.formulas if self.formulas is not None else self.arrays
        return values is not None and set(values) != {ALL_CONDITIONS}

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Get the shape of the arrays, `None` for an input of formulas."""
        if self.arrays is None:
            return None
        return np.shape(next(iter(self.arrays.values())))

    def formula_of(self, condition: str) -> str | None:
        """Get the formula of a condition.

        Args:
            condition: id of the condition.

        Returns:
            The formula, `None` for an input of arrays and for a condition
            without a formula.
        """
        if self.formula is not None:
            return self.formula
        if self.formulas is None:
            return None
        return self.formulas.get(condition, self.formulas.get(ALL_CONDITIONS))

    def array_of(self, condition: str) -> np.ndarray | None:
        """Get the array of a condition.

        Args:
            condition: id of the condition.

        Returns:
            The array, `None` for an input of formulas and for a condition
            without an array.
        """
        if self.arrays is None:
            return None
        array = self.arrays.get(condition, self.arrays.get(ALL_CONDITIONS))
        return None if array is None else np.asarray(array, dtype=float)

    def __eq__(self, other: object) -> bool:
        """Check whether two inputs are equal, the arrays element by element."""
        if not isinstance(other, NetworkInput):
            return NotImplemented
        if self.formula != other.formula or self.formulas != other.formulas:
            return False
        if self.arrays is None or other.arrays is None:
            return self.arrays is None and other.arrays is None
        return self.arrays.keys() == other.arrays.keys() and all(
            np.array_equal(array, other.arrays[condition])
            for condition, array in self.arrays.items()
        )

    def __hash__(self) -> int:
        """Get the hash of the formula, equal inputs have one formula."""
        return hash(self.formula)


def input_shapes(
    network: Network, inputs: Mapping[str, NetworkInput]
) -> list[tuple[int, ...]]:
    """Get the shapes of the inputs of the forward pass of a network.

    The shape of an input which is given as an array is the shape of its
    arrays. The shape of an input which is given element by element follows
    from the indices of its elements, which have to cover it.

    Args:
        network: the network.
        inputs: id of the input -> the input, see `Hybridization`.

    Returns:
        The shape of every input, in the order of the forward pass.

    Raises:
        NetworkHybridizationError: if an id is not the id of an input, if an
            input is given as an array and element by element, if an input of
            the forward pass is missing, or if the elements of an input do
            not cover a shape.
    """
    sid = network.sid

    def error(message: str) -> NetworkHybridizationError:
        return NetworkHybridizationError(f"Network '{sid}': {message}")

    n_inputs = sum(node.op == PLACEHOLDER for node in network.model.forward)
    arrays: dict[int, tuple[int, ...]] = {}
    elements: dict[int, set[tuple[int, ...]]] = {}
    for key, network_input in inputs.items():
        if not isinstance(network_input, NetworkInput):
            raise error(f"the input '{key}' is not a `NetworkInput`")
        try:
            k, index = parse_io_id(sid, "input", key)
        except ValueError as err:
            raise NetworkHybridizationError(str(err)) from err
        if k >= n_inputs:
            raise error(
                f"'{key}' is the input {k}, but the forward pass has {n_inputs} inputs"
            )
        shape = network_input.shape
        if index is None:
            if shape is None:
                raise error(
                    f"the input '{key}' is a formula, which is the value of one "
                    f"element: name the element, e.g. '{input_id(sid, k, (0,))}'"
                )
            arrays[k] = shape
        else:
            if shape not in (None, ()):
                raise error(
                    f"the input '{key}' is an element, but its arrays have the "
                    f"shape {shape}"
                )
            elements.setdefault(k, set()).add(index)
    both = sorted(set(arrays) & set(elements))
    if both:
        raise error(f"the inputs {both} are given as an array and element by element")
    shapes: list[tuple[int, ...]] = []
    for k in range(n_inputs):
        if k in arrays:
            shapes.append(arrays[k])
            continue
        if k not in elements:
            raise error(
                f"the input {k} of the forward pass is missing, the inputs are "
                f"{sorted(inputs)}"
            )
        indices = elements[k]
        ndims = {len(index) for index in indices}
        if len(ndims) != 1:
            raise error(
                f"the elements of the input {k} differ in their number of axes: "
                f"{sorted(indices)}"
            )
        shape = tuple(
            max(index[axis] for index in indices) + 1 for axis in range(ndims.pop())
        )
        missing = sorted(set(np.ndindex(shape)) - indices)
        if missing:
            raise error(
                f"the input {k} has the shape {shape}, but its elements {missing} "
                f"are missing"
            )
        shapes.append(shape)
    return shapes


def output_shapes(
    network: Network, shapes: list[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    """Get the shapes of the outputs of a network for inputs of given shapes.

    Args:
        network: the network, with the values of its arrays.
        shapes: the shape of every input, in the order of the forward pass.

    Returns:
        The shape of every output, in the order of the forward pass.

    Raises:
        NetworkHybridizationError: if the network cannot be evaluated on
            inputs of the shapes.
        NetworkImportError: if an array of the network has no values.
    """
    try:
        outputs = network.forward(*[np.zeros(shape) for shape in shapes])
    except NetworkImportError:
        raise
    except ValueError as err:
        raise NetworkHybridizationError(
            f"Network '{network.sid}': inputs of the shapes {shapes} do not fit "
            f"the network: {err}"
        ) from err
    return [output.shape for output in outputs]


def entity_of(target: str) -> str:
    """Get the entity of the model a target names.

    Args:
        target: the entity or the selection of its concentration, e.g. `prey`
            or `[prey]`.

    Returns:
        The id of the entity, e.g. `prey`.
    """
    if target.startswith("[") and target.endswith("]"):
        return target[1:-1]
    return target


@dataclass(frozen=True)
class Hybridization:
    """A network with its inputs, its outputs and its place in a problem.

    The hybridization is validated when it is created, as far as it can be
    without the model: `validate` checks it against the model. Two
    hybridizations are equal when their attributes are; a hybridization is
    not hashable, its attributes are dictionaries.

    Attributes:
        network: the network.
        pattern: where the network sits.
        model: id of the model in the experiment.
        inputs: id of the input -> the input. The id is `<net>__input<k>` for
            an input which is given as an array and
            `<net>__input<k>__<index>` for an element of an input which is
            given element by element, see `sbmlsim.sciml.network.input_id`.
        outputs: id of the element of an output -> target, see
            `sbmlsim.sciml.network.output_id`. The target is an entity of the
            model for `RHS`, an entity or the selection of its concentration
            (`[prey]`) for `PRE_INITIALIZATION`, and the symbol an observable
            uses for `OBSERVABLE`, which the compiler adds to the model. An
            output without a target is not used.
        frozen: ids of the elements of the arrays which are not estimated.
        constants: id -> value of the symbols of the formulas which are
            neither entities of the model nor parameters of the fit.
    """

    network: Network
    pattern: NetworkPattern
    model: str
    inputs: Mapping[str, NetworkInput]
    outputs: Mapping[str, str]
    frozen: Collection[str] = field(default_factory=frozenset)
    constants: Mapping[str, float] = field(default_factory=dict)
    #: the shapes of the inputs of the forward pass, see `input_shapes`
    _input_shapes: tuple[tuple[int, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    #: the sorted ids and the values of the frozen elements in the network
    _frozen_values: tuple[tuple[str, ...], np.ndarray] = field(
        init=False, repr=False, compare=False
    )

    __hash__ = None

    def __post_init__(self) -> None:
        """Validate the hybridization against its network.

        Raises:
            NetworkHybridizationError: if the pattern is not a pattern, if an
                id is not the id of an input or output of the network, if
                the inputs do not cover the inputs of the forward pass, if
                the outputs are not outputs of the network, if two outputs
                have one target, if an element which is frozen is not an
                element of the network, if `frozen` is a single id, if a
                constant is not a finite number, or if a network which is
                compiled has a layer without expressions or an input which
                differs in its formula between conditions.
            NetworkImportError: if an array of the network has no values.
        """
        object.__setattr__(self, "pattern", self._member(NetworkPattern, self.pattern))
        object.__setattr__(self, "inputs", dict(self.inputs))
        object.__setattr__(self, "outputs", dict(self.outputs))
        if isinstance(self.frozen, str):
            raise self.error(
                f"the frozen elements are a collection of ids, not the id "
                f"'{self.frozen}'"
            )
        object.__setattr__(self, "frozen", frozenset(self.frozen))
        object.__setattr__(self, "constants", self._numbers(self.constants))
        if not isinstance(self.model, str) or not self.model:
            raise self.error(f"the id of the model '{self.model}' is not an id")
        unknown = sorted(set(self.frozen) - set(self.network.parameter_ids()))
        if unknown:
            raise self.error(
                f"the frozen elements {unknown} are not elements of the network"
            )
        self.network.check_arrays(self.network.parameters, complete=True)

        shapes = input_shapes(self.network, self.inputs)
        object.__setattr__(self, "_input_shapes", tuple(shapes))
        # the elements of an array without values are not compiled
        arrays = self.network.parameters
        frozen = {
            sid: float(arrays[layer][name][index])
            for sid, (layer, name, index) in self.network.parameter_ids().items()
            if sid in self.frozen and name in arrays.get(layer, {})
        }
        ids = tuple(sorted(frozen))
        values = np.array([frozen[sid] for sid in ids], dtype=float)
        object.__setattr__(self, "_frozen_values", (ids, values))
        if self.pattern.is_compiled:
            if BackendKind.SYMPY not in self.network.backends():
                raise self.error(
                    f"the pattern '{self.pattern.value}' compiles the network into "
                    f"the model, but it has layers or functions which are not "
                    f"evaluated on expressions, e.g. a convolution. Such a "
                    f"network runs before the simulation, i.e. with the pattern "
                    f"'{NetworkPattern.PRE_INITIALIZATION.value}'"
                )
            for sid, network_input in self.inputs.items():
                if network_input.formulas is not None:
                    raise self.error(
                        f"the input '{sid}' has the formulas of the conditions "
                        f"{sorted(network_input.formulas)}, but a network of the "
                        f"pattern '{self.pattern.value}' is a part of the model "
                        f"and has one formula per input"
                    )
        self._check_outputs(shapes)

    def _member(self, enum: type[StrEnum], value: Any) -> Any:
        """Get the member of an enumeration with a value.

        Raises:
            NetworkHybridizationError: if no member has the value.
        """
        try:
            return enum(value)
        except ValueError as err:
            raise self.error(
                f"'{value}' is not one of {[member.value for member in enum]}"
            ) from err

    def _numbers(self, constants: Mapping[str, Any]) -> dict[str, float]:
        """Get the constants as floats.

        Raises:
            NetworkHybridizationError: if a constant is not a finite number.
        """
        numbers: dict[str, float] = {}
        for sid, value in constants.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = np.nan
            if not np.isfinite(number):
                raise self.error(f"the constant '{sid}' is '{value}', not a number")
            numbers[sid] = number
        return numbers

    def error(self, message: str) -> NetworkHybridizationError:
        """Get an error which names the network.

        Args:
            message: what is wrong.

        Returns:
            The error, to be raised.
        """
        return NetworkHybridizationError(f"Network '{self.network.sid}': {message}")

    def input_shapes(self) -> list[tuple[int, ...]]:
        """Get the shapes of the inputs of the forward pass, see `input_shapes`."""
        return list(self._input_shapes)

    def output_shapes(self) -> list[tuple[int, ...]]:
        """Get the shapes of the outputs of the network for its inputs."""
        return output_shapes(self.network, self.input_shapes())

    def _check_outputs(self, input_shapes: list[tuple[int, ...]]) -> None:
        """Check the outputs against the network, see `__post_init__`."""
        sid = self.network.sid
        if not self.outputs:
            raise self.error("no output has a target, the network is not used")
        shapes = self.output_shapes()
        targets: dict[str, str] = {}
        for key, target in self.outputs.items():
            try:
                k, index = parse_io_id(sid, "output", key)
            except ValueError as err:
                raise NetworkHybridizationError(str(err)) from err
            if index is None:
                raise self.error(
                    f"the output '{key}' names no element, e.g. "
                    f"'{output_id(sid, k, (0,))}'"
                )
            if k >= len(shapes):
                raise self.error(
                    f"'{key}' is the output {k}, but the forward pass has "
                    f"{len(shapes)} outputs"
                )
            if len(index) != len(shapes[k]) or not all(
                0 <= i < n for i, n in zip(index, shapes[k], strict=True)
            ):
                raise self.error(
                    f"'{key}' is not an element of the output {k}, which has the "
                    f"shape {shapes[k]} for inputs of the shapes {input_shapes}"
                )
            if not isinstance(target, str) or not entity_of(target):
                raise self.error(f"the target '{target}' of '{key}' is not an id")
            entity = entity_of(target)
            if entity in targets:
                raise self.error(
                    f"the outputs '{targets[entity]}' and '{key}' both set '{entity}'"
                )
            targets[entity] = key

    # --- WHAT A FIT NEEDS ---

    def fit_parameters(
        self,
        estimate: Mapping[str, bool],
        bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> tuple[list[FitParameter], Hybridization]:
        """Get the parameters of a fit of the network and freeze the rest.

        The pattern decides whether the elements are entities of the model,
        see `sbmlsim.sciml.parameters.network_fit_parameters`, and every
        element which is not estimated is frozen.

        Args:
            estimate: key of the entry -> whether the elements are estimated,
                for the network, a layer or an array.
            bounds: key of the entry -> lower and upper bound, none by
                default.

        Returns:
            The parameters of the fit and the hybridization with the other
            elements frozen.

        Raises:
            NetworkImportError: if a key does not name the network, a layer
                or an array, or if an estimated element has no value.
        """
        parameters = network_fit_parameters(
            self.network,
            estimate=estimate,
            bounds=bounds or {},
            external=not self.pattern.is_compiled,
        )
        frozen = set(self.network.parameter_ids()) - {p.pid for p in parameters}
        return parameters, replace(self, frozen=frozen)

    def summary(self) -> HookSummary:
        """Describe the network for the console and the report.

        Returns:
            The id of the network, its pattern, its layers with their types
            in the order of the forward pass, the targets of its outputs and
            one group per array of the layers the forward pass calls, with
            the ids of all elements of the array.
        """
        network = self.network
        types = {layer.layer_id: layer.layer_type for layer in network.model.layers}
        used = network.used_layers()
        groups: dict[tuple[str, str], list[str]] = {}
        for sid, (layer, name, _) in network.parameter_ids().items():
            if layer in used:
                groups.setdefault((layer, name), []).append(sid)
        return HookSummary(
            name=network.sid,
            kind=self.pattern.value,
            description=", ".join(f"{layer} ({types[layer]})" for layer in used),
            targets=tuple(sorted(self.outputs.values())),
            groups=tuple(
                ParameterGroup(label=f"{network.sid}.{layer}.{name}", ids=tuple(ids))
                for (layer, name), ids in groups.items()
            ),
        )

    def symbols(self) -> frozenset[str]:
        """Get the ids whose values `derived_changes` reads.

        Returns:
            The symbols of the formulas of the inputs and the ids of the
            elements which are not frozen, for a network before the
            simulation. For a network which is compiled, which the model
            evaluates, the ids of the frozen elements and of the outputs:
            the model must have them, and `derived_changes` checks that the
            model carries the values of the frozen elements.
        """
        if self.pattern.is_compiled:
            return frozenset(self._frozen_values[0]) | frozenset(self.outputs)
        symbols = {
            symbol
            for network_input in self.inputs.values()
            for formula in network_input.all_formulas()
            for symbol in formula_symbols(formula)
        }
        elements = set(self.network.parameter_ids()) - set(self.frozen)
        return frozenset(symbols | elements)

    def targets(self) -> frozenset[str]:
        """Get the entities of the model `derived_changes` sets.

        The targets are the keys of `derived_changes` as they are written:
        the id of an entity or the selection of the concentration of a
        species (`[prey]`), which `sbmlsim.fit.derived` compares by their
        entity (`entity_of`).

        Returns:
            The targets of the outputs for a network before the simulation,
            and the ids of the elements of the inputs which are arrays of
            conditions for a network which is compiled.
        """
        if not self.pattern.is_compiled:
            return frozenset(self.outputs.values())
        return frozenset(sid for sid, _ in self._conditional_elements(condition=None))

    def _conditional_elements(
        self, condition: str | None
    ) -> list[tuple[str, float | None]]:
        """Get the elements of the inputs which are arrays of conditions.

        Args:
            condition: id of the condition, `None` for the ids alone.

        Returns:
            The id of the element and its value in the condition, for every
            input which differs between the conditions.

        Raises:
            NetworkHybridizationError: if an input has no array for the
                condition.
        """
        elements: list[tuple[str, float | None]] = []
        for key, network_input in self.inputs.items():
            if network_input.arrays is None or not network_input.is_conditional:
                continue
            k, index = parse_io_id(self.network.sid, "input", key)
            array = None if condition is None else network_input.array_of(condition)
            if condition is not None and array is None:
                raise self.error(
                    f"the input '{key}' has no array for the condition "
                    f"'{condition}', it has arrays for "
                    f"{sorted(network_input.arrays)}"
                )
            shape = network_input.shape or ()
            for element in np.ndindex(shape):
                sid = (
                    key if index is not None else input_id(self.network.sid, k, element)
                )
                elements.append((sid, None if array is None else float(array[element])))
        return elements

    def check_parameters(self, targets: Collection[str]) -> None:
        """Check the targets of the parameters of a fit against the network.

        Args:
            targets: the entities the parameters of the fit write, without
                the prefix of a target which is not an entity of the model.

        Raises:
            NetworkHybridizationError: if a parameter writes an element which
                is frozen, an output or its target, or an element of an input.
        """
        frozen = sorted(set(targets) & set(self.frozen))
        if frozen:
            raise self.error(
                f"the elements {frozen} are frozen, but parameters of the fit "
                f"write them. An element is frozen or estimated"
            )
        outputs = {entity_of(target) for target in self.outputs.values()}
        outputs |= set(self.outputs)
        written = sorted({entity_of(target) for target in targets} & outputs)
        if written:
            raise self.error(
                f"{written} are set by the outputs of the network, but "
                f"parameters of the fit write them. An entity is estimated or "
                f"calculated by the network"
            )

    def input_values(
        self, values: Mapping[str, float], condition: str
    ) -> list[np.ndarray]:
        """Get the inputs of the network for the values of a simulation.

        Args:
            values: id -> value of the parameters of the fit, of the changes
                of the simulation and of the entities of the model, in the
                units of the model. The constants of the hybridization are
                the values of the symbols which are not part of it.
            condition: id of the condition of the simulation.

        Returns:
            The inputs, one array per input of the forward pass.

        Raises:
            NetworkHybridizationError: if an input has no formula or array
                for the condition, if a symbol of a formula has no value, or
                if the value of a formula is not defined or not a finite
                number.
        """
        variables = {**self.constants, **values}
        sid = self.network.sid
        inputs = [np.zeros(shape) for shape in self.input_shapes()]
        for key, network_input in self.inputs.items():
            k, index = parse_io_id(sid, "input", key)
            if network_input.arrays is not None:
                array = network_input.array_of(condition)
                if array is None:
                    raise self.error(
                        f"the input '{key}' has no array for the condition "
                        f"'{condition}', it has arrays for "
                        f"{sorted(network_input.arrays)}"
                    )
                if index is None:
                    inputs[k] = np.array(array, dtype=float)
                else:
                    inputs[k][index] = float(array)
                continue
            formula = network_input.formula_of(condition)
            if formula is None:
                raise self.error(
                    f"the input '{key}' has no formula for the condition "
                    f"'{condition}', it has formulas for "
                    f"{sorted(network_input.formulas or {})}"
                )
            try:
                value = np.asarray(evaluate_formula(formula, variables), dtype=float)
            except (ArithmeticError, TypeError, ValueError) as err:
                raise self.error(f"input '{key}': {err}") from err
            if value.shape != () or not np.isfinite(value):
                raise self.error(
                    f"input '{key}': the formula '{formula}' has the value "
                    f"'{value}', which is not a finite number"
                )
            if index is None:
                raise self.error(f"the input '{key}' is a formula without an element")
            inputs[k][index] = float(value)
        return inputs

    def derived_changes(
        self, values: Mapping[str, float], condition: str
    ) -> dict[str, float]:
        """Get the changes of a simulation which follow from its values.

        A network before the simulation is evaluated: its inputs are resolved
        with `input_values`, its arrays are the nominal values with the
        values of the elements which are not frozen, which the fit estimates,
        and its outputs are the changes of their targets. For a network which
        is compiled the changes are the elements of the inputs which are
        arrays of conditions, and the values of its frozen elements are
        compared with the network, which is how a model which was compiled
        from another network is found.

        Args:
            values: id -> value, see `input_values`. The values of the
                elements of the network which are not frozen are read from it
                by their id, the values of the frozen ones are ignored.
            condition: id of the condition of the simulation.

        Returns:
            target -> value, in the unit of the target in the model.

        Raises:
            NetworkHybridizationError: if an input cannot be resolved, see
                `input_values`, if an element which is not frozen has no
                value, if an output is not a finite number, or if the values
                of the frozen elements of a network which is compiled are
                missing or differ from the network.
        """
        if self.pattern.is_compiled:
            self._check_compiled_values(values)
            return {
                sid: float(value)
                for sid, value in self._conditional_elements(condition)
                if value is not None
            }
        inputs = self.input_values(values, condition)
        estimated = [
            sid for sid in self.network.parameter_ids() if sid not in self.frozen
        ]
        missing = [sid for sid in estimated if sid not in values]
        if missing:
            raise self.error(
                f"the elements {missing[:5]}{' ...' if len(missing) > 5 else ''} "
                f"({len(missing)} of {len(estimated)} which are not frozen) have "
                f"no value. The elements which are not frozen are estimated, "
                f"the fit gives their values"
            )
        elements = {sid: float(values[sid]) for sid in estimated}
        outputs = self.network.forward(
            *inputs, parameters=self.network.with_values(elements)
        )
        changes: dict[str, float] = {}
        for key, target in self.outputs.items():
            k, index = parse_io_id(self.network.sid, "output", key)
            value = float(outputs[k][index])
            if not np.isfinite(value):
                raise self.error(
                    f"the output '{key}' is '{value}' in the condition "
                    f"'{condition}', which is not a finite number"
                )
            changes[target] = value
        return changes

    def _check_compiled_values(self, values: Mapping[str, float]) -> None:
        """Check the values of the frozen elements of a compiled network.

        The model gives the values of the frozen elements, which are the ones
        of the network when it was compiled, see `FROZEN_RTOL`.

        Args:
            values: id -> value, which holds the values of the model.

        Raises:
            NetworkHybridizationError: if a frozen element has no value or a
                value which differs from the network.
        """
        ids, expected = self._frozen_values
        missing = [sid for sid in ids if sid not in values]
        if missing:
            raise self.error(
                f"the frozen elements {_some(missing)} have no value, the model "
                f"of a compiled network gives them"
            )
        actual = np.array([values[sid] for sid in ids], dtype=float)
        differ = np.abs(actual - expected) > FROZEN_RTOL * np.maximum(
            np.abs(actual), np.abs(expected)
        )
        if np.any(differ):
            others = [sid for sid, d in zip(ids, differ, strict=True) if d]
            raise self.error(
                f"the model carries other values of the frozen elements "
                f"{_some(others)} than the network, compile the network again"
            )

    # --- THE MODEL ---

    def validate(self, sbml_path: Path) -> None:
        """Check the hybridization against the model.

        Args:
            sbml_path: the SBML model the network is a part of, without the
                network.

        Raises:
            NetworkHybridizationError: if the model cannot be read, if a
                target cannot be set by the network, see `_check_target`, if
                a constant is an entity of the model, if a formula uses a
                symbol which is an output of the network, or if an input of
                `PRE_INITIALIZATION` is not a constant of the simulation,
                i.e. depends on the time, a species, a reaction, a species
                reference, an entity which a rule sets or an event changes.
        """
        _, model = read_model(sbml_path, self.network.sid)
        for key, target in self.outputs.items():
            self._check_target(model, sbml_path.name, key, target)
        for sid in self.constants:
            if model.getElementBySId(sid) is not None:
                raise self.error(
                    f"the constant '{sid}' is an entity of the model "
                    f"'{sbml_path.name}', the model gives its value"
                )
        outputs = {entity_of(target) for target in self.outputs.values()}
        for key, network_input in self.inputs.items():
            for formula in network_input.all_formulas():
                symbols = formula_symbols(formula)
                circular = sorted(symbols & outputs)
                if circular:
                    raise self.error(
                        f"the input '{key}' uses {circular}, which the outputs "
                        f"of the network set"
                    )
                if self.pattern is not NetworkPattern.PRE_INITIALIZATION:
                    continue
                for symbol in sorted(symbols):
                    reason = _varies(model, symbol)
                    if reason is not None:
                        raise self.error(
                            f"the input '{key}' of a network which runs before "
                            f"the simulation is a constant, but its formula "
                            f"'{formula}' uses '{symbol}', which is {reason}"
                        )

    def _check_target(
        self, model: libsbml.Model, name: str, key: str, target: str
    ) -> None:
        """Check that the network can set a target in the model.

        Args:
            model: the model without the network.
            name: the name of the file of the model, for the message.
            key: id of the element of the output.
            target: the target of the element.

        Raises:
            NetworkHybridizationError: if a target of `OBSERVABLE` is a
                concentration, not a valid SBML id or an entity of the model;
                if a target of `RHS` is not a parameter of the model or is set
                by a rule or changed by an event; if a target of
                `PRE_INITIALIZATION` is not a parameter, a species or a
                compartment, or is set by an assignment rule; or if a
                concentration is not the one of a species before the
                simulation.
        """
        entity = entity_of(target)
        element: libsbml.SBase | None = model.getElementBySId(entity)
        about = f"the target '{target}' of '{key}'"
        if self.pattern is NetworkPattern.OBSERVABLE:
            if target != entity:
                raise self.error(
                    f"{about} is the symbol of an observable and not a concentration"
                )
            if not libsbml.SyntaxChecker.isValidSBMLSId(target):
                raise self.error(
                    f"{about} is not a valid SBML id, but it is the symbol of an "
                    f"observable, which is added to the model"
                )
            if element is not None:
                raise self.error(
                    f"{about} is the symbol of an observable, which is added to "
                    f"the model, but the model '{name}' has an entity '{entity}'"
                )
            return
        if element is None:
            raise self.error(f"{about} is not an entity of the model '{name}'")
        if target != entity and (
            self.pattern is NetworkPattern.RHS or model.getSpecies(entity) is None
        ):
            raise self.error(
                f"{about} is a concentration, which is the initial value of a "
                f"species for the pattern "
                f"'{NetworkPattern.PRE_INITIALIZATION.value}'"
            )
        kind = _kind(element)
        rule: libsbml.Rule | None = model.getRuleByVariable(entity)
        if self.pattern is NetworkPattern.RHS:
            if element.getTypeCode() != libsbml.SBML_PARAMETER:
                raise self.error(
                    f"{about} is {kind}, but a network in the right hand side "
                    f"sets a parameter by an assignment rule"
                )
            reason = (
                "set by a rule"
                if rule is not None
                else "changed by an event"
                if _changed_by_event(model, entity)
                else None
            )
            if reason is not None:
                raise self.error(
                    f"{about} is {reason} of the model '{name}', but the network "
                    f"sets it by an assignment rule"
                )
            return
        if element.getTypeCode() not in _INITIAL_VALUES:
            raise self.error(
                f"{about} is {kind}, but a network before the simulation sets "
                f"a parameter, a species or a compartment"
            )
        if rule is not None and rule.isAssignment():
            raise self.error(
                f"{about} is set by an assignment rule of the model '{name}', "
                f"which replaces the value the network sets before the "
                f"simulation"
            )


def read_model(
    sbml_path: Path, network: str
) -> tuple[libsbml.SBMLDocument, libsbml.Model]:
    """Read the model of an SBML file.

    Args:
        sbml_path: the SBML file.
        network: id of the network, for the message.

    Returns:
        The document and its model. The model is a part of the document and
        is valid as long as the document is referenced.

    Raises:
        NetworkHybridizationError: if the file does not exist or holds no
            model.
    """
    if not Path(sbml_path).is_file():
        raise NetworkHybridizationError(
            f"Network '{network}': the model '{sbml_path}' does not exist"
        )
    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(sbml_path))
    model: libsbml.Model | None = document.getModel()
    if model is None:
        raise NetworkHybridizationError(
            f"Network '{network}': '{sbml_path}' holds no SBML model"
        )
    return document, model


#: the kinds of the entities of a model, for the messages
_KINDS: dict[int, str] = {
    libsbml.SBML_PARAMETER: "a parameter",
    libsbml.SBML_SPECIES: "a species",
    libsbml.SBML_COMPARTMENT: "a compartment",
    libsbml.SBML_REACTION: "a reaction",
    libsbml.SBML_SPECIES_REFERENCE: "a species reference",
}

#: the kinds of the entities a network before the simulation sets
_INITIAL_VALUES = frozenset(
    {libsbml.SBML_PARAMETER, libsbml.SBML_SPECIES, libsbml.SBML_COMPARTMENT}
)


def _some(ids: list[str], n: int = 5) -> str:
    """Get the first ids of a list and how many there are, for a message."""
    if len(ids) <= n:
        return str(ids)
    return f"{ids[:n]} ... ({len(ids)} in total)"


def _kind(element: libsbml.SBase) -> str:
    """Get the kind of an element of a model, e.g. `a species`."""
    return _KINDS.get(element.getTypeCode(), f"a {element.getElementName()}")


def _changed_by_event(model: libsbml.Model, sid: str) -> bool:
    """Check whether an event of a model assigns an entity."""
    return any(
        model.getEvent(k).getEventAssignment(sid) is not None
        for k in range(model.getNumEvents())
    )


def _varies(model: libsbml.Model, symbol: str) -> str | None:
    """Get why the value of a symbol is not a constant of a simulation.

    Args:
        model: the model.
        symbol: a symbol of a formula.

    Returns:
        The reason, `None` for a symbol which is a constant.
    """
    if symbol == TIME:
        return "the time"
    if model.getSpecies(symbol) is not None:
        return "a species"
    if model.getReaction(symbol) is not None:
        return "a reaction, whose rate depends on the state"
    if model.getSpeciesReference(symbol) is not None:
        return "a species reference"
    if model.getRuleByVariable(symbol) is not None:
        return "set by a rule of the model"
    if _changed_by_event(model, symbol):
        return "changed by an event of the model"
    return None
