"""The compilation of networks into an SBML model.

A network in the right hand side of a model or in an observable is evaluated
by the integrator at every step. `compile_network` writes it into the model as
parameters with assignment rules, which roadrunner simulates like any other
rule:

1. Every element of the arrays is a constant parameter with its nominal value.
2. Every element of an input is a parameter with an assignment rule of its
   formula, or a constant parameter for an element of an array.
3. The forward pass runs on expressions (`SympyBackend`). After every node
   the expressions of its units are replaced by symbols, and every unit is a
   parameter with an assignment rule of its expression. The rules stay one
   layer deep, so the size of the model grows with the number of units and
   not with the depth of the network.
4. Every element of an output is a parameter with an assignment rule, and
   the target of an output has the assignment rule `target = output`. The
   target of `RHS` is a parameter of the model, which becomes variable, the
   target of `OBSERVABLE` is a parameter which is added.

The ids of the parameters are the ids of `sbmlsim.sciml.network`. The
expressions are written as the MathML of SBML by `sbmlmath`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import sympy
from petab_sciml import Node

from sbmlsim.mathml import TIME, expression_to_astnode, formula_symbols
from sbmlsim.sciml.backend import SympyBackend
from sbmlsim.sciml.errors import NetworkCompilationError, NetworkHybridizationError
from sbmlsim.sciml.hybridization import (
    ALL_CONDITIONS,
    Hybridization,
    NetworkInput,
    NetworkPattern,
    read_model,
)
from sbmlsim.sciml.interpreter import evaluate
from sbmlsim.sciml.network import (
    Network,
    input_id,
    output_id,
    parse_io_id,
    unit_id,
)

logger = logging.getLogger(__name__)

#: suffix of the model which carries the networks
MODEL_SUFFIX = "_sciml"

#: the unit of the parameters of a network
UNIT = "dimensionless"

#: the entities whose id is a value in the math of a model
_VALUES = frozenset(
    {
        libsbml.SBML_PARAMETER,
        libsbml.SBML_SPECIES,
        libsbml.SBML_COMPARTMENT,
        libsbml.SBML_REACTION,
        libsbml.SBML_SPECIES_REFERENCE,
    }
)

#: the first level and version of SBML whose math has `max`
_MAX_LEVEL = (3, 2)


def compiled_path(sbml_path: Path, directory: Path | None = None) -> Path:
    """Get the path of the model which carries the networks of a model.

    Args:
        sbml_path: the model without the networks.
        directory: the directory the model is written to, the directory of
            the model by default.

    Returns:
        `<stem>_sciml.xml` in the directory.
    """
    sbml_path = Path(sbml_path)
    parent = sbml_path.parent if directory is None else Path(directory)
    return parent / f"{sbml_path.stem}{MODEL_SUFFIX}{sbml_path.suffix}"


class _Model:
    """The model the networks are written into.

    Attributes:
        document: the SBML document.
        model: its model.
        name: the name of the file of the model, for the messages.
        created: id of every parameter which was added -> what it is.
        constants: id -> value of the constants of the hybridizations which
            were added, which the networks of the model share.
        targets: id of every target which has a rule -> the network and the
            output which set it.
        has_max: whether the math of the level and version of the model has
            `max`, otherwise a maximum is written as a piecewise.
    """

    def __init__(self, sbml_path: Path, network: str) -> None:
        """Read the model, see `read_model`.

        Raises:
            NetworkCompilationError: if the model cannot be read or is not
                valid SBML.
        """
        try:
            self.document, self.model = read_model(sbml_path, network)
        except NetworkHybridizationError as err:
            raise NetworkCompilationError(str(err)) from err
        self.name = Path(sbml_path).name
        self.created: dict[str, str] = {}
        self.constants: dict[str, float] = {}
        self.targets: dict[str, tuple[str, str]] = {}
        level: int = self.document.getLevel()
        version: int = self.document.getVersion()
        self.has_max = (level, version) >= _MAX_LEVEL
        errors = self.errors()
        if errors:
            raise NetworkCompilationError(
                f"Network '{network}': the model '{self.name}' is not valid SBML: "
                f"{errors}"
            )

    def errors(self) -> str:
        """Check the model for consistency.

        Returns:
            The messages of the errors of the model, empty for a valid model.
        """
        self.document.getErrorLog().clearLog()
        self.document.checkConsistency()
        errors = [
            self.document.getError(k)
            for k in range(self.document.getNumErrors())
            if self.document.getError(k).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
        ]
        return "; ".join(error.getMessage().strip() for error in errors)

    def has(self, sid: str) -> bool:
        """Check whether the model has an element of an id, of any kind."""
        return self.model.getElementBySId(sid) is not None

    def is_value(self, sid: str) -> bool:
        """Check whether an id is a value in the math of the model.

        Returns:
            Whether the id is the id of a parameter of the model, a species, a
            compartment, a reaction or a species reference. A local parameter
            of a reaction, an event or a function definition has an id, but
            no value in a rule.
        """
        element: libsbml.SBase | None = self.model.getElementBySId(sid)
        return element is not None and element.getTypeCode() in _VALUES

    def add_parameter(
        self, network: str, sid: str, what: str, value: float | None = None
    ) -> libsbml.Parameter:
        """Add a parameter of a network to the model.

        Args:
            network: id of the network.
            sid: id of the parameter.
            what: what the parameter is, for the messages.
            value: the value of a constant parameter, `None` for a parameter
                which an assignment rule sets.

        Returns:
            The parameter.

        Raises:
            NetworkCompilationError: if the model has an entity of the id, or
                if the id is the id of another parameter of a network.
        """
        if sid in self.created:
            raise NetworkCompilationError(
                f"Network '{network}': {what} and {self.created[sid]} both have "
                f"the id '{sid}'"
            )
        if self.has(sid):
            raise NetworkCompilationError(
                f"Network '{network}': {what} has the id '{sid}', which is the id "
                f"of an entity of the model '{self.name}'"
            )
        parameter: libsbml.Parameter = self.model.createParameter()
        parameter.setId(sid)
        parameter.setUnits(UNIT)
        parameter.setConstant(value is not None)
        if value is not None:
            parameter.setValue(float(value))
        self.created[sid] = what
        return parameter

    def add_constant(self, network: str, sid: str, value: float) -> None:
        """Add a constant of a hybridization, once for all networks.

        Args:
            network: id of the network.
            sid: id of the constant.
            value: its value.

        Raises:
            NetworkCompilationError: if a network added the constant with
                another value, or if the id is taken, see `add_parameter`.
        """
        if sid in self.constants:
            if self.constants[sid] != value:
                raise NetworkCompilationError(
                    f"Network '{network}': the constant '{sid}' has the value "
                    f"'{value}', but another network of the model gives it the "
                    f"value '{self.constants[sid]}'"
                )
            return
        self.add_parameter(network, sid, f"the constant '{sid}'", value=value)
        self.constants[sid] = value

    def add_rule(self, network: str, sid: str, what: str, math: Any) -> None:
        """Add the assignment rule of an entity.

        Args:
            network: id of the network.
            sid: id of the entity the rule sets.
            what: what the entity is, for the messages.
            math: the expression of the rule, or its syntax tree.

        Raises:
            NetworkCompilationError: if the expression has no MathML of SBML.
        """
        rule: libsbml.AssignmentRule = self.model.createAssignmentRule()
        rule.setVariable(sid)
        if isinstance(math, sympy.Basic):
            if not self.has_max:
                math = math.replace(sympy.Max, _piecewise_max)
            try:
                math = expression_to_astnode(math)
            except ValueError as err:
                raise NetworkCompilationError(
                    f"Network '{network}', {what}: {err}"
                ) from err
        if rule.setMath(math) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise NetworkCompilationError(
                f"Network '{network}', {what}: the math of the rule of '{sid}' "
                f"is not math of the model '{self.name}'"
            )

    def write(self, output_path: Path) -> Path:
        """Check the model and write it.

        Args:
            output_path: the path of the model.

        Returns:
            The path.

        Raises:
            NetworkCompilationError: if the model is not valid SBML.
        """
        errors = self.errors()
        if errors:
            raise NetworkCompilationError(
                f"The model '{self.name}' with the networks is not valid SBML: {errors}"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not libsbml.writeSBMLToFile(self.document, str(output_path)):
            raise NetworkCompilationError(
                f"The model with the networks cannot be written to '{output_path}'"
            )
        return output_path


def _piecewise_max(*args: sympy.Expr) -> sympy.Basic:
    """Write the maximum of expressions as a piecewise, for SBML before L3V2.

    The piece of an argument is taken if it is not smaller than the
    arguments after it: the first such argument is the maximum. The size of
    the piecewise grows with the square of the number of arguments, not
    exponentially as a nesting of maxima of two arguments.

    Args:
        args: the arguments of `Max`.

    Returns:
        The piecewise.
    """
    pieces: list[tuple[sympy.Expr, Any]] = [
        (arg, sympy.And(*(arg >= other for other in args[k + 1 :])))
        for k, arg in enumerate(args[:-1])
    ]
    return sympy.Piecewise(*pieces, (args[-1], True))


def _merge(hybridizations: Sequence[Hybridization]) -> list[Hybridization]:
    """Merge the hybridizations of one network, e.g. of its two patterns.

    Args:
        hybridizations: the hybridizations which are compiled.

    Returns:
        One hybridization per network with the outputs and the constants of
        all of them, in the order of the first hybridization of a network. The
        pattern of a merged hybridization is the one of the first,
        `compile_network` keeps the pattern of every output.

    Raises:
        NetworkCompilationError: if two hybridizations of one network differ
            in the network, the model or the inputs, or if two constants of
            one id differ in their value.
    """
    merged: dict[str, Hybridization] = {}
    for hybridization in hybridizations:
        sid = hybridization.network.sid
        if sid not in merged:
            merged[sid] = hybridization
            continue
        first = merged[sid]
        for name in ("network", "model", "inputs"):
            if getattr(first, name) != getattr(hybridization, name):
                raise NetworkCompilationError(
                    f"Network '{sid}': two hybridizations of the network differ "
                    f"in '{name}', a network is compiled once"
                )
        try:
            merged[sid] = Hybridization(
                network=first.network,
                pattern=first.pattern,
                model=first.model,
                inputs=first.inputs,
                outputs={**first.outputs, **hybridization.outputs},
                frozen=set(first.frozen) | set(hybridization.frozen),
                constants=_constants(sid, first, hybridization),
            )
        except NetworkHybridizationError as err:
            raise NetworkCompilationError(str(err)) from err
        if len(merged[sid].outputs) != len(first.outputs) + len(hybridization.outputs):
            raise NetworkCompilationError(
                f"Network '{sid}': two hybridizations of the network use the "
                f"outputs {sorted(set(first.outputs) & set(hybridization.outputs))}"
            )
    return list(merged.values())


def _constants(sid: str, *hybridizations: Hybridization) -> dict[str, float]:
    """Get the constants of hybridizations, see `_merge`."""
    constants: dict[str, float] = {}
    for hybridization in hybridizations:
        for key, value in hybridization.constants.items():
            if key in constants and constants[key] != value:
                raise NetworkCompilationError(
                    f"Network '{sid}': the constant '{key}' has the values "
                    f"'{constants[key]}' and '{value}'"
                )
            constants[key] = value
    return constants


def _check_target(model: _Model, network: str, key: str, target: str) -> None:
    """Check that a rule can set a parameter of the model, and let it vary.

    `Hybridization.validate` has checked that the target of `RHS` is a
    parameter of the model which no rule sets and no event changes. An
    initial assignment is what the validation leaves to the compiler: the
    model would give the parameter two values at the start.

    Args:
        model: the model.
        network: id of the network.
        key: id of the output which sets the target.
        target: id of the parameter.

    Raises:
        NetworkCompilationError: if the target is not a parameter of the
            model, or if an initial assignment sets it.
    """
    parameter: libsbml.Parameter | None = model.model.getParameter(target)
    prefix = f"Network '{network}': the target '{target}' of '{key}'"
    if parameter is None or target in model.created:
        raise NetworkCompilationError(
            f"{prefix} is not a parameter of the model '{model.name}'"
        )
    if model.model.getInitialAssignmentBySymbol(target) is not None:
        raise NetworkCompilationError(
            f"{prefix} has an initial assignment in the model '{model.name}', "
            f"which the assignment rule of the network replaces"
        )
    parameter.setConstant(False)


def _symbols(ids: np.ndarray) -> np.ndarray:
    """Get the array of the symbols of an array of ids."""
    return np.asarray(np.frompyfunc(sympy.Symbol, 1, 1)(ids), dtype=object)


def _input_symbols(model: _Model, hybridization: Hybridization) -> list[np.ndarray]:
    """Add the inputs of a network to the model.

    Args:
        model: the model.
        hybridization: the hybridization of the network.

    Returns:
        The symbols of the elements of the inputs, one array per input.

    Raises:
        NetworkCompilationError: if a formula uses a symbol which is not a
            value of the model, see `_Model.is_value`, or if an id is taken.
    """
    sid = hybridization.network.sid
    shapes = hybridization.input_shapes()
    ids = [np.empty(shape, dtype=object) for shape in shapes]
    for k, shape in enumerate(shapes):
        for index in np.ndindex(shape):
            ids[k][index] = input_id(sid, k, index)

    for key, network_input in hybridization.inputs.items():
        k, index = parse_io_id(sid, "input", key)
        if network_input.arrays is not None:
            array = _compiled_array(network_input)
            for element in np.ndindex(array.shape):
                model.add_parameter(
                    sid,
                    ids[k][element if index is None else index],
                    f"the input '{key}'",
                    value=float(array[element]),
                )
            continue
        formula = network_input.formula
        if formula is None or index is None:
            # excluded by the validation of the hybridization
            raise NetworkCompilationError(
                f"Network '{sid}': the input '{key}' has no formula"
            )
        unknown = sorted(
            symbol
            for symbol in formula_symbols(formula)
            if symbol != TIME and not model.is_value(symbol)
        )
        if unknown:
            raise NetworkCompilationError(
                f"Network '{sid}', input '{key}': the formula '{formula}' uses "
                f"{unknown}, which are neither parameters, species, compartments, "
                f"reactions or species references of the model '{model.name}' "
                f"nor constants of the hybridization"
            )
        math: libsbml.ASTNode | None = libsbml.parseL3FormulaWithModel(
            formula, model.model
        )
        if math is None:
            raise NetworkCompilationError(
                f"Network '{sid}', input '{key}': the formula '{formula}' is not "
                f"valid math: {libsbml.getLastParseL3Error()}"
            )
        model.add_parameter(sid, ids[k][index], f"the input '{key}'")
        model.add_rule(sid, ids[k][index], f"input '{key}'", math)
    return [_symbols(array) for array in ids]


def _compiled_array(network_input: NetworkInput) -> np.ndarray:
    """Get the values of an input of arrays the model is written with.

    Args:
        network_input: an input of arrays.

    Returns:
        The array of all conditions, and the array of the first condition for
        an input without one: a fit sets the values of a condition before
        every simulation, see `Hybridization.derived_changes`.
    """
    arrays = network_input.arrays or {}
    condition = ALL_CONDITIONS if ALL_CONDITIONS in arrays else sorted(arrays)[0]
    return np.asarray(arrays[condition], dtype=float)


def _element_symbols(model: _Model, network: Network) -> dict[str, dict[str, Any]]:
    """Add the elements of the arrays of a network to the model.

    Args:
        model: the model.
        network: the network.

    Returns:
        The arrays of the symbols of the elements, layer id -> array name ->
        symbols. The arrays which are not parameters, i.e. the running
        statistics of a normalization, are the numbers.
    """
    symbols: dict[str, dict[str, Any]] = {
        layer: {name: np.array(array, dtype=object) for name, array in arrays.items()}
        for layer, arrays in network.parameters.items()
    }
    for sid, (layer, name, index) in network.parameter_ids().items():
        if name not in network.parameters.get(layer, {}):
            continue
        model.add_parameter(
            network.sid,
            sid,
            f"the element '{sid}'",
            value=float(network.parameters[layer][name][index]),
        )
        symbols[layer][name][index] = sympy.Symbol(sid)
    return symbols


def _compile(model: _Model, hybridization: Hybridization) -> None:
    """Write one network into the model.

    Args:
        model: the model.
        hybridization: the hybridization of the network, with all its outputs.

    Raises:
        NetworkCompilationError: if the network cannot be compiled.
        UnsupportedLayerError: if a layer or function is not evaluated on
            expressions. `Hybridization` refuses such a network for the
            patterns which are compiled, this is the fallback of a direct
            call.
    """
    network = hybridization.network
    sid = network.sid
    for key, value in hybridization.constants.items():
        model.add_constant(sid, key, value)
    parameters = _element_symbols(model, network)
    inputs = _input_symbols(model, hybridization)

    def on_node(node: Node, value: np.ndarray) -> np.ndarray:
        """Replace the expressions of a node by the symbols of its units."""
        ids = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            ids[index] = unit_id(sid, node.name, index)
            what = f"the unit {index} of the node '{node.name}'"
            model.add_parameter(sid, ids[index], what)
            model.add_rule(
                sid, ids[index], f"node '{node.name}'", sympy.sympify(value[index])
            )
        return _symbols(ids)

    try:
        outputs = evaluate(network.model, parameters, inputs, SympyBackend(), on_node)
    except ValueError as err:
        if isinstance(err, NetworkCompilationError):
            raise
        raise NetworkCompilationError(str(err)) from err

    for k, output in enumerate(outputs):
        for index in np.ndindex(output.shape):
            key = output_id(sid, k, index)
            model.add_parameter(sid, key, f"the output '{key}'")
            model.add_rule(sid, key, f"output '{key}'", sympy.sympify(output[index]))


def _compile_targets(
    model: _Model, hybridization: Hybridization, patterns: dict[str, NetworkPattern]
) -> None:
    """Write the rules of the targets of the outputs of a network.

    Args:
        model: the model.
        hybridization: the hybridization of the network, with all its outputs.
        patterns: id of the output -> the pattern of its hybridization.

    Raises:
        NetworkCompilationError: if a target cannot be set by a rule, or if
            an output of another network sets it.
    """
    sid = hybridization.network.sid
    for key, target in hybridization.outputs.items():
        if target in model.targets:
            network, output = model.targets[target]
            raise NetworkCompilationError(
                f"Network '{sid}': the target '{target}' of '{key}' is set by "
                f"the output '{output}' of the network '{network}', a target "
                f"has one rule"
            )
        model.targets[target] = (sid, key)
        if patterns[key] is NetworkPattern.OBSERVABLE:
            model.add_parameter(
                sid, target, f"the target '{target}' of the output '{key}'"
            )
        else:
            _check_target(model, sid, key, target)
        model.add_rule(sid, target, f"target '{target}'", sympy.Symbol(key))


def compile_network(
    sbml_path: Path, hybridizations: Sequence[Hybridization], output_path: Path
) -> Path:
    """Write a model with the networks of its right hand side and observables.

    All networks of the patterns `RHS` and `OBSERVABLE` are added in one pass
    and one model is written. The hybridizations of the pattern
    `PRE_INITIALIZATION` are not a part of the model and are left out.

    Args:
        sbml_path: the SBML model without the networks.
        hybridizations: the hybridizations of the model.
        output_path: the path the model with the networks is written to, see
            `compiled_path`.

    Returns:
        The path of the model which was written.

    Raises:
        NetworkCompilationError: if no hybridization is compiled, if the
            hybridizations name different models, if two hybridizations of
            a network differ, if two networks give a constant different
            values, if two networks set one target, if an id of a network is
            an id of the model or of another part of a network, if a formula of an input uses a symbol the
            model does not have, if an initial assignment sets a target, if
            an expression has no MathML of SBML, or if the model with the
            networks is not valid SBML. The message names the network.
        NetworkHybridizationError: if the model does not exist or a
            hybridization does not fit it, e.g. a target of `RHS` which is
            not a parameter or which a rule or an event sets, see
            `Hybridization.validate`. `Hybridization` also refuses a network
            with a layer or function which is not evaluated on expressions
            for the patterns which are compiled.
    """
    compiled = [h for h in hybridizations if h.pattern.is_compiled]
    if not compiled:
        raise NetworkCompilationError(
            f"No network is compiled into '{sbml_path}': the patterns of the "
            f"hybridizations are {[str(h.pattern) for h in hybridizations]}"
        )
    models = sorted({h.model for h in compiled})
    if len(models) > 1:
        raise NetworkCompilationError(
            f"The networks {[h.network.sid for h in compiled]} are compiled "
            f"into one model, but name the models {models}"
        )
    if Path(output_path).resolve() == Path(sbml_path).resolve():
        raise NetworkCompilationError(
            f"The model with the networks replaces the model '{sbml_path}', "
            f"write it to another path"
        )
    for hybridization in compiled:
        hybridization.validate(Path(sbml_path))
    patterns = {key: h.pattern for h in compiled for key in h.outputs}

    model = _Model(Path(sbml_path), compiled[0].network.sid)
    merged = _merge(compiled)
    for hybridization in merged:
        _compile(model, hybridization)
    for hybridization in merged:
        _compile_targets(model, hybridization, patterns)
    path = model.write(output_path)
    logger.info(
        "The networks %s are compiled into '%s': %d parameters",
        [h.network.sid for h in merged],
        path.name,
        len(model.created),
    )
    return path
