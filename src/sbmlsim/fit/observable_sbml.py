"""The observables of a fit as entities of the model the engine simulates.

A fit mapping which observes a formula of the selections of a simulation,
e.g. `[A] * scale + offset` of a PEtab observable, is no column of a
simulation: the engine reads a curve off a column of the table of its
condition, so such a mapping kept the fit of its problem in python
(`sbmlsim.fit.fit_core`, the gap `observable`).

This module writes those observables into the model instead. An observable
becomes a global parameter of the SBML with an assignment rule of its
formula, so the compiler of the engine reads it as it reads the math of the
model itself, the engine selects it as an ordinary column, and its forward
sensitivities are the ones of a column of the model. Nothing of the math is
reimplemented and nothing is evaluated twice: the formula is written as the
MathML of SBML by `sbmlsim.simulator.formula.formula_mathml`, which writes
the parse tree `compile_formula` compiles, so the model computes what the
observable of the problem computes.

The model of the user is not changed: the SBML of the model is read, the
observables are added to the copy, and the copy is loaded as a model of its
own for the fit (`observable_model`), keyed on its content as every model of
the engine is, so one problem compiles it once.

The formula of an observable is the math of PEtab over the selections of
roadrunner, which are not the identifiers of SBML: the concentration `[S]`
is no SId, and a species in the math of SBML is its concentration where
roadrunner's `S` is its amount. Nothing of a formula is substituted as text
but the brackets of a concentration, which are unambiguous: every other
symbol is carried by the `names` of `formula_mathml`, and a symbol which is
no identifier of the model at all, i.e. an amount or a concentration which
needs the size of a compartment, and a placeholder, which is a value per
measurement, become helper parameters of the model (`HELPER_PREFIX`) whose
value or assignment rule the formula then reads.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import libsbml

from sbmlsim.simulator.formula import compile_formula, formula_mathml

if TYPE_CHECKING:
    from sbmlsim.fit.objects import ObservableModel
    from sbmlsim.model.symbols import ModelSymbols

logger = logging.getLogger(__name__)

#: the prefix of the parameter of an observable in the model of a fit, which
#: no entity of a model is expected to carry
OBSERVABLE_PREFIX: Final = "sbmlsim_observable__"

#: the prefix of a helper parameter of an observable, which carries a value
#: or a formula the observable reads, see the module
HELPER_PREFIX: Final = "sbmlsim_observable_helper__"

#: the unit of a parameter this module adds; the values of a fit carry the
#: units of the problem and the model of a fit is never read for units
HELPER_UNIT: Final = "dimensionless"

#: a concentration of the selections of roadrunner, `[S]`
_CONCENTRATION: Final = re.compile(r"\[\s*([A-Za-z_]\w*)\s*\]")


@dataclass
class Entities:
    """The parameters an observable of a fit adds to the model.

    Attributes:
        constants: the value of every constant helper parameter, a
            placeholder which is a number.
        rules: the formula of every helper parameter which has an assignment
            rule, over the identifiers of SBML.
        observables: the formula of every observable, over the identifiers of
            SBML, in the order the observables were added.
    """

    constants: dict[str, float] = field(default_factory=dict)
    rules: dict[str, str] = field(default_factory=dict)
    observables: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Whether anything is added to the model."""
        return bool(self.constants or self.rules or self.observables)


def _legal(formula: str) -> tuple[str, dict[str, str]]:
    """Rewrite the concentrations of a formula into identifiers of SBML.

    Args:
        formula: the formula, the math of PEtab over the selections.

    Returns:
        The formula whose `[S]` are SIds, and the species of every such SId.
    """
    species: dict[str, str] = {}

    def name(match: re.Match[str]) -> str:
        sid = f"{HELPER_PREFIX}conc_{match.group(1)}"
        species[sid] = match.group(1)
        return sid

    return _CONCENTRATION.sub(name, formula), species


def _names_of(
    formula: str,
    symbols: ModelSymbols,
    entities: Entities,
    at: str,
    given: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Get a formula over identifiers of SBML and the names its symbols map to.

    A species in the math of SBML is its concentration unless it has
    `hasOnlySubstanceUnits`, where it is its amount, while the selection `S`
    of roadrunner is the amount of a species and `[S]` its concentration, so
    the two differ by the size of the compartment of the species, each the
    other way round:

    | selection | `hasOnlySubstanceUnits` | the math of SBML |
    |---|---|---|
    | `[S]` | false | `S` |
    | `[S]` | true | `S / c` |
    | `S` | false | `S * c` |
    | `S` | true | `S` |

    The two rows which are no identifier become a helper parameter of
    `entities` with that formula as its assignment rule. A parameter, a
    compartment and anything else is the identifier itself.

    Args:
        formula: the formula, the math of PEtab over the selections.
        symbols: the symbols of the model.
        entities: the entities of the model, which helpers are added to.
        at: what a problem names, the observable the formula belongs to.
        given: the identifier of the model of a symbol the caller names, e.g.
            the parameter of an observable of the same simulation group which
            the formula reads; it wins over every rule below.

    Returns:
        The formula whose symbols are SIds, and the identifier of the model of
        every symbol which is not itself.

    Raises:
        ValueError: if the formula is not valid math of PEtab, or if it reads
            the concentration of a species the model does not have.
    """
    rewritten, concentrations = _legal(formula)
    compiled = compile_formula(rewritten)
    names: dict[str, str] = dict(given or {})
    for symbol in compiled.symbols:
        if symbol in names:
            # the caller knows the identifier of this symbol
            continue
        species = concentrations.get(symbol)
        if species is not None:
            if species not in symbols.species:
                raise ValueError(
                    f"The observable '{at}' reads the concentration '[{species}]', "
                    f"which is no species of the model."
                )
            if species not in symbols.only_substance:
                names[symbol] = species
                continue
            compartment = symbols.species_compartment[species]
            entities.rules[symbol] = _written(f"{species} / {compartment}", {})
            continue
        if symbol in symbols.species and symbol not in symbols.only_substance:
            # the amount of a species whose math of SBML is its concentration
            helper = f"{HELPER_PREFIX}amount_{symbol}"
            compartment = symbols.species_compartment[symbol]
            entities.rules[helper] = _written(f"{symbol} * {compartment}", {})
            names[symbol] = helper
    return rewritten, names


def add_observable(
    name: str,
    observable: ObservableModel,
    values: dict[str, float | str],
    symbols: ModelSymbols,
    entities: Entities,
    names: Mapping[str, str] | None = None,
) -> None:
    """Add an observable of a fit to the entities of the model.

    The placeholders of the observable become helper parameters, a number a
    constant one and a formula one with that formula as its assignment rule,
    so the formula of the observable reads them as identifiers and no value
    is substituted as text.

    Args:
        name: the id of the parameter of the observable.
        observable: the observable of the fit mapping.
        values: the value of every placeholder, a number or a formula of the
            selections.
        symbols: the symbols of the model.
        entities: the entities of the model, which the observable is added
            to.
        names: the identifier of the model of a symbol the caller names, e.g.
            the helper parameter of the column of the simulation which the
            observable of a noise formula stands for
            (`sbmlsim.fit.fit_core._noise_entities`), see `_names_of`.

    Raises:
        ValueError: if a placeholder has no value, if the formula or the
            formula of a value is not valid math, or if the formula reads the
            concentration of a species the model does not have.
    """
    rewritten, resolved = _names_of(observable.formula, symbols, entities, name, names)
    for placeholder in observable.placeholders:
        if placeholder not in values:
            raise ValueError(
                f"The observable '{name}' has the placeholder '{placeholder}' "
                f"without a value."
            )
        value = values[placeholder]
        helper = f"{HELPER_PREFIX}{name[len(OBSERVABLE_PREFIX) :]}_{placeholder}"
        if isinstance(value, str):
            inner, inner_names = _names_of(value, symbols, entities, name)
            entities.rules[helper] = _written(inner, inner_names)
        else:
            entities.constants[helper] = float(value)
        resolved[placeholder] = helper
    entities.observables[name] = _written(rewritten, resolved)


def add_formula(
    name: str,
    formula: str,
    symbols: ModelSymbols,
    entities: Entities,
    names: Mapping[str, str] | None = None,
) -> None:
    """Add an observable of a formula of the selections to the entities of a model.

    `add_observable` is the same for an `ObservableModel`, whose placeholders
    become helper parameters; this is the plain case, a formula of the
    selections and of parameters the model already has, which the logarithmic
    column of a mapping of log-normal noise is
    (`sbmlsim.fit.fit_core._log_observable`).

    Args:
        name: the id of the parameter of the observable.
        formula: the formula, the math of PEtab over the selections.
        symbols: the symbols of the model.
        entities: the entities of the model, which the observable is added
            to.
        names: the identifier of the model of a symbol the caller names, e.g.
            the parameter of an observable which the formula reads, see
            `_names_of`.

    Raises:
        ValueError: if the formula is not valid math, or if it reads the
            concentration of a species the model does not have.
    """
    rewritten, resolved = _names_of(formula, symbols, entities, name, names)
    entities.observables[name] = _written(rewritten, resolved)


def _written(formula: str, names: dict[str, str]) -> str:
    """Get the MathML of a formula, checked, with the names of its symbols.

    Args:
        formula: the formula whose symbols are SIds.
        names: the identifier of the model of a symbol of the formula.

    Returns:
        The MathML of SBML.

    Raises:
        ValueError: if the formula is not valid math of PEtab.
    """
    return formula_mathml(formula, names)


def with_observables(sbml: str, entities: Entities) -> str:
    """Get the SBML of a model with the observables of a fit.

    Every observable and every helper becomes a global parameter, the
    observables and the helpers of a formula with an assignment rule of their
    MathML. The document is read and written by libsbml alone, so nothing
    else of the model is touched.

    Args:
        sbml: the SBML of the model.
        entities: the entities to add (`add_observable`).

    Returns:
        The SBML with the observables, the given one when there are none.

    Raises:
        ValueError: if the SBML cannot be read, if an id is an entity of the
            model, or if a formula is no MathML libsbml reads.
    """
    if not entities:
        return sbml
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    model: libsbml.Model | None = doc.getModel()
    if model is None:
        raise ValueError(
            "The SBML of the model of a fit cannot be read, so the observables "
            "of the fit cannot be written into it."
        )
    units = doc.getLevel() > 2
    for name, value in entities.constants.items():
        _parameter(model, name, units, constant=True, value=value)
    for name, math in {**entities.rules, **entities.observables}.items():
        _parameter(model, name, units, constant=False, value=0.0)
        _rule(model, name, math)
    written: str = libsbml.writeSBMLToString(doc)
    return written


def _parameter(
    model: libsbml.Model, name: str, units: bool, *, constant: bool, value: float
) -> None:
    """Add a global parameter of an observable to a model.

    Args:
        model: the model of the document.
        name: the id of the parameter.
        units: whether the level of the document has units of a parameter.
        constant: whether the parameter is constant.
        value: the value of the parameter.

    Raises:
        ValueError: if the id is an entity of the model, or if libsbml
            refuses a setter.
    """
    if model.getElementBySId(name) is not None:
        raise ValueError(
            f"The observable '{name}' of the fit is an entity of the model, so "
            f"it cannot be added to it as a parameter."
        )
    parameter: libsbml.Parameter = model.createParameter()
    _checked(parameter.setId(name), name)
    _checked(parameter.setConstant(constant), name)
    _checked(parameter.setValue(float(value)), name)
    if units:
        _checked(parameter.setUnits(HELPER_UNIT), name)


def _rule(model: libsbml.Model, name: str, math: str) -> None:
    """Add the assignment rule of a parameter of an observable to a model.

    Args:
        model: the model of the document.
        name: the id of the parameter, the variable of the rule.
        math: the MathML of the rule.

    Raises:
        ValueError: if the MathML cannot be read, or if libsbml refuses a
            setter.
    """
    node = libsbml.readMathMLFromString(math)
    if node is None:
        raise ValueError(
            f"The formula of the observable '{name}' of the fit is no MathML "
            f"libsbml reads."
        )
    rule: libsbml.AssignmentRule = model.createAssignmentRule()
    _checked(rule.setVariable(name), name)
    _checked(rule.setMath(node), name)


def _checked(code: int, what: str) -> None:
    """Check the return code of a setter of libsbml.

    Args:
        code: the return code.
        what: the observable the setter sets, which a problem names.

    Raises:
        ValueError: for a code which is no success.
    """
    if code != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise ValueError(
            f"The observable '{what}' of the fit could not be written into the "
            f"model: libsbml answered "
            f"'{libsbml.OperationReturnValue_toString(code)}'."
        )


def observable_id(mapping: str) -> str:
    """Get the id of the parameter of the observable of a fit mapping.

    Args:
        mapping: the key of the fit mapping.

    Returns:
        The id, `OBSERVABLE_PREFIX` and the key with every character which is
        no letter, digit or underscore replaced by an underscore, so that it
        is an SId.
    """
    safe = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in mapping
    )
    return f"{OBSERVABLE_PREFIX}{safe}"


def observable_model(model: Any, entities: Entities) -> Any:
    """Get the model of a fit, with the observables of its mappings.

    The model is loaded again from the description of the given one
    (`sbmlsim.model.model_sbml.description`) with the observables written
    into its SBML, so it keeps the added parameters, the changes, the
    selections, the id, the name and the base path of the model of the
    problem. The added parameters matter: a parameter of the fit which PEtab
    declares and the model does not have is one of them, so an observable
    which reads it only compiles in a model which carries them.

    Args:
        model: the model of the problem, a `CoreSBMLModel`.
        entities: the entities of its observables (`add_observable`).

    Returns:
        The model itself when there are no observables, else a model of the
        same engine with them.
    """
    if not entities:
        return model
    from sbmlsim.model.model import AbstractModel
    from sbmlsim.model.model_resources import Source
    from sbmlsim.model.model_sbml import description, load_model

    given = description(model)
    derived = AbstractModel(
        source=Source(
            source=given.source.source,
            path=given.source.path,
            content=with_observables(model.sbml, entities),
        ),
        sid=given.sid,
        name=given.name,
        language_type=given.language_type,
        base_path=given.base_path,
        changes=dict(given.changes),
        selections=list(given.selections) if given.selections is not None else None,
        parameters=given.parameters,
    )
    loaded = load_model(derived, model.engine, ureg=model.uinfo.ureg, reload=True)
    # the model of the fit integrates as the model of the problem does: a
    # fresh load has the settings of the engine and not the tolerances of the
    # `FitSettings` the problem was initialized with, which made such a fit
    # integrate at the defaults of the engine (#613)
    loaded.set_integrator_state(model.integrator_state())
    logger.debug(
        "the model of the fit carries the observables %s and the helpers %s",
        sorted(entities.observables),
        sorted({**entities.constants, **entities.rules}),
    )
    return loaded
