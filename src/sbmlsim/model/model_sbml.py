"""The SBML model independent of the engine which simulates it.

`SBMLModel` reads the SBML of a model once, flattens a hierarchical model,
adds the parameters of its description and derives its symbols and units,
which the compile step and the results need whatever engine simulates the
model. An engine subclasses it with the model it loads, e.g.
`sbmlsim.model.model_roadrunner.RoadrunnerSBMLModel`, and `load_model` is the
one place a model of an engine is created.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypedDict

import libsbml
from pint.facets.plain import PlainUnit

from sbmlsim.engine import Engine, resolve_engine
from sbmlsim.model.event_rows import MARKER_PREFIX
from sbmlsim.model.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.symbols import ModelSymbols
from sbmlsim.model.tolerances import AbsoluteTolerance
from sbmlsim.units import Quantity, UnitRegistry, UnitsInformation
from sbmlsim.units import ureg as package_ureg

if TYPE_CHECKING:
    from sbmlsim.model.model_roadrunner import JitOptimization
    from sbmlsim.simulator.plan import Assignment

logger = logging.getLogger(__name__)

#: a setting of the integrator, see `SBMLModel.set_integrator_settings`
IntegratorSetting = float | int | bool | AbsoluteTolerance


class IntegratorSettings(TypedDict, total=False):
    """The settings of the integrator a simulator takes, by their names in roadrunner.

    The settings of CVODE, the integrator of roadrunner; COPASI takes
    `relative_tolerance`, `absolute_tolerance`, `maximum_num_steps` and
    `maximum_time_step` and ignores the others, see
    `sbmlsim.model.model_copasi.CopasiSBMLModel.set_integrator_settings`.
    `sbmlsim.simulator.Simulator(**settings)` checks the names and types of
    a dictionary of this type; a dictionary of settings of other names or
    types, `dict[str, IntegratorSetting]`, e.g. of another integrator of
    roadrunner, is set with `Simulator.set_integrator_settings` or
    `SBMLModel.set_integrator_settings`, which take every name.
    """

    absolute_tolerance: float | AbsoluteTolerance
    relative_tolerance: float
    stiff: bool
    variable_step_size: bool
    initial_time_step: float
    minimum_time_step: float
    maximum_time_step: float
    maximum_num_steps: int
    maximum_bdf_order: int
    maximum_adams_order: int
    multiple_steps: bool
    max_output_rows: int


#: suffix of the parameter whose assignment rule is the math of an initial
#: assignment, see `RoadrunnerSBMLModel.initialize`
INITIAL_SUFFIX = "__initial"


class SBMLModel(AbstractModel, ABC):
    """An SBML model, loaded by an engine, see the module.

    Attributes:
        engine: the engine which simulates the model.
        sbml: the SBML of the model as sbmlsim hands it to the engine,
            flattened, before the added parameters: the units of the model
            are read from it and a worker loads it, so it derives the symbols
            and the units the parent has, see
            `sbmlsim.simulator.worker.ModelSpec`.
        hierarchical: whether the model uses the package `comp` of
            hierarchical models, i.e. `sbml` is its flattened SBML.
        symbols: the symbols of the model with the added parameters.
        uinfo: the units of the model.
        reported_triggers: the events whose trigger was reported as no formula
            the simulator evaluates, once per event of this instance (a worker
            of a pool loads its own instance and reports it once as well), see
            `sbmlsim.simulator.changes`.
        warned_start: whether the start of a simulation was reported, see
            `sbmlsim.simulator.executor`.
        selections_given: whether the selections were given, `False` for
            the default selections of the engine, see `default_selections`;
            a model loaded in another engine gets that engine's defaults, see
            `load_model`.
    """

    engine: ClassVar[Engine]

    def __init__(
        self,
        source: str | Path | Source,
        base_path: Path | None = None,
        changes: dict | None = None,
        sid: str | None = None,
        name: str | None = None,
        selections: list[str] | None = None,
        ureg: UnitRegistry | None = None,
        parameters: dict[str, float] | None = None,
    ):
        """Read the SBML of the model, its symbols and its units.

        The `parameters` are added to the model as constant parameters, see
        `AbstractModel`. The changes are normalized to the units of the model.
        The subclass loads the model in its engine from the SBML which
        `_take_engine_sbml` hands over, and sets the selections after it.
        """
        super().__init__(
            source=source,
            language_type=AbstractModel.LanguageType.SBML,
            changes=changes,
            sid=sid,
            name=name,
            base_path=base_path,
            selections=selections,
            parameters=parameters,
        )

        # check SBML
        if self.language_type != AbstractModel.LanguageType.SBML:
            raise ValueError(f"language_type not supported '{self.language_type}'.")

        # the SBML of the model and its symbols, read once
        sbml: str = (
            self.source.content
            if self.source.content is not None
            else Path(str(self.source.path)).read_text(encoding="utf-8")
        )
        # roadrunner looks for the package comp in the text of the <sbml> tag
        # and misses it in a file with the line endings of Windows and in the
        # SBML as a string, it then simulates a hierarchical model unflattened;
        # the model is flattened here, the symbols are the ones it simulates
        self.hierarchical: bool = _is_hierarchical(sbml)
        if self.hierarchical:
            sbml = _flatten(sbml, path=self.source.path, base_path=self.base_path)
        self.sbml: str = sbml
        if self.parameters:
            sbml = _with_parameters(sbml, self.parameters)
        self.symbols: ModelSymbols = ModelSymbols.from_sbml(sbml)
        #: the SBML the engine loads, `sbml` with the added parameters, until
        #: the subclass takes it, see `_take_engine_sbml`
        self._engine_sbml: str | None = sbml

        self.warned_start: bool = False
        self.reported_triggers: set[str] = set()
        self.selections_given: bool = selections is not None

        # normalize model changes
        self.uinfo: UnitsInformation = self.parse_units(
            ureg if ureg is not None else package_ureg
        )
        self.normalize(uinfo=self.uinfo)

    def _take_engine_sbml(self) -> str:
        """Hand the SBML the engine loads over to the subclass, once.

        The SBML is `sbml` with the added parameters; it is not kept after
        the model was loaded.

        Raises:
            RuntimeError: if the SBML was taken already.
        """
        sbml = self._engine_sbml
        if sbml is None:
            raise RuntimeError("The SBML of the engine was taken already.")
        self._engine_sbml = None
        return sbml

    @property
    def label(self) -> str:
        """Get the name of the model in a message.

        The id of the model, else the id of its SBML model, else the name of
        its SBML model, else `model`.
        """
        return self.sid or self._sbml_label

    @cached_property
    def _sbml_label(self) -> str:
        """Get the id or else the name of the SBML model, read once."""
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(self.sbml)
        model: libsbml.Model | None = doc.getModel()
        if model is None:
            return "model"
        return str(model.getId() or model.getName() or "model")

    def Q(self, value: Any, unit: str | PlainUnit | None = None) -> Quantity:
        """Create a quantity in a unit of pint and of the unit ids of the model.

        A unit id of the model is its expression, e.g. `mg_per_min/kg` is a
        milligram per minute and kilogram for a model whose unit definition
        `mg_per_min` is a milligram per minute, see
        `UnitsInformation.quantity`. `sbmlsim.Q` takes the units of pint only.
        A unit of pint, e.g. the `units` of a quantity, is taken as it is.

        Args:
            value: the magnitude, a number or an array.
            unit: the unit, a string or a unit of pint, dimensionless without
                one.

        Returns:
            The quantity.

        Raises:
            UndefinedUnitError: for a unit which is neither a unit of pint nor
                a unit id of the model.
        """
        return self.uinfo.quantity(value, unit)

    def parse_units(self, ureg: UnitRegistry) -> UnitsInformation:
        """Parse the units of the model from its SBML, see `sbml`.

        A hierarchical model has the units of its flattened SBML, which include
        the ones of its submodels.
        """
        return UnitsInformation.from_sbml(sbml=self.sbml, ureg=ureg)

    def default_selections(self) -> list[str]:
        """Get the selections of a model loaded without selections.

        Every entity of the model which has a value, the selections roadrunner
        makes by default in its order: the time, the amounts of the floating
        and of the boundary species, the global parameters, the species
        references with an id whose species is a boundary species
        (roadrunner's global parameters, the other ones are entries of the
        stoichiometry matrix), the reactions and the compartments, then the
        concentrations of the floating and of the boundary species. The
        species, the parameters and the compartments are each in the order
        of the SBML with the entities without a rule first and the ones of a
        rate or an assignment rule after them, as roadrunner orders them
        (an initial assignment or an event assignment is no rule). The rule
        of the species references was determined empirically, against
        roadrunner on every case of the SBML Test Suite. A compartment whose
        size is `NaN` and set by no initial assignment or rule, e.g. a
        membrane whose area the model does not use, has no value to record (a
        compartment without a size has the size 1 in roadrunner); the
        parameters added to the model and the helpers of sbmlsim are not
        selected.

        Returns:
            The selections.
        """
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(self.sbml)
        model: libsbml.Model = doc.getModel()
        ruled = {r.getVariable() for r in model.getListOfRules() if not r.isAlgebraic()}

        def ordered(ids: list[str]) -> list[str]:
            return [
                *(i for i in ids if i not in ruled),
                *(i for i in ids if i in ruled),
            ]

        floating: list[str] = []
        boundary: list[str] = []
        for s in model.getListOfSpecies():
            (boundary if s.getBoundaryCondition() else floating).append(s.getId())
        floating, boundary = ordered(floating), ordered(boundary)
        sized = {a.getSymbol() for a in model.getListOfInitialAssignments()} | ruled
        compartments = ordered(
            [
                c.getId()
                for c in model.getListOfCompartments()
                if c.getId() in sized or not (c.isSetSize() and math.isnan(c.getSize()))
            ]
        )
        stoichiometries = [
            sr.getId()
            for reaction in model.getListOfReactions()
            for sr in (*reaction.getListOfReactants(), *reaction.getListOfProducts())
            if sr.isSetId() and sr.getSpecies() in boundary
        ]
        return [
            "time",
            *floating,
            *boundary,
            *ordered(
                [
                    p.getId()
                    for p in model.getListOfParameters()
                    if not _is_helper(p.getId()) and p.getId() not in self.parameters
                ]
            ),
            *stoichiometries,
            *[r.getId() for r in model.getListOfReactions()],
            *compartments,
            *[f"[{sid}]" for sid in (*floating, *boundary)],
        ]

    @abstractmethod
    def initialize(self, assignments: Sequence[Assignment]) -> None:
        """Initialize the model with values before its initialization.

        Args:
            assignments: the pre-initialization assignments of a plan.
        """

    @abstractmethod
    def reset(self) -> None:
        """Set the model to the state it was loaded in, without a simulation."""

    @abstractmethod
    def simulated(self) -> None:
        """Record that the model was simulated since its last initialization."""

    @abstractmethod
    def loaded_value(self, selection: str) -> float:
        """Get the value of a parameter, a compartment or a species of the model as loaded.

        Args:
            selection: the id of a parameter, a compartment or a species (its
                amount), or `[S]`, the concentration of the species `S`.

        Returns:
            The value as loaded.

        Raises:
            KeyError: if the selection is none of these.
        """

    @abstractmethod
    def value(self, selection: str) -> float:
        """Get the current value of a selection.

        Args:
            selection: the selection, e.g. `S`, `[S]`, a parameter or a
                reaction.

        Returns:
            The value in the current state of the model.

        Raises:
            RuntimeError: if the model has no selection of the name, in
                every engine.
        """

    @abstractmethod
    def has_selection(self, selection: str) -> bool:
        """Check whether the model has a selection of a name.

        Args:
            selection: the name, e.g. `S`, `[S]`, a parameter or `time`.

        Returns:
            Whether the loaded model can select it.
        """

    @abstractmethod
    def set_selections(self, selections: Sequence[str] | None) -> None:
        """Set the selections of the simulations of the model.

        Args:
            selections: the selections, `default_selections` for `None`,
                see `selections_given`.
        """

    @abstractmethod
    def set_integrator_settings(self, **settings: IntegratorSetting) -> None:
        """Set settings of the integrator.

        Args:
            **settings: the settings by their names, `absolute_tolerance` a
                float or an `AbsoluteTolerance`.

        Raises:
            ValueError: if the integrator has no setting of a name.
        """

    @abstractmethod
    def integrator_state(self) -> tuple[str, tuple[tuple[str, Any], ...]]:
        """Get the name of the integrator of the model and every one of its settings.

        The absolute tolerance is the `AbsoluteTolerance` of the model and
        not the tolerances of its states.

        Returns:
            The name of the integrator and its settings, sorted by name.
        """

    @abstractmethod
    def set_integrator_state(
        self, state: tuple[str, tuple[tuple[str, Any], ...]]
    ) -> None:
        """Set the integrator of the model and its settings, see `integrator_state`.

        A worker of a pool loads a model with the state of the integrator of
        the model of the parent, see `sbmlsim.simulator.worker.ModelSpec`.

        Args:
            state: the name of the integrator and its settings.

        Raises:
            ValueError: if the engine has no integrator of the name, or the
                integrator has no setting of a name.
        """

    @abstractmethod
    def state_ids(self) -> list[str]:
        """Get the ids of the states which the integrator integrates."""


def description(model: SBMLModel) -> AbstractModel:
    """Get the description of a loaded model, which loads it again in any engine.

    It is what does not depend on the engine of the model: its `sbml` as the
    source, so a file rewritten or deleted after the model was loaded changes
    nothing, with the path of its file, its added parameters, its changes, its
    selections (none for the default selections, so the engine selects its
    own defaults, see `SBMLModel.selections_given`), its id, its name and its
    base path; a state set on the model is not part of it.

    Args:
        model: a loaded model.

    Returns:
        The description of the model.
    """
    return AbstractModel(
        source=Source(
            source=model.source.source, path=model.source.path, content=model.sbml
        ),
        sid=model.sid,
        name=model.name,
        language_type=model.language_type,
        base_path=model.base_path,
        changes=dict(model.changes),
        selections=(
            list(model.selections)
            if model.selections_given and model.selections is not None
            else None
        ),
        parameters=model.parameters,
    )


def load_model(
    model: AbstractModel | str | Path,
    engine: Engine | str,
    *,
    ureg: UnitRegistry | None = None,
    jit_optimization: JitOptimization | str | None = None,
    reload: bool = False,
) -> SBMLModel:
    """Load a model in an engine.

    A model of the engine is the model itself, as it is: `ureg` and
    `jit_optimization` do not apply to it, unless `reload` loads it again. A
    model of another engine is loaded again from its `description`; a state
    set on the model of the other engine is not carried over.

    Args:
        model: a loaded model, the description of a model or its path or SBML.
        engine: the engine or its name.
        ureg: the unit registry of the units of the model, the registry of
            the package without one; a loaded model which is loaded again
            keeps its own without one.
        jit_optimization: the optimization level of the compilation of a
            model of roadrunner, see
            `sbmlsim.model.model_roadrunner.resolve_jit_optimization`.
        reload: load a model of the engine again as well, e.g. with another
            unit registry or optimization level.

    Returns:
        The model of the engine.

    Raises:
        ValueError: for a name which is no engine.
        ImportError: for the engine COPASI without python-copasi, AMICI without
            AMICI.
    """
    engine = resolve_engine(engine)
    if isinstance(model, SBMLModel):
        if model.engine is engine and not reload:
            return model
        if ureg is None:
            ureg = model.uinfo.ureg
        model = description(model)
    if engine is Engine.ROADRUNNER:
        # the module of roadrunner imports this module
        from sbmlsim.model.model_roadrunner import RoadrunnerSBMLModel

        if isinstance(model, AbstractModel):
            return RoadrunnerSBMLModel.from_abstract_model(
                abstract_model=model, ureg=ureg, jit_optimization=jit_optimization
            )
        return RoadrunnerSBMLModel(
            source=model, ureg=ureg, jit_optimization=jit_optimization
        )
    if engine is Engine.COPASI:
        # COPASI is an optional extra, imported when a model is loaded in it
        from sbmlsim.model.model_copasi import CopasiSBMLModel

        if isinstance(model, AbstractModel):
            return CopasiSBMLModel.from_abstract_model(abstract_model=model, ureg=ureg)
        return CopasiSBMLModel(source=model, ureg=ureg)
    if engine is Engine.AMICI:
        # AMICI is an optional extra, imported when a model is loaded in it
        from sbmlsim.model.model_amici import AmiciSBMLModel

        if isinstance(model, AbstractModel):
            return AmiciSBMLModel.from_abstract_model(abstract_model=model, ureg=ureg)
        return AmiciSBMLModel(source=model, ureg=ureg)
    raise ValueError(f"The engine '{engine}' loads no models.")


def _is_helper(sid: str) -> bool:
    """Whether a parameter is a helper of sbmlsim and no parameter of the model.

    The helpers are the parameters of the initial assignments, see
    `sbmlsim.model.model_roadrunner`, and the markers of the events, see
    `sbmlsim.model.event_rows`.
    """
    return sid.endswith(INITIAL_SUFFIX) or sid.startswith(MARKER_PREFIX)


def _is_hierarchical(sbml: str) -> bool:
    """Check whether a model uses the package `comp` of hierarchical models."""
    if "comp" not in sbml:
        return False
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    return bool(doc.isPackageEnabled("comp"))


def _flatten(sbml: str, path: Path | None, base_path: Path | None = None) -> str:
    """Flatten a hierarchical model.

    The model is read from its file if it has one, so that libsbml resolves
    its external model definitions relative to the file. Inline SBML resolves
    external models relative to its base directory.

    Args:
        sbml: the SBML of the model.
        path: the file of the model, `None` for the SBML as a string.
        base_path: the resource directory of inline SBML, the current working
            directory by default.

    Returns:
        The SBML of the flattened model.

    Raises:
        ValueError: if the model cannot be flattened.
    """
    doc: libsbml.SBMLDocument = (
        libsbml.readSBMLFromFile(str(path))
        if path is not None
        else libsbml.readSBMLFromString(sbml)
    )
    if path is None:
        # libsbml accepts native absolute paths as location URIs, as used by
        # readSBMLFromFile. It does not decode escaped spaces in file:// URIs.
        location = (base_path or Path.cwd()).resolve() / "__sbmlsim_inline__.xml"
        doc.setLocationURI(str(location))
    properties = libsbml.ConversionProperties()
    properties.addOption("flatten comp", True)
    properties.addOption("performValidation", False)
    if doc.convert(properties) != libsbml.LIBSBML_OPERATION_SUCCESS:
        errors = [
            doc.getError(k).getMessage().strip()
            for k in range(doc.getNumErrors())
            if doc.getError(k).getSeverity() >= libsbml.LIBSBML_SEV_ERROR
        ]
        raise ValueError(
            f"The hierarchical model '{path or sbml[:200]}' cannot be flattened: "
            + "; ".join(errors)
        )
    return libsbml.writeSBMLToString(doc)


def _with_parameters(sbml: str, parameters: Mapping[str, float]) -> str:
    """Add constant parameters to a model.

    Args:
        sbml: the SBML of the model.
        parameters: the parameters by their id and value.

    Returns:
        The SBML with the parameters.

    Raises:
        ValueError: if the model already has an entity of the id of a
            parameter.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    model: libsbml.Model = doc.getModel()
    for pid, value in parameters.items():
        if model.getElementBySId(pid) is not None:
            raise ValueError(
                f"The parameter '{pid}' which is added to the model is an entity "
                f"of the model already."
            )
        parameter: libsbml.Parameter = model.createParameter()
        parameter.setId(pid)
        parameter.setValue(float(value))
        parameter.setConstant(True)
    return libsbml.writeSBMLToString(doc)


def sbml_maths(model: libsbml.Model) -> Iterator[tuple[str, libsbml.ASTNode]]:
    """Get every math of a model with the name of its element, e.g. for a message."""
    for function in model.getListOfFunctionDefinitions():
        yield f"function definition '{function.getId()}'", function.getMath()
    for assignment in model.getListOfInitialAssignments():
        yield f"initial assignment of '{assignment.getSymbol()}'", assignment.getMath()
    for rule in model.getListOfRules():
        kind = "rate rule" if rule.isRate() else "assignment rule"
        if rule.isAlgebraic():
            kind = "algebraic rule"
        yield f"{kind} of '{rule.getVariable()}'", rule.getMath()
    for constraint in model.getListOfConstraints():
        yield "constraint", constraint.getMath()
    for reaction in model.getListOfReactions():
        if reaction.isSetKineticLaw():
            law = reaction.getKineticLaw()
            yield f"kinetic law of the reaction '{reaction.getId()}'", law.getMath()
    for event in model.getListOfEvents():
        where = f"event '{event.getId()}'"
        if event.isSetTrigger():
            yield where, event.getTrigger().getMath()
        if event.isSetDelay():
            yield where, event.getDelay().getMath()
        if event.isSetPriority():
            yield where, event.getPriority().getMath()
        for assignment in event.getListOfEventAssignments():
            yield where, assignment.getMath()


#: the definition URL of the delay csymbol
DELAY_URL = "http://www.sbml.org/sbml/symbols/delay"

#: the definition URL of the csymbol rateOf
RATE_OF_URL = "http://www.sbml.org/sbml/symbols/rateOf"


def reads_csymbol(math_: libsbml.ASTNode | None, url: str, kind: int) -> bool:
    """Whether a math reads a csymbol, found by its type or its definition URL.

    libsbml gives some csymbols no definition URL, e.g. rateOf, so a node is
    the csymbol if it has its type or its URL.

    Args:
        math_: the math, `None` for none.
        url: the definition URL of the csymbol.
        kind: the type of libsbml of the csymbol, e.g. `AST_FUNCTION_DELAY`.

    Returns:
        Whether a node of the math is the csymbol.
    """
    stack = [math_] if math_ is not None else []
    while stack:
        node = stack.pop()
        if node.getType() == kind or node.getDefinitionURLString() == url:
            return True
        stack.extend(node.getChild(k) for k in range(node.getNumChildren()))
    return False


def reads_delay(math_: libsbml.ASTNode | None) -> bool:
    """Whether a math reads the delay csymbol, see `reads_csymbol`."""
    return reads_csymbol(math_, DELAY_URL, libsbml.AST_FUNCTION_DELAY)


def reads_rate_of(math_: libsbml.ASTNode | None) -> bool:
    """Whether a math reads the csymbol rateOf, see `reads_csymbol`."""
    return reads_csymbol(math_, RATE_OF_URL, libsbml.AST_FUNCTION_RATE_OF)


def conversion_factors(model: libsbml.Model) -> set[str]:
    """Get the conversion factors of a model and of its species, constants of SBML."""
    factors = {
        s.getConversionFactor()
        for s in model.getListOfSpecies()
        if s.isSetConversionFactor()
    }
    if model.isSetConversionFactor():
        factors.add(model.getConversionFactor())
    return factors


def rate_rule_entities(model: libsbml.Model, symbols: ModelSymbols) -> list[str]:
    """Get the parameters and compartments of rate rules, in the order of the SBML.

    The states an engine integrates after its species, see
    `SBMLModel.state_ids`.
    """
    return [
        e.getId()
        for e in (*model.getListOfParameters(), *model.getListOfCompartments())
        if e.getId() in symbols.rate_rules
    ]


class Feature(StrEnum):
    """A feature of SBML which an engine other than roadrunner may not simulate."""

    ALGEBRAIC_RULE = "algebraic rule"
    STOICHIOMETRY_MATH = "stoichiometry math"
    NON_CONSTANT_REFERENCE = "non-constant species reference"
    REFERENCE_INITIAL_ASSIGNMENT = "initial assignment of a species reference"
    DELAY = "delay csymbol"


def variable_features(model: libsbml.Model) -> Iterator[tuple[Feature, str]]:
    """Get the features of a model which COPASI and AMICI do not simulate as roadrunner.

    The algebraic rules, the stoichiometry maths, the species references
    which are not constant, the initial assignments of species references
    and the delay csymbol, each with its element for a message, e.g. `the
    non-constant species reference 'B_sr' of the reaction 'J1'`; an engine
    adds why it does not simulate it.

    Args:
        model: the model of the SBML.

    Yields:
        The feature and its element.
    """
    for rule in model.getListOfRules():
        if rule.isAlgebraic():
            yield Feature.ALGEBRAIC_RULE, "an algebraic rule"
    level = model.getLevel()
    references: set[str] = set()
    for reaction in model.getListOfReactions():
        for sr in (*reaction.getListOfReactants(), *reaction.getListOfProducts()):
            if sr.isSetId():
                references.add(sr.getId())
            if sr.isSetStoichiometryMath():
                yield (
                    Feature.STOICHIOMETRY_MATH,
                    f"the stoichiometry math of the species '{sr.getSpecies()}' in "
                    f"the reaction '{reaction.getId()}'",
                )
            if level >= 3 and sr.isSetConstant() and not sr.getConstant():
                name = sr.getId() or sr.getSpecies()
                yield (
                    Feature.NON_CONSTANT_REFERENCE,
                    f"the non-constant species reference '{name}' of the reaction "
                    f"'{reaction.getId()}'",
                )
    for assignment in model.getListOfInitialAssignments():
        if assignment.getSymbol() in references:
            yield (
                Feature.REFERENCE_INITIAL_ASSIGNMENT,
                f"an initial assignment of the species reference "
                f"'{assignment.getSymbol()}'",
            )
    for where, ast in sbml_maths(model):
        if reads_delay(ast):
            yield Feature.DELAY, f"the delay csymbol in the {where}"


#: the settings of roadrunner which COPASI and AMICI take as well
PORTABLE_SETTINGS = frozenset(
    {
        "absolute_tolerance",
        "maximum_num_steps",
        "maximum_time_step",
        "relative_tolerance",
    }
)

#: the settings of roadrunner which COPASI and AMICI have no equivalent of,
#: ignored with a warning: the settings of CVODE which they do not set, e.g.
#: the method (both integrate stiff systems) and the optimization of the
#: compilation of roadrunner
ROADRUNNER_ONLY_SETTINGS = frozenset(
    {
        "initial_time_step",
        "jit_optimization",
        "max_output_rows",
        "maximum_adams_order",
        "maximum_bdf_order",
        "minimum_time_step",
        "multiple_steps",
        "stiff",
        "variable_step_size",
    }
)


class EngineSettings:
    """The settings of roadrunner in an engine which takes `PORTABLE_SETTINGS`.

    Checks and validates the settings and reports, once per model, the ones
    the engine ignores and an absolute tolerance per kind which it takes as
    one number.
    """

    def __init__(
        self, engine: str, integrator: str, label: str, log: logging.Logger
    ) -> None:
        """Create the settings of a model.

        Args:
            engine: the name of the engine in a message, e.g. `COPASI`.
            integrator: the name of its integrator.
            label: the name of the model in a message.
            log: the logger of the engine, which reports.
        """
        self.engine = engine
        self.integrator = integrator
        self.label = label
        self._log = log
        self._ignored: set[str] = set()
        self._tolerance_reported = False

    def checked(
        self, settings: Mapping[str, IntegratorSetting]
    ) -> dict[str, float | int | AbsoluteTolerance]:
        """Check settings and get the ones the engine takes, validated.

        A setting of roadrunner which the engine has no equivalent of is
        reported once per model and left out.

        Args:
            settings: the settings by their names in roadrunner.

        Returns:
            The settings of `PORTABLE_SETTINGS`: the absolute tolerance an
            `AbsoluteTolerance`, the maximum number of steps an integer, the
            others floats.

        Raises:
            ValueError: if neither the engine nor roadrunner has a setting of
                a name, or a value is not valid.
        """
        unknown = sorted(set(settings) - PORTABLE_SETTINGS - ROADRUNNER_ONLY_SETTINGS)
        if unknown:
            raise ValueError(
                f"The integrator '{self.integrator}' of {self.engine} has no settings "
                f"{unknown}, its settings are {sorted(PORTABLE_SETTINGS)}."
            )
        ignored = sorted((set(settings) & ROADRUNNER_ONLY_SETTINGS) - self._ignored)
        if ignored:
            self._ignored.update(ignored)
            self._log.warning(
                "%s has no equivalent of the settings %s of roadrunner, the "
                "model '%s' ignores them.",
                self.engine,
                ignored,
                self.label,
            )
        checked: dict[str, float | int | AbsoluteTolerance] = {}
        for key, value in settings.items():
            if key not in PORTABLE_SETTINGS:
                continue
            if key == "absolute_tolerance":
                if isinstance(value, bool):
                    raise ValueError("The absolute tolerance is a number.")
                checked[key] = AbsoluteTolerance.of(value)
            elif key == "maximum_num_steps":
                if (
                    isinstance(value, bool | AbsoluteTolerance)
                    or not float(value).is_integer()
                    or value < 1
                ):
                    raise ValueError(
                        f"The maximum number of steps is a positive integer, not "
                        f"{value}."
                    )
                checked[key] = int(value)
            else:
                if isinstance(value, bool | AbsoluteTolerance) or not value >= 0:
                    raise ValueError(f"The setting '{key}' is a number, not {value}.")
                checked[key] = float(value)
        return checked

    def scalar_tolerance(
        self, tolerance: AbsoluteTolerance, factor: float = 1.0
    ) -> float:
        """Get the one absolute tolerance of the engine of a tolerance per kind.

        The smallest tolerance of the kinds and of the overrides, reported
        once per model when they differ, times a factor of the engine.

        Args:
            tolerance: the tolerance per kind.
            factor: the factor of the engine, e.g. `COPASI_ATOL_FACTOR`.

        Returns:
            The absolute tolerance.
        """
        smallest = tolerance.smallest()
        if len(tolerance.distinct_values()) > 1 and not self._tolerance_reported:
            self._tolerance_reported = True
            times = f", times {factor}" if factor != 1.0 else ""
            self._log.warning(
                "%s has one absolute tolerance, the model '%s' integrates "
                "with the smallest one of %s, %s%s.",
                self.engine,
                self.label,
                tolerance,
                smallest,
                times,
            )
        return smallest * factor


class _ValuesOf(Protocol):
    """A model whose current values are read and set in the selections of roadrunner."""

    def value(self, selection: str) -> float:
        """Get the current value of a selection."""
        ...

    def set_value(self, selection: str, value: float) -> None:
        """Set the current value of a selection."""
        ...


class EngineValues:
    """The current values of a model of an engine in the selections of roadrunner.

    The `ModelValues` of `sbmlsim.simulator.changes`, with which an executor
    reads and sets values between two integrations, of a model with `value`
    and `set_value`, e.g. `CopasiSBMLModel`.
    """

    def __init__(self, model: _ValuesOf) -> None:
        """Get the values of a model."""
        self._model = model

    def getValue(self, selection: str, /) -> float:
        """Get the current value of a selection, see the `value` of the model."""
        return self._model.value(selection)

    def setValue(self, selection: str, value: float, /) -> None:
        """Set the current value of a selection, see the `set_value` of the model."""
        self._model.set_value(selection, value)
