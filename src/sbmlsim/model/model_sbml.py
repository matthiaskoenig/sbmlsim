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
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import libsbml
from pint.facets.plain import PlainUnit

from sbmlsim.engine import Engine
from sbmlsim.model.event_rows import MARKER_PREFIX
from sbmlsim.model.model import AbstractModel
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
    """

    engine: ClassVar[Engine]

    def __init__(
        self,
        source: str | Path,
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

        Every entity of the model which has a value: the time, the amounts of
        the floating and of the boundary species, the global parameters, the
        reactions and the compartments with a size, then the concentrations
        of the floating and of the boundary species, each in the order of the
        SBML. A compartment without a size (`NaN`), e.g. a membrane whose area
        the model does not use, has no value to record; the parameters added
        to the model and the helpers of sbmlsim are not selected.

        Returns:
            The selections.
        """
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(self.sbml)
        model: libsbml.Model = doc.getModel()
        floating: list[str] = []
        boundary: list[str] = []
        for s in model.getListOfSpecies():
            (boundary if s.getBoundaryCondition() else floating).append(s.getId())
        sized = {a.getSymbol() for a in model.getListOfInitialAssignments()} | {
            r.getVariable() for r in model.getListOfRules() if not r.isAlgebraic()
        }
        compartments = [
            c.getId()
            for c in model.getListOfCompartments()
            if c.getId() in sized or (c.isSetSize() and not math.isnan(c.getSize()))
        ]
        return [
            "time",
            *floating,
            *boundary,
            *[
                p.getId()
                for p in model.getListOfParameters()
                if not _is_helper(p.getId()) and p.getId() not in self.parameters
            ],
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
            RuntimeError: if the model has no selection of the name.
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
            selections: the selections, `default_selections` for `None`.
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
    def state_ids(self) -> list[str]:
        """Get the ids of the states which the integrator integrates."""


def load_model(
    model: AbstractModel | str | Path,
    engine: Engine,
    *,
    ureg: UnitRegistry | None = None,
    jit_optimization: JitOptimization | str | None = None,
) -> SBMLModel:
    """Load a model in an engine.

    A model of the engine is the model itself. A model of another engine is
    loaded again from what does not depend on its engine: its `sbml`, its
    added parameters, its changes, its selections, its id and its base path;
    a state set on the model of the other engine is not carried over.

    Args:
        model: a loaded model, the description of a model or its path or SBML.
        engine: the engine.
        ureg: the unit registry of the units of the model, the registry of
            the package without one; a model of another engine keeps its own.
        jit_optimization: the optimization level of the compilation of a
            model of roadrunner, see
            `sbmlsim.model.model_roadrunner.resolve_jit_optimization`.

    Returns:
        The model of the engine.

    Raises:
        ImportError: for the engine COPASI without python-copasi.
    """
    if isinstance(model, SBMLModel):
        if model.engine is engine:
            return model
        if ureg is None:
            ureg = model.uinfo.ureg
        model = AbstractModel(
            source=model.sbml,
            sid=model.sid,
            name=model.name,
            base_path=model.base_path,
            changes=dict(model.changes),
            selections=None if model.selections is None else list(model.selections),
            parameters=model.parameters,
        )
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
    # COPASI is an optional extra, imported when a model is loaded in it
    from sbmlsim.model.model_copasi import CopasiSBMLModel  # ty: ignore[unresolved-import]

    if isinstance(model, AbstractModel):
        return CopasiSBMLModel.from_abstract_model(abstract_model=model, ureg=ureg)
    return CopasiSBMLModel(source=model, ureg=ureg)


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
