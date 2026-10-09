"""RoadRunner model."""

import logging
import weakref
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import libsbml
import numpy as np
import pandas as pd
import roadrunner
from roadrunner import _roadrunner  # ty: ignore[unresolved-import]

from sbmlsim.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.symbols import ModelSymbols, TargetKind
from sbmlsim.model.tolerances import (
    AbsoluteTolerance,
    StateTolerance,
    state_tolerances,
)
from sbmlsim.units import UnitRegistry, UnitsInformation
from sbmlsim.units import ureg as package_ureg
from sbmlsim.utils import md5_for_path

if TYPE_CHECKING:
    from sbmlsim.simulator.plan import Assignment

#: suffix of the parameter whose assignment rule is the math of an initial
#: assignment, see `RoadrunnerSBMLModel.initialize`
INITIAL_SUFFIX = "__initial"

#: what `RoadRunner.resetAll` resets: every variable to its initial value, with
#: the initial assignments evaluated
RESET_ALL = (
    roadrunner.SelectionRecord.TIME
    | roadrunner.SelectionRecord.RATE
    | roadrunner.SelectionRecord.FLOATING
    | roadrunner.SelectionRecord.BOUNDARY
    | roadrunner.SelectionRecord.COMPARTMENT
    | roadrunner.SelectionRecord.GLOBAL_PARAMETER
    | roadrunner.SelectionRecord.STOICHIOMETRY
)


def reset_all(r: roadrunner.RoadRunner) -> None:
    """Reset a model to its initial values, as `RoadRunner.resetAll`.

    roadrunner exposes the symbols of a model as attributes, so a model with a
    species or parameter named `reset` hides the method `reset`, which
    `resetAll` calls (case 00952 of the SBML Test Suite). The binding is
    called directly.

    Args:
        r: the roadrunner instance.
    """
    _roadrunner.RoadRunner_reset(r, RESET_ALL)


logger = logging.getLogger(__name__)


class RoadrunnerSBMLModel(AbstractModel):
    """Roadrunner model wrapper."""

    def __init__(
        self,
        source: str | Path,
        base_path: Path | None = None,
        changes: dict | None = None,
        sid: str | None = None,
        name: str | None = None,
        selections: list[str] | None = None,
        ureg: UnitRegistry | None = None,
        settings: dict | None = None,
        parameters: dict[str, float] | None = None,
    ):
        """Load the model into roadrunner, with changes, selections and settings.

        The `parameters` are added to the model as constant parameters, see
        `AbstractModel`; they are not selected by default.
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
        hierarchical = _is_hierarchical(sbml)
        if hierarchical:
            sbml = _flatten(sbml, path=self.source.path)
        if self.parameters:
            sbml = _with_parameters(sbml, self.parameters)
        self.symbols: ModelSymbols = ModelSymbols.from_sbml(sbml)
        #: entity with an initial assignment -> the parameter whose
        #: assignment rule is the math of the initial assignment, see
        #: `initialize`
        self.initial_helpers: dict[str, str] = {}
        if self.symbols.initial_assignments:
            sbml, self.initial_helpers = _with_initial_helpers(sbml)

        # load model
        self.r: roadrunner.RoadRunner | None = (
            roadrunner.RoadRunner(sbml)
            if hierarchical
            or self.initial_helpers
            or self.parameters
            or self.source.content is not None
            else self.load_roadrunner_model(source=self.source)
        )

        #: whether the instance was reset or loaded and not simulated since,
        #: see `initialize`
        self._reset_pending: bool = True
        #: the values an initialization changed since the last reset, the
        #: amount of a species, see `initialize`
        self._restore: dict[str, float] = {}
        #: whether the start of a simulation was reported, see `executor`
        self.warned_start: bool = False

        # set selections
        # logger.info("set selections")
        self.selections = self.set_timecourse_selections(
            self.r, selections=self.selections, exclude=set(self.parameters)
        )

        #: the absolute tolerance set last and the tolerances of the states,
        #: see `set_integrator_settings`
        self.absolute_tolerance: AbsoluteTolerance = AbsoluteTolerance()
        self._state_tolerances: list[StateTolerance] = []
        #: what the tolerances of the states were set for: the instance of
        #: roadrunner (a weak reference, which keeps no replaced instance
        #: alive), the initial volumes of its compartments and the vector of
        #: CVODE, see `_has_absolute_tolerance`
        self._tolerances_of: (
            tuple[weakref.ref[roadrunner.RoadRunner], list[float], list[float]] | None
        ) = None
        #: compartments whose volume was reported as raised to the floor
        self._raised_reported: set[str] = set()
        self.set_integrator_settings(
            **{"absolute_tolerance": AbsoluteTolerance(), **(settings or {})}
        )

        # normalize model changes
        self.uinfo = self.parse_units(ureg if ureg is not None else package_ureg)
        self.normalize(uinfo=self.uinfo)

    @property
    def r_loaded(self) -> roadrunner.RoadRunner:
        """Get the roadrunner instance of the model.

        Raises:
            ValueError: if no model is loaded.
        """
        if self.r is None:
            raise ValueError(f"The model '{self.sid}' is not loaded.")
        return self.r

    def initialize(self, assignments: Sequence["Assignment"]) -> None:
        """Initialize the model with values before its initialization.

        PEtab v2 sets the values of the parameter table and of the first
        conditions on the model before it is initialized, so an initial
        assignment follows a changed parameter, and a target which is set
        replaces its own initial assignment. roadrunner reinitializes the
        model when an initial value is set with `init(...)`, which costs a
        compilation of the model per value. The model is therefore
        initialized as it was loaded, the values are set as the current
        values, and the initial assignments which read a changed entity are
        evaluated again, in their order, from the assignment rule of their
        helper parameter, which roadrunner evaluates on the current values.

        roadrunner queues the events which fire at the time 0 with every
        reset, a loaded model counts as one, and a simulation fires them once:
        a second reset without a simulation in between fires them twice (case
        01757 of the SBML Test Suite). A model which was not simulated since
        its last reset is therefore not reset again, the values an earlier
        initialization set are restored instead; `simulated` records a
        simulation.

        A compartment keeps the concentration of the species whose initial
        value is a concentration, as at the initialization of SBML, and of a
        species which is set as a concentration, and the amount of every
        other species.

        Args:
            assignments: the pre-initialization assignments of a plan. A
                formula reads the parameters after the values are set.
        """
        self._reset()
        r = self.r_loaded
        symbols = self.symbols
        entities: set[str] = set()
        concentrations: set[str] = set()
        amounts: set[str] = set()
        for a in assignments:
            entity = symbols.entity(a.target)
            entities.add(entity)
            if a.kind is TargetKind.SPECIES_CONCENTRATION:
                concentrations.add(entity)
            elif a.kind is TargetKind.SPECIES_AMOUNT:
                amounts.add(entity)
        # the values first, then the formulas, which read the values and are
        # evaluated before any of them is set
        values = [a for a in assignments if a.value is not None]
        for a in values:
            if a.kind is TargetKind.COMPARTMENT:
                self._set_compartment(a.target, float(a.value), concentrations, amounts)  # ty: ignore[invalid-argument-type]
        for a in values:
            if a.kind is not TargetKind.COMPARTMENT:
                self._set(a.target, float(a.value))  # ty: ignore[invalid-argument-type]
        # the package of the simulator imports the models
        from sbmlsim.simulator.formula import compile_formula

        evaluated: list[tuple[Assignment, float]] = []
        for a in assignments:
            if a.formula is not None:
                formula = compile_formula(a.formula)
                evaluated.append(
                    (a, formula.evaluate([r.getValue(s) for s in formula.symbols]))
                )
        for a, value in evaluated:
            if a.kind is TargetKind.COMPARTMENT:
                self._set_compartment(a.target, value, concentrations, amounts)
            else:
                self._set(a.target, value)

        dependencies = symbols.initial_assignment_dependencies or {}
        changed = set(entities)
        for entity in symbols.initial_assignment_order:
            if (
                entity in entities
                or entity not in self.initial_helpers
                or not (dependencies[entity] & changed)
            ):
                continue
            value = float(r.getValue(self.initial_helpers[entity]))
            if entity in symbols.compartments:
                self._set_compartment(entity, value, concentrations, amounts)
            elif entity in symbols.species and entity not in symbols.only_substance:
                self._set(f"[{entity}]", value)
            else:
                self._set(entity, value)
            changed.add(entity)

    def _reset(self) -> None:
        """Set the model to the state it was loaded in, see `initialize`."""
        r = self.r_loaded
        if self._reset_pending:
            for key, value in self._restore.items():
                r.setValue(key, value)
        else:
            reset_all(r)
            self._reset_pending = True
        self._restore = {}

    def simulated(self) -> None:
        """Record that the model was simulated since its last initialization.

        The next `initialize` resets the model, see there.
        """
        self._reset_pending = False

    def _record(self, key: str) -> None:
        """Record the value an initialization replaces, see `initialize`."""
        if key not in self._restore:
            self._restore[key] = float(self.r_loaded.getValue(key))

    def _set(self, selection: str, value: float) -> None:
        """Set a value in an initialization and record the value it replaces.

        A species is recorded by its amount, which does not depend on the
        compartment, so the records restore in any order.
        """
        self._record(self.symbols.entity(selection))
        self.r_loaded.setValue(selection, value)

    def _set_compartment(
        self,
        compartment: str,
        value: float,
        concentrations: set[str],
        amounts: set[str],
    ) -> None:
        """Set a compartment before the initialization.

        Args:
            compartment: the compartment.
            value: its size.
            concentrations: species which are set as concentrations.
            amounts: species which are set as amounts.
        """
        r = self.r_loaded
        symbols = self.symbols
        kept = {
            s: r.getValue(f"[{s}]")
            for s, c in symbols.species_compartment.items()
            if c == compartment
            and s not in amounts
            and (s in symbols.initial_concentration or s in concentrations)
        }
        for species in kept:
            self._record(species)
        self._set(compartment, value)
        for species, concentration in kept.items():
            r.setValue(f"[{species}]", concentration)

    @staticmethod
    def from_abstract_model(
        abstract_model: AbstractModel,
        selections: list[str] | None = None,
        ureg: UnitRegistry | None = None,
        settings: dict | None = None,
    ):
        """Create from AbstractModel."""
        logger.debug("RoadrunnerSBMLModel from AbstractModel")
        return RoadrunnerSBMLModel(
            source=abstract_model.source.content
            if abstract_model.source.content is not None
            else abstract_model.source.source,
            changes=abstract_model.changes,
            parameters=abstract_model.parameters,
            sid=abstract_model.sid,
            name=abstract_model.name,
            base_path=abstract_model.base_path,
            selections=selections,
            ureg=ureg,
            settings=settings,
        )

    @classmethod
    def load_roadrunner_model(
        cls,
        source: Source,
    ) -> roadrunner.RoadRunner:
        """Load model from given source.

        :param source: path to SBML model or SBML string
        :param state_path: path to rr state
        :return: roadrunner instance
        """
        if isinstance(source, (str, Path)):
            source = Source.from_source(source=source)

        # load model
        if source.path is not None:
            sbml_path: Path = source.path
            # state_path: Path = RoadrunnerSBMLModel.get_state_path(sbml_path=sbml_path)

            r = roadrunner.RoadRunner(str(sbml_path))
            # FIXME: see https://github.com/sys-bio/roadrunner/issues/963
            # if state_path.exists():
            #     logger.debug(f"Load model from state: '{state_path}'")
            #     r = roadrunner.RoadRunner()
            #     r.loadState(str(state_path))
            #     # with open(state_path, "rb") as fin:
            #     #     r.loadStateS(fin.read())
            #     logger.debug(f"Model loaded from state: '{state_path}'")
            # else:
            #     logger.info(f"Load model from SBML: '{sbml_path}'")
            #     r = roadrunner.RoadRunner(str(sbml_path))
            #     # save state
            #     r.saveState(str(state_path))
            #     # with open(state_path, "wb") as fout:
            #     #     fout.write(r.saveStateS(opt="b"))
            #     logger.info(f"Save state: '{state_path}'")

        elif source.is_content():
            r = roadrunner.RoadRunner(str(source.content))

        return r

    @staticmethod
    def get_state_path(sbml_path: Path) -> Path | None:
        """Get path of the state file.

        The state file is a binary file which allows fast model loading.
        """
        md5 = md5_for_path(sbml_path)
        return Path(f"{sbml_path}_rr{roadrunner.__version__}_{md5}.state")

    def parse_units(self, ureg: UnitRegistry) -> UnitsInformation:
        """Parse units from SBML model."""
        uinfo: UnitsInformation
        if self.source.content is not None:
            uinfo = UnitsInformation.from_sbml(sbml=self.source.content, ureg=ureg)
        elif self.source.path is not None:
            uinfo = UnitsInformation.from_sbml(sbml=self.source.path, ureg=ureg)
        else:
            raise ValueError(f"Model source has no content and no path: {self.source}")

        return uinfo

    @classmethod
    def set_timecourse_selections(
        cls,
        r: roadrunner.RoadRunner,
        selections: list[str] | None = None,
        exclude: Collection[str] = (),
    ) -> list[str]:
        """Set the selections of the simulations.

        Without selections every entity of the model is selected, except a
        compartment without a size (`NaN`), e.g. a membrane whose area the
        model does not use, which has no value to record.

        Args:
            r: the roadrunner instance.
            selections: the selections, every entity of the model without.
            exclude: ids which are not selected by default, e.g. the
                parameters a problem added to the model.

        Returns:
            The selections.
        """
        if selections is None:
            r_model: roadrunner.ExecutableModel = r.model
            compartments = [
                cid
                for cid, size in zip(
                    r_model.getCompartmentIds(),
                    r_model.getCompartmentVolumes(),
                    strict=True,
                )
                if not np.isnan(size)
            ]

            r.timeCourseSelections = [
                "time",
                *r_model.getFloatingSpeciesIds(),
                *r_model.getBoundarySpeciesIds(),
                *[
                    pid
                    for pid in r_model.getGlobalParameterIds()
                    if not pid.endswith(INITIAL_SUFFIX) and pid not in exclude
                ],
                *r_model.getReactionIds(),
                *compartments,
            ]
            r.timeCourseSelections += [
                f"[{key}]"
                for key in (
                    r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds()
                )
            ]
        else:
            r.timeCourseSelections = selections
        return list(r.timeCourseSelections)

    def set_selections(self, selections: Sequence[str] | None) -> None:
        """Set the selections of the simulations of the model.

        Args:
            selections: the selections, every entity of the model for `None`,
                see `set_timecourse_selections`; the parameters added to the
                model are not selected by default.

        Raises:
            RuntimeError: if roadrunner has no selection of a name.
        """
        self.selections = self.set_timecourse_selections(
            self.r_loaded,
            selections=None if selections is None else list(selections),
            exclude=set(self.parameters),
        )

    def set_integrator_settings(
        self, **kwargs: float | int | bool | AbsoluteTolerance
    ) -> roadrunner.Integrator:
        """Set settings of the integrator.

        Every setting of the integrator of roadrunner is passed on, for CVODE
        e.g. `relative_tolerance`, `stiff`, `variable_step_size`,
        `initial_time_step`, `minimum_time_step`, `maximum_time_step` and
        `maximum_num_steps`. `absolute_tolerance`, a float or an
        `AbsoluteTolerance`, is set as one tolerance per state, see
        `sbmlsim.model.tolerances`.

        A setting the integrator has already is not set again, so that a
        simulator which applies its settings before every simulation costs
        nothing: the tolerances of the states are computed again only for
        another tolerance, another instance of roadrunner, other initial
        volumes of the compartments or a vector of CVODE which was set
        elsewhere.

        Args:
            **kwargs: the settings by their names in roadrunner.

        Returns:
            The integrator.

        Raises:
            ValueError: if the integrator has no setting of a name, or an
                override of the absolute tolerance is not a state.
        """
        integrator: roadrunner.Integrator = self.r_loaded.getIntegrator()
        names = _setting_names(integrator)
        unknown = sorted(set(kwargs) - names)
        if unknown:
            raise ValueError(
                f"The integrator '{integrator.getName()}' has no settings "
                f"{unknown}, its settings are {sorted(names)}."
            )
        for key, value in kwargs.items():
            if key == "absolute_tolerance":
                if isinstance(value, bool):
                    raise ValueError("The absolute tolerance is a number.")
                tolerance = AbsoluteTolerance.of(value)
                if not self._has_absolute_tolerance(tolerance, integrator):
                    self._set_absolute_tolerance(tolerance)
            elif integrator.getValue(key) != value:
                integrator.setValue(key, value)
                logger.debug("Integrator setting: '%s = %s'", key, value)
        return integrator

    def _has_absolute_tolerance(
        self, tolerance: AbsoluteTolerance, integrator: roadrunner.Integrator
    ) -> bool:
        """Check whether CVODE integrates with the tolerances of a tolerance.

        The tolerances of the states depend on the instance of roadrunner and
        the initial volumes of the compartments; the vector of CVODE is
        compared as well, it may have been set on the integrator directly.

        Args:
            tolerance: the tolerance.
            integrator: the integrator of the model.
        """
        if tolerance != self.absolute_tolerance or self._tolerances_of is None:
            return False
        r = self.r_loaded
        instance, volumes, vector = self._tolerances_of
        return (
            instance() is r
            and _floats(r.model.getCompartmentInitVolumes()) == volumes
            and _floats(integrator.getAbsoluteToleranceVector()) == vector
        )

    def state_ids(self) -> list[str]:
        """Get the ids of the states which the integrator integrates."""
        r = self.r_loaded
        n = len(r.getIntegrator().getAbsoluteToleranceVector())
        return [r.model.getStateVectorId(k) for k in range(n)]

    def _set_absolute_tolerance(self, tolerance: AbsoluteTolerance) -> None:
        """Set the absolute tolerance of every state, see `tolerances`.

        roadrunner turns a single value into a vector by its own scaling, so
        the tolerances are set as the vector of CVODE, in the order of
        `state_ids`. `setIndividualTolerance` is not used: it indexes a species
        by its index among the floating species and a rate rule after them,
        while CVODE integrates the rate rules first.
        """
        r = self.r_loaded
        volumes = dict(
            zip(
                r.model.getCompartmentIds(),
                (float(v) for v in r.model.getCompartmentInitVolumes()),
                strict=True,
            )
        )
        states = state_tolerances(self.state_ids(), self.symbols, volumes, tolerance)
        for state in states:
            if (
                state.volume_raised
                and state.compartment is not None
                and state.compartment not in self._raised_reported
            ):
                self._raised_reported.add(state.compartment)
                logger.warning(
                    "The compartment '%s' of the model '%s' has the initial volume "
                    "%s; the absolute tolerances of its species use the volume %s.",
                    state.compartment,
                    self.sid or r.model.getModelName(),
                    volumes.get(state.compartment),
                    state.volume,
                )
        integrator: roadrunner.Integrator = r.getIntegrator()
        integrator.setValue(
            "absolute_tolerance", [state.absolute_tolerance for state in states]
        )
        self.absolute_tolerance = tolerance
        self._state_tolerances = states
        self._tolerances_of = (
            weakref.ref(r),
            _floats(r.model.getCompartmentInitVolumes()),
            _floats(integrator.getAbsoluteToleranceVector()),
        )

    def tolerances(self) -> pd.DataFrame:
        """Get the absolute tolerance of every state.

        Returns:
            A row per state with `sid`, `kind`, `compartment`, `volume` (the
            reference volume of a concentration species) and
            `absolute_tolerance`.
        """
        return pd.DataFrame(
            [
                {
                    "sid": s.sid,
                    "kind": s.kind.value,
                    "compartment": s.compartment,
                    "volume": s.volume,
                    "absolute_tolerance": s.absolute_tolerance,
                }
                for s in self._state_tolerances
            ],
            columns=["sid", "kind", "compartment", "volume", "absolute_tolerance"],
        )

    @staticmethod
    def parameter_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
        """Create GlobalParameter DataFrame.

        :return: pandas DataFrame
        """
        r_model: roadrunner.ExecutableModel = r.model
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(r.getCurrentSBML())
        model: libsbml.Model = doc.getModel()
        # the helpers of the initial assignments are not parameters of the
        # model, see `initialize`
        sids = [
            sid
            for sid in r_model.getGlobalParameterIds()
            if not sid.endswith(INITIAL_SUFFIX)
        ]
        parameters: list[libsbml.Parameter] = [model.getParameter(sid) for sid in sids]
        data = {
            "sid": sids,
            "value": [r.getValue(sid) for sid in sids],
            "unit": [p.getUnits() for p in parameters],
            "constant": [p.getConstant() for p in parameters],
            "name": [p.getName() for p in parameters],
        }
        return pd.DataFrame(
            data, columns=pd.Index(["sid", "value", "unit", "constant", "name"])
        )

    @staticmethod
    def species_df(r: roadrunner.RoadRunner) -> pd.DataFrame:
        """Create FloatingSpecies DataFrame.

        :return: pandas DataFrame
        """
        r_model: roadrunner.ExecutableModel = r.model
        sbml_str = r.getCurrentSBML()

        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml_str)
        model: libsbml.Model = doc.getModel()

        sids = r_model.getFloatingSpeciesIds() + r_model.getBoundarySpeciesIds()
        species: list[libsbml.Species] = [model.getSpecies(sid) for sid in sids]

        data = {
            "sid": sids,
            "concentration": np.concatenate(
                [
                    r_model.getFloatingSpeciesConcentrations(),
                    r_model.getBoundarySpeciesConcentrations(),
                ],
                axis=0,
            ),
            "amount": np.concatenate(
                [
                    r.model.getFloatingSpeciesAmounts(),
                    r.model.getBoundarySpeciesAmounts(),
                ],
                axis=0,
            ),
            "unit": [s.getUnits() for s in species],
            "constant": [s.getConstant() for s in species],
            "boundaryCondition": [s.getBoundaryCondition() for s in species],
            "name": [s.getName() for s in species],
        }

        return pd.DataFrame(
            data,
            columns=pd.Index(
                [
                    "sid",
                    "concentration",
                    "amount",
                    "unit",
                    "constant",
                    "boundaryCondition",
                    "name",
                ]
            ),
        )


#: the names of the settings of an integrator by its name, see `_setting_names`
_SETTING_NAMES: dict[str, frozenset[str]] = {}


def _setting_names(integrator: roadrunner.Integrator) -> frozenset[str]:
    """Get the names of the settings of an integrator, read once per kind."""
    name = str(integrator.getName())
    names = _SETTING_NAMES.get(name)
    if names is None:
        names = frozenset(integrator.getSettings())
        _SETTING_NAMES[name] = names
    return names


def _floats(values: Sequence[float] | np.ndarray) -> list[float]:
    """Get the values of an array of roadrunner as floats."""
    return np.asarray(values, dtype=float).tolist()


def _with_initial_helpers(sbml: str) -> tuple[str, dict[str, str]]:
    """Add a helper parameter for every initial assignment of a model.

    The helper `<entity>__initial` is a parameter with an assignment rule
    which is the math of the initial assignment of the entity, so roadrunner
    evaluates the math on the current values whenever the helper is read, see
    `RoadrunnerSBMLModel.initialize`. The initial assignments stay.

    Args:
        sbml: the SBML of the model.

    Returns:
        The SBML with the helpers and the helper of every entity.

    Raises:
        ValueError: if the model already has an entity of the id of a helper.
    """
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    model: libsbml.Model = doc.getModel()
    helpers: dict[str, str] = {}
    for assignment in list(model.getListOfInitialAssignments()):
        if assignment.getMath() is None:
            # an initial assignment without math assigns nothing
            continue
        entity = assignment.getSymbol()
        helper = f"{entity}{INITIAL_SUFFIX}"
        if model.getElementBySId(helper) is not None:
            raise ValueError(
                f"The model already has an entity '{helper}', the id of the "
                f"helper of the initial assignment of '{entity}'."
            )
        parameter: libsbml.Parameter = model.createParameter()
        parameter.setId(helper)
        parameter.setConstant(False)
        rule: libsbml.AssignmentRule = model.createAssignmentRule()
        rule.setVariable(helper)
        rule.setMath(assignment.getMath().deepCopy())
        helpers[entity] = helper
    return libsbml.writeSBMLToString(doc), helpers


def _is_hierarchical(sbml: str) -> bool:
    """Check whether a model uses the package `comp` of hierarchical models."""
    if "comp" not in sbml:
        return False
    doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(sbml)
    return bool(doc.isPackageEnabled("comp"))


def _flatten(sbml: str, path: Path | None) -> str:
    """Flatten a hierarchical model.

    The model is read from its file if it has one, so that libsbml resolves
    its external model definitions relative to the file.

    Args:
        sbml: the SBML of the model.
        path: the file of the model, `None` for the SBML as a string.

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
