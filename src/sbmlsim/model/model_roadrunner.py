"""RoadRunner model."""

import logging
import tempfile
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import libsbml
import numpy as np
import pandas as pd
import roadrunner
from roadrunner import _roadrunner  # ty: ignore[unresolved-import]

from sbmlsim.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.symbols import ModelSymbols, TargetKind
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

    IntegratorSettingKeys: ClassVar[set[str]] = {
        "variable_step_size",
        "stiff",
        "absolute_tolerance",
        "relative_tolerance",
    }

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
        r: roadrunner.RoadRunner | None = None
        if _is_hierarchical(sbml):
            # roadrunner flattens a hierarchical model and resolves its
            # external model definitions relative to the file, the symbols are
            # the ones of the model it simulates
            r = self.load_roadrunner_model(source=self.source)
            sbml = r.getCurrentSBML()
        if self.parameters:
            sbml = _with_parameters(sbml, self.parameters)
            r = None
        self.symbols: ModelSymbols = ModelSymbols.from_sbml(sbml)
        #: entity with an initial assignment -> the parameter whose
        #: assignment rule is the math of the initial assignment, see
        #: `initialize`
        self.initial_helpers: dict[str, str] = {}
        if self.symbols.initial_assignments:
            sbml, self.initial_helpers = _with_initial_helpers(sbml)
            r = None

        # load model
        self.r: roadrunner.RoadRunner | None = (
            r
            if r is not None
            else (
                roadrunner.RoadRunner(sbml)
                if self.initial_helpers
                or self.parameters
                or self.source.content is not None
                else self.load_roadrunner_model(source=self.source)
            )
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

        # set integrator settings
        # logger.info("set integrator settings")
        if settings:
            RoadrunnerSBMLModel.set_integrator_settings(self.r, **settings)

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
            assignments: the pre-initialization assignments of a plan, values
                only.

        Raises:
            ValueError: if an assignment is a formula.
        """
        self._reset()
        r = self.r_loaded
        symbols = self.symbols
        entities: set[str] = set()
        concentrations: set[str] = set()
        amounts: set[str] = set()
        for a in assignments:
            if a.value is None:
                raise ValueError(
                    f"The pre-initialization value of '{a.target}' is the "
                    f"formula '{a.formula}', it must be a number."
                )
            entity = symbols.entity(a.target)
            entities.add(entity)
            if a.kind is TargetKind.SPECIES_CONCENTRATION:
                concentrations.add(entity)
            elif a.kind is TargetKind.SPECIES_AMOUNT:
                amounts.add(entity)
        for a in assignments:
            if a.kind is TargetKind.COMPARTMENT:
                self._set_compartment(a.target, float(a.value), concentrations, amounts)  # ty: ignore[invalid-argument-type]
        for a in assignments:
            if a.kind is not TargetKind.COMPARTMENT:
                self._set(a.target, float(a.value))  # ty: ignore[invalid-argument-type]

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

    @classmethod
    def copy_roadrunner_model(cls, r: roadrunner.RoadRunner) -> roadrunner.RoadRunner:
        """Copy roadrunner model by using the state."""
        with tempfile.NamedTemporaryFile() as ftmp:
            filename = ftmp.name
            r.saveState(filename)
            r2 = roadrunner.RoadRunner()
            r2.loadState(filename)
        return r2

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
                *r_model.getCompartmentIds(),
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

    @staticmethod
    def set_integrator_settings(
        r: roadrunner.RoadRunner, **kwargs
    ) -> roadrunner.Integrator:
        """Set integrator settings.

        Keys are:
            variable_step_size [boolean]
            stiff [boolean]
            absolute_tolerance [float]
            relative_tolerance [float]

        """
        integrator: roadrunner.Integrator = r.getIntegrator()
        for key, value in kwargs.items():
            if key not in RoadrunnerSBMLModel.IntegratorSettingKeys:
                logger.debug(
                    "Unsupported integrator key for roadrunner integrator: '%s'", key
                )
                continue

            # adapt the absolute_tolerance relative to the amounts
            if key == "absolute_tolerance":
                value = min(
                    value,
                    value * RoadrunnerSBMLModel._tolerance_volume_factor(r),
                )

            integrator.setValue(key, value)
            logger.debug("Integrator setting: '%s = %s'", key, value)
        return integrator

    @staticmethod
    def _tolerance_volume_factor(r: roadrunner.RoadRunner) -> float:
        """Get the factor of the absolute tolerance for amounts.

        The species of a model are integrated as amounts, so the absolute
        tolerance of the concentrations is scaled by the smallest volume. The
        initial volumes are used, not the current ones, so that the tolerance
        does not depend on the state an earlier simulation left behind;
        compartments without a finite positive volume are ignored.

        Args:
            r: the roadrunner instance with a loaded model.

        Returns:
            The smallest finite positive initial volume, 1 if there is none.
        """
        volumes = np.asarray(r.model.getCompartmentInitVolumes(), dtype=float)
        volumes = volumes[np.isfinite(volumes) & (volumes > 0)]
        if volumes.size == 0:
            return 1.0
        return float(np.nanmin(volumes))

    @staticmethod
    def set_default_settings(r: roadrunner.RoadRunner, **kwargs):
        """Set default settings of integrator."""
        RoadrunnerSBMLModel.set_integrator_settings(
            r,
            variable_step_size=True,
            stiff=True,
            absolute_tolerance=1e-8,
            relative_tolerance=1e-8,
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
