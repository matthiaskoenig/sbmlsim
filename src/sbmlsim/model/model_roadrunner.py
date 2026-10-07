"""RoadRunner model."""

import logging
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import libsbml
import numpy as np
import pandas as pd
import roadrunner

from sbmlsim.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.symbols import ModelSymbols
from sbmlsim.units import Quantity, UnitRegistry, UnitsInformation
from sbmlsim.units import ureg as package_ureg
from sbmlsim.utils import md5_for_path

if TYPE_CHECKING:
    from sbmlsim.simulator.plan import Assignment

#: suffix of the parameter which holds a freed initial assignment, see
#: `RoadrunnerSBMLModel.free_initial_assignments`
INITIAL_SUFFIX = "__initial"

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
    ):
        """Load the model into roadrunner, with changes, selections and settings."""
        super().__init__(
            source=source,
            language_type=AbstractModel.LanguageType.SBML,
            changes=changes,
            sid=sid,
            name=name,
            base_path=base_path,
            selections=selections,
        )

        # check SBML
        if self.language_type != AbstractModel.LanguageType.SBML:
            raise ValueError(f"language_type not supported '{self.language_type}'.")

        # load model
        self.r: roadrunner.RoadRunner | None = self.load_roadrunner_model(
            source=self.source
        )
        #: the SBML the instance was loaded from, with the freed initial
        #: assignments, see `free_initial_assignments`
        self._sbml: str = (
            self.source.content
            if self.source.content is not None
            else Path(str(self.source.path)).read_text(encoding="utf-8")
        )
        #: the symbols of the model, read once
        self.symbols: ModelSymbols = ModelSymbols.from_sbml(self._sbml)
        #: the original initial value of every target a plan set, by its key
        #: `init(...)`, see `set_initial_values`
        self._init_original: dict[str, float] = {}
        #: entity -> the parameter which holds its initial assignment
        self.derived_initial: dict[str, str] = {}

        # set selections
        # logger.info("set selections")
        self.selections = self.set_timecourse_selections(
            self.r, selections=self.selections
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

    def _entity_init_key(self, entity: str) -> str:
        """Get the selection of the initial value of an entity as the model means it.

        A species is its concentration unless it has only substance units,
        which is what an initial assignment of the species means.
        """
        if entity in self.symbols.species and entity not in self.symbols.only_substance:
            return f"init([{entity}])"
        return f"init({entity})"

    def set_initial_values(self, assignments: Sequence["Assignment"]) -> None:
        """Set the initial values of a plan and restore the ones it does not set.

        roadrunner keeps a value set with `init(...)` across `resetToOrigin`,
        so the model records the original initial value of every target the
        first time it is set and writes it back before a plan which does not
        set it. An entity whose initial assignment was freed
        (`free_initial_assignments`) has no fixed original value: when the
        plan does not set it, it gets the value of its initial assignment,
        which needs a `reset` in between. The caller initializes the model
        with `reset()` afterwards.

        Args:
            assignments: the pre-initialization assignments of a plan, values
                only.

        Raises:
            ValueError: if an assignment is a formula.
        """
        r = self.r_loaded
        # `init(S)` is the initial amount, `init([S])` the initial concentration
        values: dict[str, float] = {}
        entities: dict[str, str] = {}
        for a in assignments:
            if a.value is None:
                raise ValueError(
                    f"The pre-initialization value of '{a.target}' is the "
                    f"formula '{a.formula}', it must be a number."
                )
            key = f"init({a.target})"
            values[key] = a.value
            entities[key] = self.symbols.entity(a.target)

        for key, value in self._init_original.items():
            if key not in values:
                r.setValue(key, value)
        for key, value in values.items():
            if key not in self._init_original and (
                entities[key] not in self.derived_initial
            ):
                self._init_original[key] = float(r.getValue(key))
            r.setValue(key, value)

        set_entities = set(entities.values())
        missing = [e for e in self.derived_initial if e not in set_entities]
        if missing:
            r.resetAll()
            initial = {e: float(r.getValue(self.derived_initial[e])) for e in missing}
            for entity, value in initial.items():
                r.setValue(self._entity_init_key(entity), value)

    def free_initial_assignments(self, entities: Collection[str]) -> None:
        """Make entities with an initial assignment settable before initialization.

        roadrunner refuses `init(p)` for a parameter with an initial
        assignment, and a value set with `init(...)` replaces the initial
        assignment of a species for good. The model is therefore derived once:
        the initial assignment of every such entity moves to a new parameter
        `<entity>__initial`, and `set_initial_values` sets the entity to the
        value of that parameter whenever a plan does not set it. The model is
        loaded again from the derived SBML with its integrator settings and
        selections.

        Args:
            entities: entities a plan sets before the initialization; an
                entity without an initial assignment or already freed is
                skipped.
        """
        new = sorted(
            e
            for e in entities
            if e in self.symbols.initial_assignments and e not in self.derived_initial
        )
        if not new:
            return
        doc: libsbml.SBMLDocument = libsbml.readSBMLFromString(self._sbml)
        model: libsbml.Model = doc.getModel()
        for entity in new:
            helper = f"{entity}{INITIAL_SUFFIX}"
            if model.getElementBySId(helper) is not None:
                raise ValueError(
                    f"The model already has an entity '{helper}', the id of the "
                    f"initial assignment of '{entity}'."
                )
            assignment: libsbml.InitialAssignment = model.getInitialAssignmentBySymbol(
                entity
            )
            math: libsbml.ASTNode = assignment.getMath().deepCopy()
            model.removeInitialAssignment(entity)

            parameter: libsbml.Parameter = model.createParameter()
            parameter.setId(helper)
            parameter.setConstant(True)
            moved: libsbml.InitialAssignment = model.createInitialAssignment()
            moved.setSymbol(helper)
            moved.setMath(math)

            # the entity needs a value of its own, the plan or the helper
            # replaces it before every simulation
            element = model.getElementBySId(entity)
            if isinstance(element, libsbml.Parameter) and not element.isSetValue():
                element.setValue(0.0)
            elif isinstance(element, libsbml.Species) and not (
                element.isSetInitialAmount() or element.isSetInitialConcentration()
            ):
                if element.getHasOnlySubstanceUnits():
                    element.setInitialAmount(0.0)
                else:
                    element.setInitialConcentration(0.0)
            elif isinstance(element, libsbml.Compartment) and not element.isSetSize():
                element.setSize(1.0)
            self.derived_initial[entity] = helper

        self._sbml = libsbml.writeSBMLToString(doc)
        old = self.r_loaded
        r = roadrunner.RoadRunner(self._sbml)
        integrator: roadrunner.Integrator = old.getIntegrator()
        for key in self.IntegratorSettingKeys:
            r.getIntegrator().setValue(key, integrator.getValue(key))
        r.timeCourseSelections = list(old.timeCourseSelections)
        self.r = r
        self._init_original = {}
        logger.info(
            "The initial assignments of %s of '%s' are set before the "
            "initialization, the model is derived",
            new,
            self.sid,
        )

    @property
    def Q_(self) -> type[Quantity]:
        """Quantity to create quantities for model changes."""
        return self.uinfo.ureg.Quantity

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
            source=abstract_model.source.source,
            changes=abstract_model.changes,
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
        cls, r: roadrunner.RoadRunner, selections: list[str] | None = None
    ) -> list[str]:
        """Set the model selections for timecourse simulation."""
        if selections is None:
            r_model: roadrunner.ExecutableModel = r.model

            r.timeCourseSelections = [
                "time",
                *r_model.getFloatingSpeciesIds(),
                *r_model.getBoundarySpeciesIds(),
                *[
                    pid
                    for pid in r_model.getGlobalParameterIds()
                    if not pid.endswith(INITIAL_SUFFIX)
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
        sids = r_model.getGlobalParameterIds()
        parameters: list[libsbml.Parameter] = [model.getParameter(sid) for sid in sids]
        data = {
            "sid": sids,
            "value": r_model.getGlobalParameterValues(),
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
