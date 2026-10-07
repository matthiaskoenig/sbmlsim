"""RoadRunner model."""

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import libsbml
import numpy as np
import pandas as pd
import roadrunner

from sbmlsim.model import AbstractModel
from sbmlsim.model.model_resources import Source
from sbmlsim.model.symbols import ModelSymbols, TargetKind
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

        # the SBML of the model and its symbols, read once
        sbml: str = (
            self.source.content
            if self.source.content is not None
            else Path(str(self.source.path)).read_text(encoding="utf-8")
        )
        self.symbols: ModelSymbols = ModelSymbols.from_sbml(sbml)
        #: entity with an initial assignment -> the parameter whose
        #: assignment rule is the math of the initial assignment, see
        #: `initialize`
        self.initial_helpers: dict[str, str] = {}
        if self.symbols.initial_assignments:
            sbml, self.initial_helpers = _with_initial_helpers(sbml)

        # load model
        self.r: roadrunner.RoadRunner | None = roadrunner.RoadRunner(sbml)

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

    def initialize(self, assignments: Sequence["Assignment"]) -> None:
        """Initialize the model with values before its initialization.

        PEtab v2 sets the values of the parameter table and of the first
        conditions on the model before it is initialized, so an initial
        assignment follows a changed parameter, and a target which is set
        replaces its own initial assignment. roadrunner reinitializes the
        model when an initial value is set with `init(...)`, which costs a
        compilation of the model per value. The model is therefore
        initialized as it was loaded (`resetAll`), the values are set as the
        current values, and the initial assignments which read a changed
        entity are evaluated again, in their order, from the assignment rule
        of their helper parameter, which roadrunner evaluates on the current
        values. Nothing is left to restore: the next `resetAll` starts from
        the model as it was loaded.

        A compartment keeps the concentration of the species whose initial
        value is a concentration, as at the initialization of SBML.

        Args:
            assignments: the pre-initialization assignments of a plan, values
                only.

        Raises:
            ValueError: if an assignment is a formula.
        """
        r = self.r_loaded
        r.resetAll()
        symbols = self.symbols
        entities: set[str] = set()
        for a in assignments:
            if a.value is None:
                raise ValueError(
                    f"The pre-initialization value of '{a.target}' is the "
                    f"formula '{a.formula}', it must be a number."
                )
            entities.add(symbols.entity(a.target))
        for a in assignments:
            if a.kind is TargetKind.COMPARTMENT:
                self._set_compartment(a.target, float(a.value), entities)  # ty: ignore[invalid-argument-type]
        for a in assignments:
            if a.kind is not TargetKind.COMPARTMENT:
                r.setValue(a.target, a.value)

        dependencies = symbols.initial_assignment_dependencies or {}
        changed = set(entities)
        for entity in symbols.initial_assignment_order:
            if entity in entities or not (dependencies[entity] & changed):
                continue
            value = float(r.getValue(self.initial_helpers[entity]))
            if entity in symbols.compartments:
                self._set_compartment(entity, value, entities)
            elif entity in symbols.species and entity not in symbols.only_substance:
                r.setValue(f"[{entity}]", value)
            else:
                r.setValue(entity, value)
            changed.add(entity)

    def _set_compartment(self, compartment: str, value: float, kept: set[str]) -> None:
        """Set a compartment before the initialization.

        Args:
            compartment: the compartment.
            value: its size.
            kept: entities which are set themselves and are not rescaled.
        """
        r = self.r_loaded
        symbols = self.symbols
        concentrations = {
            s: r.getValue(f"[{s}]")
            for s, c in symbols.species_compartment.items()
            if c == compartment and s in symbols.initial_concentration and s not in kept
        }
        r.setValue(compartment, value)
        for species, concentration in concentrations.items():
            r.setValue(f"[{species}]", concentration)

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
