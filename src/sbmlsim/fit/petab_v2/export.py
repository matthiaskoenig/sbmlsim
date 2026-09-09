"""Write an `sbmlsim` optimization problem as a PEtab v2 problem.

The export walks an initialized `OptimizationProblem`, i.e., a problem whose
fit mappings are resolved, and builds the tables of PEtab v2 from it:

- every model of the fit is a model of the problem,
- the fit mappings which share a model and a simulation are one experiment, its
  periods are the timecourses of the `TimecourseSim` and their changes are the
  conditions,
- every fit mapping is one observable, its reference data are the measurements
  of that observable,
- every fit parameter is a parameter which is estimated.

What the tables do not hold goes into the `sbmlsim` extension of the problem,
see `sbmlsim.fit.petab_v2.extension`, and what is lost is reported by
`sbmlsim.fit.petab_v2.gaps`.
"""

import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import petab.v2 as petab_v2
from petab.models.sbml_model import SbmlModel  # ty: ignore[unresolved-import]
from petab.v2 import Problem as PetabProblem

from sbmlsim.fit.objects import MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, WeightingCurvesType
from sbmlsim.fit.petab_v2.extension import (
    EXTENSION_ID,
    SbmlsimExtension,
)
from sbmlsim.fit.petab_v2.gaps import Gap, GapKind, gaps_dict, gaps_of_problem
from sbmlsim.fit.petab_v2.symbols import condition_target, observable_formula
from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
from sbmlsim.units import Quantity

logger = logging.getLogger(__name__)

#: PEtab identifiers are the identifiers of SBML, i.e. letters, digits and `_`
_ID_FORBIDDEN = re.compile(r"[^0-9a-zA-Z_]")

#: name of the YAML file of the problem
YAML_FILE = "problem.yaml"

#: the placeholder of the noise of a measurement, i.e. the standard deviation
#: of its data. PEtab v2 declares the placeholders of an observable in
#: `noisePlaceholders`; the `noiseParameter${n}_${observableId}` names of v1
#: are gone
NOISE_PLACEHOLDER = "sd"

#: the file of every table of the problem, `to_files` skips a table without one
TABLE_FILES: dict[str, str] = {
    "condition_tables": "conditions.tsv",
    "experiment_tables": "experiments.tsv",
    "observable_tables": "observables.tsv",
    "measurement_tables": "measurements.tsv",
    "parameter_tables": "parameters.tsv",
}


def _table(petab_problem: PetabProblem, attribute: str) -> Any:
    """Get the table the export writes into, created if the problem has none.

    The `conditions`, `experiments`, `observables`, `measurements` and
    `parameters` of a `petab.v2.Problem` are read only views over its tables,
    i.e., appending to them is lost; the entries go into a table.

    Args:
        petab_problem: problem which is built.
        attribute: name of the tables, e.g. `condition_tables`.

    Returns:
        The last table of that kind.
    """
    tables = getattr(petab_problem, attribute)
    if not tables:
        table_class = {
            "condition_tables": petab_v2.ConditionTable,
            "experiment_tables": petab_v2.ExperimentTable,
            "observable_tables": petab_v2.ObservableTable,
            "measurement_tables": petab_v2.MeasurementTable,
            "parameter_tables": petab_v2.ParameterTable,
        }[attribute]
        # the tables have a default for their elements, which ty does not see
        tables.append(table_class(elements=[]))
    return tables[-1]


def _set_table_paths(petab_problem: PetabProblem) -> None:
    """Give every table which has entries the file it is written to.

    `Problem.to_files` writes the tables which have a `rel_path` and names the
    YAML after `config.filepath`, which a problem that was built rather than
    read does not have.
    """
    for attribute, filename in TABLE_FILES.items():
        for k, table in enumerate(getattr(petab_problem, attribute)):
            if not table.elements:
                continue
            if table.rel_path is None:
                stem, suffix = filename.rsplit(".", 1)
                name = filename if k == 0 else f"{stem}_{k}.{suffix}"
                table.rel_path = Path(name)
    if petab_problem.config is None:
        petab_problem.config = petab_v2.ProblemConfig(format_version="2.0.0")
    if not petab_problem.config.filepath:
        petab_problem.config.filepath = Path(YAML_FILE)


def petab_id(*parts: str) -> str:
    """Get a PEtab identifier from the parts of an `sbmlsim` key.

    Args:
        parts: parts of the identifier, joined with `__`.

    Returns:
        An identifier of letters, digits and underscores which does not start
        with a digit.
    """
    sid = "__".join(str(part) for part in parts if part)
    sid = _ID_FORBIDDEN.sub("_", sid)
    if sid and sid[0].isdigit():
        sid = f"_{sid}"
    return sid


def _magnitude(value: Any) -> float:
    """Get the number of a change, which is a quantity in model units."""
    if isinstance(value, Quantity):
        return float(value.magnitude)
    return float(value)


def _unit(value: Any) -> str | None:
    """Get the unit of a change, `None` if it is a plain number."""
    if isinstance(value, Quantity):
        return str(value.units)
    return None


class PetabExporter:
    """Write an optimization problem as a PEtab v2 problem."""

    def __init__(
        self,
        problem: OptimizationProblem,
        settings: FitSettings | None = None,
        kinds: set[MappingKind] | None = None,
        required_extension: bool = True,
    ):
        """Initialize the export of a problem.

        Args:
            problem: problem to export, it is initialized if it is not.
            settings: settings of the fit, required if the problem is not
                initialized.
            kinds: kinds of fit mappings to write, the training and the
                validation data by default. The outliers of a fit are not part
                of it, a tool which reads the problem without the extension
                would fit everything it finds.
            required_extension: mark the `sbmlsim` extension as required, which
                it is: the settings it carries are the objective of the fit, so
                a tool which does not know it has to reject the problem instead
                of fitting the same data differently. `False` writes a problem
                which other tools fit with the objective PEtab defines, see
                `sbmlsim.fit.petab_v2.extension.SbmlsimExtension`.

        Raises:
            ValueError: if the problem is not initialized and no settings are
                given.
        """
        if not problem.is_initialized:
            if settings is None:
                raise ValueError(
                    f"'{problem.opid}': the export needs the resolved mappings, "
                    f"give the `settings` of the fit or initialize the problem."
                )
            problem.initialize(settings)

        self.problem = problem
        self.required_extension = required_extension
        self.kinds: set[MappingKind] = (
            kinds
            if kinds is not None
            else {MappingKind.TRAINING, MappingKind.VALIDATION}
        )
        self.gaps: list[Gap] = gaps_of_problem(problem)

        #: indices of the fit mappings which are written
        self.indices: list[int] = [
            k
            for k in range(len(problem.mapping_keys))
            if problem.mapping_kinds[k] in self.kinds
        ]
        if not self.indices:
            raise ValueError(
                f"'{problem.opid}': no fit mapping is of the kinds "
                f"'{sorted(kind.value for kind in self.kinds)}', there is nothing "
                f"to export."
            )

        # the ids the export creates, filled by `to_problem`
        self.model_ids: dict[int, str] = {}
        #: the `libsbml.Model` per model id, for the math of the selections
        self.sbml_models: dict[str, Any] = {}
        #: the collection every experiment of the problem comes from
        self.experiment_collections: dict[str, int] = {}
        self.experiment_ids: dict[int, str] = {}
        self.observable_ids: dict[int, str] = {}

    def check(self) -> None:
        """Check that the problem can be written.

        Raises:
            ValueError: for a gap which has no representation in PEtab v2, i.e.
                a structural model change, an observable which is a python
                function or a mapping over something else than time.
        """
        unsupported = [gap for gap in self.gaps if gap.kind == GapKind.UNSUPPORTED]
        if unsupported:
            details = "\n".join(f"  - {gap.id}: {gap.detail}" for gap in unsupported)
            raise ValueError(
                f"'{self.problem.opid}': the problem uses features which PEtab v2 "
                f"cannot express:\n{details}"
            )

    def to_problem(self) -> PetabProblem:
        """Build the PEtab v2 problem.

        Returns:
            The problem with its models, conditions, experiments, observables,
            measurements and parameters, and the `sbmlsim` extension.

        Raises:
            ValueError: if the problem uses features PEtab v2 cannot express.
        """
        self.check()
        petab_problem = PetabProblem()

        self._add_models(petab_problem)
        self._add_experiments(petab_problem)
        self._add_observables_and_measurements(petab_problem)
        self._add_parameters(petab_problem)
        self._add_extension(petab_problem)
        return petab_problem

    # --- MODELS ---

    def _add_models(self, petab_problem: PetabProblem) -> None:
        """Add the models of the fit mappings, every SBML file once.

        Every simulation experiment of a fit has its own `RoadrunnerSBMLModel`,
        and several of them are usually the same file; the file is the model of
        the PEtab problem, so it is written once.
        """
        by_source: dict[Path, str] = {}
        for k in self.indices:
            model = self.problem.models[k]
            if id(model) in self.model_ids:
                continue
            source = model.source
            if source.path is None:
                raise ValueError(
                    f"'{self.problem.opid}': the model of the fit mapping "
                    f"'{self.problem.mapping_keys[k]}' has no file, only a model "
                    f"which is a file can be written as PEtab."
                )
            path = Path(source.path).resolve()
            sid = by_source.get(path)
            if sid is None:
                sid = petab_id(model.sid or path.stem)
                if sid in by_source.values():
                    sid = petab_id(sid, f"model{len(by_source)}")
                # `petab.v2.Model` is the abstract base, the SBML model is the
                # concrete class which reads a file
                model_file = SbmlModel.from_file(path, model_id=sid)
                model_file.rel_path = Path(path.name)
                petab_problem.models.append(model_file)
                self.sbml_models[sid] = model_file.sbml_model
                by_source[path] = sid
            self.model_ids[id(model)] = sid

    # --- EXPERIMENTS AND CONDITIONS ---

    def _add_experiments(self, petab_problem: PetabProblem) -> None:
        """Add the experiments of the fit mapping collections.

        A `FitMappingCollection` is the unit a fit is defined in, so it is the
        unit the problem is built from: the mappings of a collection which
        share a model and a simulation are one experiment of PEtab, its periods
        are the timecourses of the simulation and the changes of a timecourse
        are its condition. A collection whose mappings all share one simulation
        is exactly one experiment and carries its id; a collection over several
        simulations, e.g. the doses of a study, is one experiment per
        simulation, numbered after the collection.
        """
        problem = self.problem
        exported = set(self.indices)
        for collection_index, collection in enumerate(problem.mapping_collections):
            indices = [
                k for k in exported if problem.collection_indices[k] == collection_index
            ]
            if not indices:
                continue

            # the mappings of the collection which share a model and simulation
            groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            for k in sorted(indices):
                groups[(id(problem.models[k]), id(problem.simulations[k]))].append(k)

            for n, group in enumerate(groups.values()):
                k0 = group[0]
                simulation = problem.simulations[k0]
                if not isinstance(simulation, TimecourseSim):
                    raise ValueError(
                        f"'{problem.opid}': only a `TimecourseSim` is exported, "
                        f"'{simulation}' is a '{type(simulation).__name__}'."
                    )
                experiment_id = petab_id(collection.sid)
                if len(groups) > 1:
                    experiment_id = petab_id(collection.sid, f"sim{n}")
                periods = self._periods(
                    petab_problem,
                    experiment_id=experiment_id,
                    simulation=simulation,
                    sbml_model=self.sbml_models.get(
                        self.model_ids[id(problem.models[k0])]
                    ),
                )
                _table(petab_problem, "experiment_tables").experiments.append(
                    petab_v2.Experiment(id=experiment_id, periods=periods)
                )
                for k in group:
                    self.experiment_ids[k] = experiment_id
                    self.experiment_collections[experiment_id] = collection_index

    def _periods(
        self,
        petab_problem: PetabProblem,
        experiment_id: str,
        simulation: TimecourseSim,
        sbml_model: Any = None,
    ) -> list[petab_v2.ExperimentPeriod]:
        """Get the periods of an experiment and add their conditions.

        A leading timecourse which is discarded is the pre-equilibration of the
        experiment, i.e. a period at `time=-inf`; its duration is lost. The
        time of every other period is the time the timecourse starts at in the
        simulation, i.e. the sum of the durations before it.
        """
        periods: list[petab_v2.ExperimentPeriod] = []
        offset: float = simulation.time_offset
        for k, tc in enumerate(simulation.timecourses):
            if tc.discard and k > 0:
                raise ValueError(
                    f"'{self.problem.opid}': the timecourse '{k}' of "
                    f"'{experiment_id}' is discarded but is not the first, which "
                    f"is not the pre-equilibration of a PEtab experiment."
                )
            condition_ids: list[str] = []
            if tc.changes:
                condition_id = petab_id(experiment_id, f"tc{k}")
                _table(petab_problem, "condition_tables").conditions.append(
                    petab_v2.Condition(
                        id=condition_id,
                        changes=[
                            petab_v2.Change(
                                target_id=condition_target(target, sbml_model),
                                target_value=_magnitude(value),
                            )
                            for target, value in tc.changes.items()
                        ],
                    )
                )
                condition_ids.append(condition_id)

            time = float("-inf") if tc.discard else offset + tc.start
            periods.append(
                petab_v2.ExperimentPeriod(time=time, condition_ids=condition_ids)
            )
            offset += tc.end
        return periods

    # --- OBSERVABLES AND MEASUREMENTS ---

    def _add_observables_and_measurements(self, petab_problem: PetabProblem) -> None:
        """Add one observable per fit mapping with its reference data."""
        problem = self.problem
        for k in self.indices:
            observable_id = petab_id(
                problem.experiment_keys[k], problem.mapping_keys[k]
            )
            self.observable_ids[k] = observable_id

            errors = problem.y_errors[k]
            # the noise of a measurement is its standard deviation, which the
            # noise parameter of the measurement fills in. PEtab v2 declares the
            # placeholders of an observable, the `noiseParameter${n}_${id}`
            # names of v1 are gone
            placeholders = [NOISE_PLACEHOLDER] if errors is not None else []
            noise_formula = NOISE_PLACEHOLDER if errors is not None else "1.0"
            sbml_model = self.sbml_models.get(self.model_ids[id(problem.models[k])])
            _table(petab_problem, "observable_tables").observables.append(
                petab_v2.Observable(
                    id=observable_id,
                    name=f"{problem.experiment_keys[k]}.{problem.mapping_keys[k]}",
                    formula=observable_formula(problem.yid_observable[k], sbml_model),
                    noise_formula=noise_formula,
                    noise_placeholders=placeholders,
                )
            )

            experiment_id = self.experiment_ids.get(k)
            _measurements = _table(petab_problem, "measurement_tables").measurements
            times = np.asarray(problem.x_references[k], dtype=float)
            values = np.asarray(problem.y_references[k], dtype=float)
            for i in range(len(times)):
                _measurements.append(
                    petab_v2.Measurement(
                        observable_id=observable_id,
                        experiment_id=experiment_id,
                        time=float(times[i]),
                        measurement=float(values[i]),
                        observable_parameters=[],
                        noise_parameters=[float(errors[i])]
                        if errors is not None
                        else [],
                    )
                )

    # --- PARAMETERS ---

    def _add_parameters(self, petab_problem: PetabProblem) -> None:
        """Add the parameters which are estimated."""
        for parameter in self.problem.parameters:
            _table(petab_problem, "parameter_tables").parameters.append(
                petab_v2.Parameter(
                    id=parameter.pid,
                    lb=parameter.lower_bound,
                    ub=parameter.upper_bound,
                    nominal_value=parameter.start_value,
                    estimate=True,
                )
            )

    # --- EXTENSION ---

    def _add_extension(self, petab_problem: PetabProblem) -> None:
        """Add what the tables of PEtab do not hold."""
        problem = self.problem
        settings = problem.settings

        parameters = {
            parameter.pid: {
                "unit": parameter.unit,
                "start_value": parameter.start_value,
            }
            for parameter in problem.parameters
        }

        observables: dict[str, dict[str, Any]] = {}
        for k in self.indices:
            model = problem.models[k]
            observables[self.observable_ids[k]] = {
                "experiment": problem.experiment_keys[k],
                "mapping": problem.mapping_keys[k],
                "kind": problem.mapping_kinds[k].value,
                "collection": problem.mapping_collections[
                    problem.collection_indices[k]
                ].sid,
                "xid_observable": problem.xid_observable[k],
                "yid_observable": problem.yid_observable[k],
                "x_unit": str(model.uinfo[problem.xid_observable[k]]),
                "y_unit": str(model.uinfo[problem.yid_observable[k]]),
                "error_type": problem.y_errors_type[k],
                "weight_curve": float(problem.weights_curves[k]),
                "weight_mapping": self._weight_mapping(k),
                "model": self.model_ids[id(model)],
            }

        experiments: dict[str, dict[str, Any]] = {}
        for k, experiment_id in self.experiment_ids.items():
            if experiment_id in experiments:
                continue
            simulation = problem.simulations[k]
            collection_index = self.experiment_collections[experiment_id]
            experiments[experiment_id] = {
                "collection": problem.mapping_collections[collection_index].sid,
                "time_offset": simulation.time_offset,
                "reset": simulation.reset,
                "timecourses": [
                    self._timecourse_dict(tc) for tc in simulation.timecourses
                ],
            }

        models: dict[str, dict[str, Any]] = {}
        for k in self.indices:
            model = problem.models[k]
            model_id = self.model_ids[id(model)]
            if model_id in models:
                continue
            models[model_id] = {
                "sid": model.sid,
                "changes": {
                    target: _magnitude(value)
                    for target, value in (model.changes or {}).items()
                },
                "units": {
                    target: _unit(value)
                    for target, value in (model.changes or {}).items()
                },
            }

        collections: dict[str, dict[str, Any]] = {}
        for k in self.indices:
            collection = problem.mapping_collections[problem.collection_indices[k]]
            collections[collection.sid] = {
                "experiment_class": collection.experiment_class.__name__,
                "kind": collection.kind.value,
                "use_mapping_weights": collection.use_mapping_weights,
            }

        extension = SbmlsimExtension(
            required=self.required_extension,
            opid=problem.opid,
            settings=settings.to_dict() if settings is not None else {},
            parameters=parameters,
            observables=observables,
            experiments=experiments,
            collections=collections,
            models=models,
            gaps=gaps_dict(self.gaps),
        )
        if petab_problem.config is None:
            petab_problem.config = petab_v2.ProblemConfig(format_version="2.0.0")
        petab_problem.config.extensions[EXTENSION_ID] = extension

    def _weight_mapping(self, k: int) -> float:
        """Get the weight the user gave a fit mapping.

        The problem keeps the weight of a curve after the weighting was
        applied, i.e. `weight_curve = weight_mapping / len(data)` for
        `WeightingCurvesType.POINTS`. The weight the fit was defined with is
        what a reader has to give the mapping again, so the factor of the
        points is divided out.

        Args:
            k: index of the fit mapping.

        Returns:
            The weight of the mapping, `1.0` if the fit does not weight curves.
        """
        weight = float(self.problem.weights_curves[k])
        if WeightingCurvesType.POINTS in self.problem.weighting_curves:
            weight = weight * len(self.problem.y_references[k])
        return weight

    @staticmethod
    def _timecourse_dict(tc: Timecourse) -> dict[str, Any]:
        """Get the timecourse as a dictionary, with the units of its changes."""
        return {
            "start": tc.start,
            "end": tc.end,
            "steps": tc.steps,
            "discard": tc.discard,
            "changes": {
                target: _magnitude(value) for target, value in tc.changes.items()
            },
            "units": {target: _unit(value) for target, value in tc.changes.items()},
        }


def to_petab(
    problem: OptimizationProblem,
    output_dir: Path,
    settings: FitSettings | None = None,
    kinds: set[MappingKind] | None = None,
    required_extension: bool = True,
) -> Path:
    """Write an optimization problem as a PEtab v2 problem.

    Args:
        problem: problem to write, it is initialized if it is not.
        output_dir: directory the problem is written to, created if it does not
            exist.
        settings: settings of the fit, required if the problem is not
            initialized.
        kinds: kinds of fit mappings to write, the training and the validation
            data by default.
        required_extension: mark the `sbmlsim` extension as required, which it
            is: the settings it carries are the objective of the fit. `False`
            writes a problem other tools fit with the objective of PEtab.

    Returns:
        Path of the YAML file of the problem.

    Raises:
        ValueError: if the problem uses features PEtab v2 cannot express.
    """
    exporter = PetabExporter(
        problem,
        settings=settings,
        kinds=kinds,
        required_extension=required_extension,
    )
    petab_problem = exporter.to_problem()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # the models are copied, `to_files` writes them from their libsbml document
    for k in exporter.indices:
        model = problem.models[k]
        source_path = model.source.path
        if source_path is not None:
            target = output_dir / Path(source_path).name
            if not target.exists():
                shutil.copyfile(source_path, target)

    _set_table_paths(petab_problem)
    petab_problem.to_files(base_path=output_dir)
    yaml_file = output_dir / YAML_FILE
    logger.info("PEtab v2 problem written: %s", yaml_file.resolve().as_uri())
    return yaml_file
