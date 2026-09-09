"""Read a PEtab v2 problem into an `sbmlsim` optimization problem.

The tables of a PEtab problem are translated into a `SimulationExperiment`, in
the same way `sbmlsim.combine.sedml.parser` builds an experiment from a SED-ML
document: the models of the problem are the models of the experiment, its
experiments are the timecourse simulations, its observables are the fit
mappings and the measurements of an observable are its dataset.

A problem which was written by `sbmlsim.fit.petab_v2.export` carries the
`sbmlsim` extension, which holds what the tables do not: the units, the
settings of the fit, the kind of every mapping and the timecourses with their
output grid. The reader uses it when it is there, so that a round trip gives
the fit which was written, and falls back on the tables when it is not, which
is the case for a problem of another tool.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import petab.v2 as petab_v2
from petab.v2 import Problem as PetabProblem

from sbmlsim.data import DataSet
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit.objects import (
    FitData,
    FitMapping,
    FitMappingCollection,
    FitParameter,
    MappingKind,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings
from sbmlsim.fit.petab_v2.extension import SbmlsimExtension, extension_of
from sbmlsim.fit.petab_v2.symbols import selection_of_formula, split_selection
from sbmlsim.model import AbstractModel
from sbmlsim.simulation.timecourse import Timecourse, TimecourseSim
from sbmlsim.task import Task
from sbmlsim.units import UnitRegistry, UnitsInformation

logger = logging.getLogger(__name__)

#: unit of the time of a measurement if neither the `sbmlsim` extension nor
#: the model says what it is
DEFAULT_TIME_UNIT = "dimensionless"

#: unit of a measurement if neither the extension nor the model says
DEFAULT_VALUE_UNIT = "dimensionless"

#: steps of a timecourse which is built from the times of the measurements
DEFAULT_STEPS = 100

#: relative margin the simulation runs past the last measurement. The output
#: grid of a timecourse is a `linspace` which does not hit its end exactly, so
#: a measurement at the end of the simulation would be outside of the result
#: the residuals interpolate on
END_MARGIN = 1e-9

#: id of the experiment of the measurements which name none. PEtab reads an
#: empty `experimentId` as "use the model as is", i.e. a simulation from the
#: initial time of the model without conditions
DEFAULT_EXPERIMENT = "model"

#: prefix of the dataset of an observable. The keys of the datasets, the
#: simulations, the tasks and the fit mappings of a simulation experiment share
#: one namespace, and an observable is a mapping and the data behind it
DATASET_PREFIX = "data_"


def dataset_id(observable_id: str) -> str:
    """Get the id of the dataset which holds the measurements of an observable."""
    return f"{DATASET_PREFIX}{observable_id}"


class PetabReader:
    """Read a PEtab v2 problem as an `sbmlsim` simulation experiment and fit."""

    def __init__(
        self,
        petab_problem: PetabProblem,
        base_path: Path | None = None,
        name: str | None = None,
    ):
        """Initialize the reader.

        Args:
            petab_problem: problem to read.
            base_path: directory the files of the problem are relative to, the
                directory of its YAML file by default.
            name: name of the simulation experiment class which is created, the
                id of the problem by default.

        Raises:
            ValueError: if the problem has no model or no measurements.
        """
        self.petab_problem = petab_problem
        self.extension: SbmlsimExtension | None = extension_of(petab_problem.config)

        config_path = getattr(petab_problem.config, "base_path", None)
        self.base_path: Path = Path(base_path or config_path or ".")

        if not petab_problem.models:
            raise ValueError("The PEtab problem has no model.")
        if not petab_problem.measurements:
            raise ValueError("The PEtab problem has no measurements.")

        self.name: str = name or (
            self.extension.opid
            if self.extension and self.extension.opid
            else (getattr(petab_problem.config, "id", None) or "PetabExperiment")
        )
        # the units of the models: the registry has to know their definitions,
        # a `mmole_per_min` of an SBML model is not a unit pint knows, and a
        # problem without the `sbmlsim` extension has the units of the model,
        # which is what PEtab says its measurements are in
        self.uinfo: UnitsInformation | None = self._model_units()
        self.ureg: UnitRegistry = (
            self.uinfo.ureg if self.uinfo is not None else UnitRegistry()
        )

        # measurements by observable, in the order of their time
        self._measurements: dict[str, list[petab_v2.Measurement]] = {}
        for measurement in petab_problem.measurements:
            self._measurements.setdefault(measurement.observable_id, []).append(
                measurement
            )
        for measurements in self._measurements.values():
            measurements.sort(key=lambda m: m.time)

    @staticmethod
    def from_yaml(yaml_file: Path, name: str | None = None) -> "PetabReader":
        """Read the problem of a PEtab YAML file.

        Args:
            yaml_file: path of the YAML file of the problem.
            name: name of the simulation experiment class which is created.

        Returns:
            The reader of the problem.
        """
        yaml_file = Path(yaml_file)
        petab_problem = PetabProblem.from_yaml(yaml_file)
        return PetabReader(petab_problem, base_path=yaml_file.parent, name=name)

    # --- INFORMATION OF THE EXTENSION ---

    def _observable_info(self, observable_id: str) -> dict[str, Any]:
        """Get what the extension says about an observable, empty without one."""
        if self.extension is None:
            return {}
        return self.extension.observables.get(observable_id, {})

    def _experiment_info(self, experiment_id: str) -> dict[str, Any]:
        """Get what the extension says about an experiment, empty without one."""
        if self.extension is None:
            return {}
        return self.extension.experiments.get(experiment_id, {})

    @property
    def settings(self) -> FitSettings:
        """Get the settings of the fit, the defaults without the extension."""
        if self.extension is None or not self.extension.settings:
            logger.info(
                "The PEtab problem does not carry the 'sbmlsim' extension, the "
                "default `FitSettings` are used."
            )
            return FitSettings()
        return FitSettings.from_dict(self.extension.settings)

    # --- THE PARTS OF THE SIMULATION EXPERIMENT ---

    def _model_path(self, model: Any) -> Path:
        """Get the file of a model of the problem.

        Args:
            model: model of the PEtab problem.

        Returns:
            The path of the file, resolved against the directory of the problem.

        Raises:
            ValueError: if the model is not a file.
        """
        location = getattr(model, "rel_path", None)
        if location is None:
            raise ValueError(
                f"The model '{model.model_id}' of the PEtab problem has no "
                f"file, only a model which is a file can be read."
            )
        path = Path(location)
        return path if path.is_absolute() else self.base_path / path

    def _model_units(self) -> UnitsInformation | None:
        """Get the units of the models of the problem.

        The data of a fit is converted into the units of the model, and a model
        defines units of its own, e.g. the `mmole_per_min` of a reaction, so
        the registry has to be the one of the models. The units themselves are
        what a problem without the `sbmlsim` extension says its data is in:
        PEtab measures in the units of the model.

        Returns:
            The units of the models, `None` if none of them could be read.
        """
        uinfo: UnitsInformation | None = None
        for model in self.petab_problem.models:
            try:
                ureg = uinfo.ureg if uinfo is not None else None
                uinfo = UnitsInformation.from_sbml(self._model_path(model), ureg)
            except Exception as err:
                logger.warning(
                    "The units of the model '%s' could not be read: %s: %s",
                    getattr(model, "model_id", "?"),
                    type(err).__name__,
                    err,
                )
        return uinfo

    def _unit_of(self, sid: str, default: str | None) -> str | None:
        """Get the unit of an entity of the model.

        PEtab has no units: its measurements are in the units of the model and
        so are its parameters, so the model is what says what a number means.

        Args:
            sid: identifier of the entity, `time` for the time of the model.
            default: unit if the models do not say.

        Returns:
            The unit of the entity in the models, `default` if it has none. A
            model which declares an entity as dimensionless says so with an
            empty unit, which is `dimensionless` here so that it reads as a
            unit and not as a missing one.
        """
        if self.uinfo is None:
            return default
        try:
            unit = str(self.uinfo[sid])
        except (KeyError, TypeError):
            return default
        return unit if unit else DEFAULT_VALUE_UNIT

    def models(self) -> dict[str, AbstractModel]:
        """Get the models of the experiment, one per model of the problem."""
        return {
            model.model_id: AbstractModel(
                source=str(self._model_path(model)),
                sid=model.model_id,
                language_type=AbstractModel.LanguageType.SBML,
            )
            for model in self.petab_problem.models
        }

    @property
    def experiment_ids(self) -> list[str]:
        """Get the experiments which are simulated.

        A measurement which names no experiment is "use the model as is", i.e.
        `DEFAULT_EXPERIMENT`, which a problem of one condition does not have to
        declare (PEtab v2, measurement table).
        """
        ids = [experiment.id for experiment in self.petab_problem.experiments]
        if any(not m.experiment_id for m in self.petab_problem.measurements):
            ids.append(DEFAULT_EXPERIMENT)
        return ids

    def simulations(self) -> dict[str, TimecourseSim]:
        """Get the simulations, one per experiment of the problem.

        The timecourses of the extension are used if the problem carries it,
        which keeps the output grid and the pre-simulations of the fit which
        was written. Without it a timecourse is built from the periods of the
        experiment and the times of the measurements it holds.
        """
        experiments = {
            experiment.id: experiment for experiment in self.petab_problem.experiments
        }
        simulations: dict[str, TimecourseSim] = {}
        for experiment_id in self.experiment_ids:
            info = self._experiment_info(experiment_id)
            if info.get("timecourses"):
                simulations[experiment_id] = self._simulation_of_extension(info)
            elif experiment_id in experiments:
                simulations[experiment_id] = self._simulation_of_periods(
                    experiments[experiment_id]
                )
            else:
                # the model as it is, simulated over the measurements
                simulations[experiment_id] = TimecourseSim(
                    [
                        Timecourse(
                            start=0.0,
                            end=self._simulation_end(None),
                            steps=DEFAULT_STEPS,
                        )
                    ]
                )
        return simulations

    def _simulation_of_extension(self, info: dict[str, Any]) -> TimecourseSim:
        """Get the simulation the extension describes, i.e. the exact one."""
        timecourses = [
            Timecourse(
                start=tc["start"],
                end=tc["end"],
                steps=tc["steps"],
                changes=self._changes_with_units(
                    tc.get("changes", {}), tc.get("units", {})
                ),
                discard=tc.get("discard", False),
            )
            for tc in info["timecourses"]
        ]
        return TimecourseSim(
            timecourses,
            reset=info.get("reset", True),
            time_offset=info.get("time_offset", 0.0),
        )

    def _simulation_of_periods(self, experiment: petab_v2.Experiment) -> TimecourseSim:
        """Build a simulation from the periods of a PEtab experiment.

        The time of a period is the time of the simulation the condition
        becomes active at, so a period lasts until the next one starts and the
        last one until the last measurement of the experiment was taken. The
        first of them is where the simulation starts, which is the
        `time_offset` of the `TimecourseSim`: a multiple dosing experiment
        whose data is reported from the last dose starts at a negative time.

        A period at `time=-inf` is the pre-equilibration of the experiment,
        which becomes a timecourse whose result is discarded.
        """
        end = self._simulation_end(experiment.id)
        conditions = {
            condition.id: condition for condition in self.petab_problem.conditions
        }

        timecourses: list[Timecourse] = []
        periods = sorted(experiment.periods, key=lambda p: p.time)
        for k, period in enumerate(periods):
            changes: dict[str, Any] = {}
            for condition_id in period.condition_ids:
                condition = conditions.get(condition_id)
                if condition is None:
                    raise ValueError(
                        f"The experiment '{experiment.id}' uses the condition "
                        f"'{condition_id}', which the problem does not define."
                    )
                for change in condition.changes:
                    changes[change.target_id] = _to_float(change.target_value)

            if np.isinf(period.time):
                # pre-equilibration, the duration is not part of the problem
                timecourses.append(
                    Timecourse(
                        start=0.0,
                        end=end,
                        steps=DEFAULT_STEPS,
                        changes=changes,
                        discard=True,
                    )
                )
                continue

            start = float(period.time)
            stop = float(periods[k + 1].time) if k + 1 < len(periods) else end
            timecourses.append(
                Timecourse(
                    start=0.0,
                    end=max(stop - start, 0.0),
                    steps=DEFAULT_STEPS,
                    changes=changes,
                )
            )

        # the simulation starts where the first period which is not the
        # pre-equilibration starts
        finite = [period.time for period in periods if not np.isinf(period.time)]
        return TimecourseSim(
            timecourses, time_offset=float(finite[0]) if finite else 0.0
        )

    def _simulation_end(self, experiment_id: str | None) -> float:
        """Get the time the simulation of an experiment runs to.

        The simulation has to cover the measurements, so it ends just past the
        last of them, see `END_MARGIN`.

        Args:
            experiment_id: id of the experiment, `None` for the measurements
                which name no experiment.

        Returns:
            The end of the simulation.
        """
        end = self._last_measurement_time(experiment_id)
        return end * (1.0 + END_MARGIN) if end > 0.0 else end

    def _last_measurement_time(self, experiment_id: str | None) -> float:
        """Get the last time a measurement of an experiment was taken.

        Args:
            experiment_id: id of the experiment, `None` for the measurements
                which name no experiment.
        """
        times = [
            measurement.time
            for measurement in self.petab_problem.measurements
            if (measurement.experiment_id or None) == experiment_id
            and np.isfinite(measurement.time)
        ]
        return float(max(times)) if times else 0.0

    def _changes_with_units(
        self, changes: dict[str, float], units: dict[str, str | None]
    ) -> dict[str, Any]:
        """Get the changes as quantities, with the units of the extension."""
        result: dict[str, Any] = {}
        for target, value in changes.items():
            unit = units.get(target)
            result[target] = self.ureg.Quantity(value, unit) if unit else float(value)
        return result

    def tasks(self) -> dict[str, Task]:
        """Get the tasks, one per experiment of the problem."""
        model_ids = [model.model_id for model in self.petab_problem.models]
        return {
            f"task_{experiment_id}": Task(
                model=self._model_of_experiment(experiment_id, model_ids),
                simulation=experiment_id,
            )
            for experiment_id in self.experiment_ids
        }

    def _model_of_experiment(self, experiment_id: str, model_ids: list[str]) -> str:
        """Get the model an experiment is simulated with.

        A PEtab problem with one model simulates everything with it. With
        several models the extension says which one, and a measurement may name
        its model as well.
        """
        if self.extension is not None:
            for info in self.extension.observables.values():
                if info.get("model") and self._experiment_of_observable(info) == (
                    experiment_id
                ):
                    return str(info["model"])
        for measurement in self.petab_problem.measurements:
            if measurement.experiment_id == experiment_id and measurement.model_id:
                return str(measurement.model_id)
        return model_ids[0]

    def _experiment_of_observable(self, info: dict[str, Any]) -> str | None:
        """Get the PEtab experiment of an observable of the extension."""
        for observable_id, observable_info in (
            self.extension.observables.items() if self.extension else []
        ):
            if observable_info is info:
                for measurement in self.petab_problem.measurements:
                    if measurement.observable_id == observable_id:
                        return measurement.experiment_id
        return None

    def datasets(self) -> dict[str, DataSet]:
        """Get the datasets, one per observable, from its measurements."""
        datasets: dict[str, DataSet] = {}
        for observable_id, measurements in self._measurements.items():
            info = self._observable_info(observable_id)
            # without the extension the data is in the units of the model,
            # which is what PEtab measures in
            yid = info.get("yid_observable") or self._selection_of(observable_id)
            time_unit = info.get("x_unit") or self._unit_of("time", DEFAULT_TIME_UNIT)
            value_unit = info.get("y_unit") or self._unit_of(
                split_selection(yid)[0], DEFAULT_VALUE_UNIT
            )

            data: dict[str, Any] = {
                "time": [measurement.time for measurement in measurements],
                "time_unit": time_unit,
                "value": [measurement.measurement for measurement in measurements],
                "value_unit": value_unit,
            }
            errors = [
                _to_float(measurement.noise_parameters[0])
                if measurement.noise_parameters
                and _is_number(measurement.noise_parameters[0])
                else np.nan
                for measurement in measurements
            ]
            if not all(np.isnan(error) for error in errors):
                data["value_sd"] = errors
                data["value_sd_unit"] = value_unit

            datasets[dataset_id(observable_id)] = DataSet.from_df(
                pd.DataFrame(data), ureg=self.ureg
            )
        return datasets

    def fit_mappings(self, experiment: SimulationExperiment) -> dict[str, FitMapping]:
        """Get the fit mappings, one per observable of the problem.

        Args:
            experiment: experiment the mappings belong to.

        Returns:
            The mappings by the id of their observable.
        """
        mappings: dict[str, FitMapping] = {}
        for observable_id, measurements in self._measurements.items():
            info = self._observable_info(observable_id)
            experiment_id = measurements[0].experiment_id or DEFAULT_EXPERIMENT
            task_id = f"task_{experiment_id}"

            observable_yid = info.get("yid_observable") or self._selection_of(
                observable_id
            )
            observable_xid = info.get("xid_observable") or "time"

            reference = FitData(
                experiment,
                xid="time",
                yid="value",
                yid_sd="value_sd"
                if "value_sd" in experiment._datasets[dataset_id(observable_id)].columns
                else None,
                dataset=dataset_id(observable_id),
            )
            observable = FitData(
                experiment,
                xid=observable_xid,
                yid=observable_yid,
                task=task_id,
            )
            mappings[observable_id] = FitMapping(
                experiment,
                reference=reference,
                observable=observable,
                weight=info.get("weight_mapping", 1.0),
            )
        return mappings

    def _selection_of(self, observable_id: str) -> str:
        """Get the selection of roadrunner which observes an observable.

        Without the `sbmlsim` extension the formula of the observable is what
        the fit observes, i.e. the problem of another tool. The math of PEtab
        and the selections of roadrunner do not agree on the amount and the
        concentration of a species, see `sbmlsim.fit.petab_v2.symbols`.
        """
        for observable in self.petab_problem.observables:
            if observable.id == observable_id:
                return selection_of_formula(str(observable.formula), self._sbml_model())
        raise ValueError(f"The problem has no observable '{observable_id}'.")

    def _in_model(self, sid: str) -> bool:
        """Check whether an identifier is an entity of one of the models."""
        return any(model.has_entity_with_id(sid) for model in self.petab_problem.models)

    def _sbml_model(self) -> Any:
        """Get the `libsbml.Model` of the problem, `None` if it has several."""
        models = self.petab_problem.models
        if len(models) != 1:
            return None
        return getattr(models[0], "sbml_model", None)

    # --- THE SIMULATION EXPERIMENT AND THE OPTIMIZATION PROBLEM ---

    def experiment_class(self) -> type[SimulationExperiment]:
        """Create the simulation experiment class of the problem.

        Returns:
            A `SimulationExperiment` subclass which holds the models, the
            simulations, the tasks, the datasets and the fit mappings of the
            PEtab problem.
        """
        reader = self

        def f_models(obj: SimulationExperiment) -> dict[str, AbstractModel]:
            return reader.models()

        def f_datasets(obj: SimulationExperiment) -> dict[str, DataSet]:
            return reader.datasets()

        def f_simulations(obj: SimulationExperiment) -> dict[str, TimecourseSim]:
            return reader.simulations()

        def f_tasks(obj: SimulationExperiment) -> dict[str, Task]:
            return reader.tasks()

        def f_fit_mappings(obj: SimulationExperiment) -> dict[str, FitMapping]:
            return reader.fit_mappings(obj)

        return type(
            _class_name(self.name),
            (SimulationExperiment,),
            {
                "models": f_models,
                "datasets": f_datasets,
                "simulations": f_simulations,
                "tasks": f_tasks,
                "fit_mappings": f_fit_mappings,
            },
        )

    def fit_parameters(self) -> list[FitParameter]:
        """Get the parameters which are estimated.

        Returns:
            The parameters with their bounds, their start value and, if the
            problem carries the extension, their unit.
        """
        parameters: list[FitParameter] = []
        for parameter in self.petab_problem.parameters:
            if not parameter.estimate:
                continue
            if not self._in_model(parameter.id):
                # a parameter of the noise or of an observable, which PEtab
                # estimates with the parameters of the model. The objective of
                # `sbmlsim` has no such parameter, it weights the data instead
                logger.warning(
                    "The parameter '%s' is estimated by the problem but is not "
                    "an entity of a model, i.e. it is a parameter of the noise "
                    "or of an observable. `sbmlsim` fits the parameters of a "
                    "model and weights the data, so it is not fitted.",
                    parameter.id,
                )
                continue
            info = (
                self.extension.parameters.get(parameter.id, {})
                if self.extension
                else {}
            )
            nominal = parameter.nominal_value
            start_value = info.get("start_value")
            if start_value is None and isinstance(nominal, int | float):
                start_value = float(nominal)
            parameters.append(
                FitParameter(
                    pid=parameter.id,
                    start_value=start_value,
                    lower_bound=(
                        float(parameter.lb) if parameter.lb is not None else -np.inf
                    ),
                    upper_bound=(
                        float(parameter.ub) if parameter.ub is not None else np.inf
                    ),
                    # PEtab has no units, a parameter is in the unit the
                    # model gives it
                    unit=info.get("unit") or self._unit_of(parameter.id, None),
                )
            )
        return parameters

    def mapping_collections(
        self, experiment_class: type[SimulationExperiment]
    ) -> list[FitMappingCollection]:
        """Get the fit mapping collections of the problem, one per experiment.

        An experiment of PEtab is a simulation with its conditions, and the
        observables which are measured in it are the mappings which belong
        together, i.e. one `FitMappingCollection` per experiment of the problem.
        The collection carries the id of the experiment and, with the `sbmlsim`
        extension, the kind the mappings had; without it everything is training
        data, which is what a PEtab problem means.

        Args:
            experiment_class: the simulation experiment the mappings belong to.

        Returns:
            The collections, in the order of the experiments of the problem.
        """
        by_experiment: dict[str, list[str]] = {}
        for observable_id, measurements in self._measurements.items():
            experiment_id = measurements[0].experiment_id or DEFAULT_EXPERIMENT
            by_experiment.setdefault(experiment_id, []).append(observable_id)

        collections: list[FitMappingCollection] = []
        for experiment_id, mappings in by_experiment.items():
            kinds = {
                MappingKind(
                    self._observable_info(observable_id).get(
                        "kind", MappingKind.TRAINING.value
                    )
                )
                for observable_id in mappings
            }
            if len(kinds) > 1:
                # an experiment whose observables are used differently is one
                # collection per kind, the kind belongs to the selection
                for kind in sorted(kinds):
                    selected = [
                        observable_id
                        for observable_id in mappings
                        if MappingKind(
                            self._observable_info(observable_id).get(
                                "kind", MappingKind.TRAINING.value
                            )
                        )
                        is kind
                    ]
                    collections.append(
                        self._collection(
                            experiment_class,
                            sid=f"{experiment_id}_{kind.value}",
                            mappings=selected,
                            kind=kind,
                        )
                    )
                continue
            collections.append(
                self._collection(
                    experiment_class,
                    sid=experiment_id,
                    mappings=mappings,
                    kind=next(iter(kinds)),
                )
            )
        return collections

    @staticmethod
    def _collection(
        experiment_class: type[SimulationExperiment],
        sid: str,
        mappings: list[str],
        kind: MappingKind,
    ) -> FitMappingCollection:
        """Create one collection of the mappings of an experiment."""
        return FitMappingCollection(
            experiment=experiment_class,
            sid=sid,
            mappings=mappings,
            kind=kind,
            # the weights of the mappings are the weights the fit was defined
            # with, see `PetabExporter._weight_mapping`
            use_mapping_weights=True,
        )

    def to_optimization_problem(self, opid: str | None = None) -> OptimizationProblem:
        """Build the optimization problem of the PEtab problem.

        Args:
            opid: id of the problem, the id of the PEtab problem by default.

        Returns:
            The problem, which is not initialized: `initialize(settings)` with
            the `settings` of the reader resolves its data.
        """
        experiment_class = self.experiment_class()
        return OptimizationProblem(
            opid=opid or self.name,
            mapping_collections=self.mapping_collections(experiment_class),
            fit_parameters=self.fit_parameters(),
            base_path=self.base_path,
            data_path=self.base_path,
        )


def _class_name(name: str) -> str:
    """Get a python class name from the name of a problem."""
    parts = [part for part in name.replace("-", "_").split("_") if part]
    class_name = "".join(part[:1].upper() + part[1:] for part in parts)
    if not class_name or class_name[0].isdigit():
        class_name = f"Petab{class_name}"
    return class_name


def _to_float(value: Any) -> float:
    """Get the number of a value of PEtab, which is a sympy expression."""
    return float(value)


def _is_number(value: Any) -> bool:
    """Check whether a noise parameter is a number and not a parameter id."""
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def from_petab(
    yaml_file: Path, opid: str | None = None
) -> tuple[OptimizationProblem, FitSettings]:
    """Read a PEtab v2 problem as an optimization problem.

    Args:
        yaml_file: path of the YAML file of the PEtab problem.
        opid: id of the optimization problem.

    Returns:
        The problem and the settings of the fit, i.e. what `run_optimization`
        needs. The settings are the defaults if the problem does not carry the
        `sbmlsim` extension.
    """
    reader = PetabReader.from_yaml(Path(yaml_file))
    return reader.to_optimization_problem(opid=opid), reader.settings
