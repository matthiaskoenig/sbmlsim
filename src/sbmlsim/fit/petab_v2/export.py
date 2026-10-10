"""Write an `sbmlsim` optimization problem as a PEtab v2 problem.

The export walks an initialized `OptimizationProblem`, i.e., a problem whose
fit mappings are resolved, and builds the tables of PEtab v2 from it:

- every model of the fit is a model of the problem, written as the model the
  problem was defined with when the fit simulates a derived one (compiled
  networks, see `sbmlsim.model.provenance`),
- the fit mappings which share a model and a simulation are one experiment, its
  periods are the start of the `Simulation` and the times of its changes, and
  their changes are the conditions,
- every fit mapping is one observable, named after the mapping; mappings which
  observe one thing in several experiments are one observable, its reference
  data are the measurements of that observable,
- every fit parameter is a parameter which is estimated.

What the tables do not hold goes into the `sbmlsim` extension of the problem,
see `sbmlsim.fit.petab_v2.extension`, and what is lost is reported by
`sbmlsim.fit.petab_v2.gaps`.
"""

import dataclasses
import logging
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import petab.v2 as petab_v2
import sympy as sp
from petab.models.sbml_model import SbmlModel  # ty: ignore[unresolved-import]
from petab.v2 import Problem as PetabProblem
from petab.v2.math import petab_math_str, sympify_petab

from sbmlsim.fit.objects import (
    EVALUATED_KINDS,
    MappingKind,
    NoiseModel,
    NoiseParameter,
)
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import FitSettings, WeightingCurvesType
from sbmlsim.fit.parameter_mapping import has_renamed_targets
from sbmlsim.fit.parameters import ParameterSet
from sbmlsim.fit.petab_v2.extension import (
    EXTENSION_ID,
    SCIML_EXTENSION_ID,
    SbmlsimExtension,
)
from sbmlsim.fit.petab_v2.gaps import (
    Gap,
    GapKind,
    gap_mappings,
    gaps_dict,
    gaps_of_problem,
)
from sbmlsim.fit.petab_v2.likelihood import noise_model_of
from sbmlsim.fit.petab_v2.symbols import (
    condition_target,
    formula_of_selections,
    observable_formula,
)
from sbmlsim.model.provenance import derivation_of, strip_derivation
from sbmlsim.simulator.plan import Assignment
from sbmlsim.units import Quantity

logger = logging.getLogger(__name__)

#: PEtab identifiers are the identifiers of SBML, i.e. letters, digits and `_`
_ID_FORBIDDEN = re.compile(r"[^0-9a-zA-Z_]")

#: name of the YAML file of the problem
YAML_FILE = "problem.yaml"

#: the file of every table of the problem, `to_files` skips a table without one
TABLE_FILES: dict[str, str] = {
    "condition_tables": "conditions.tsv",
    "experiment_tables": "experiments.tsv",
    "observable_tables": "observables.tsv",
    "measurement_tables": "measurements.tsv",
    "parameter_tables": "parameters.tsv",
    "mapping_tables": "mapping.tsv",
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
            "mapping_tables": petab_v2.MappingTable,
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


def period_condition_id(experiment_id: str, k: int) -> str:
    """Get the id of the condition of a period of an experiment.

    Args:
        experiment_id: id of the experiment.
        k: index of the period of the experiment.

    Returns:
        The id, e.g. `e1__tc0`, which the arrays of the inputs of the
        networks are keyed by for the first period.
    """
    return petab_id(experiment_id, f"tc{k}")


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
    """Write an optimization problem as a PEtab v2 problem.

    A problem with hybridizations is written with its networks as PEtab
    SciML, see `sbmlsim.fit.petab_v2.sciml_export`, which needs the extra
    `sciml`.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        settings: FitSettings | None = None,
        kinds: set[MappingKind] | None = None,
        required_extension: bool = True,
        parameter_set: ParameterSet | None = None,
    ):
        """Initialize the export of a problem.

        Args:
            problem: problem to export, it is initialized if it is not.
            settings: settings of the fit, required if the problem is not
                initialized.
            kinds: kinds of fit mappings to write, everything the problem
                resolved by default, i.e. the training data, the validation
                data and the outliers, so that a round trip keeps the fit. The
                data the model does not describe is not part of the problem.
                Which of them a fit uses is in the extension: a tool which
                reads the problem without it would fit everything it finds.
            required_extension: mark the `sbmlsim` extension as required, which
                it is: the settings it carries are the objective of the fit, so
                a tool which does not know it has to reject the problem instead
                of fitting the same data differently. `False` writes a problem
                which other tools fit with the objective PEtab defines, see
                `sbmlsim.fit.petab_v2.extension.SbmlsimExtension`.
            parameter_set: values of the parameters of the fit to write, e.g.
                the result of a fit, which must have every parameter of the
                problem. They are the nominal values of the parameter table,
                the start values of the `sbmlsim` block and the values of the
                arrays of the networks, i.e. the exported problem starts from
                the set: a problem which is read has one start value per
                parameter and the arrays of a network cannot carry a second
                one, so a set which was written next to the start values would
                read back as the elements of the set and the parameters of the
                start. The problem is not changed. An element of a network
                which the problem freezes keeps its value.

        Raises:
            ValueError: if the problem is not initialized and no settings are
                given, or if the parameter set has a value for an element of a
                network which is no parameter of the fit.
            KeyError: if the parameter set lacks a parameter of the problem.
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
        #: the value of every parameter of the fit which is written, the start
        #: value of the parameter if there is no parameter set
        self.values: dict[str, float | None] = {
            p.pid: p.start_value for p in problem.parameters
        }
        if parameter_set is not None:
            self.values = dict(
                zip(problem.pids, parameter_set.x(problem.pids).tolist(), strict=True)
            )
        self.kinds: set[MappingKind] = (
            kinds if kinds is not None else set(EVALUATED_KINDS)
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
        #: the key of every fit mapping in the block `observables` of the
        #: extension: the key the reader gives the mapping when it reads the
        #: problem again, see `_name_observables`
        self.info_keys: dict[int, str] = {}

        #: index into `problem.mapping_groups` of the simulation group a fit
        #: mapping belongs to, i.e. the group `ParameterMapping.indices_for`
        #: resolves for it. `_group_mappings` already partitions every fit
        #: mapping this way; this is that partition read backwards, not a
        #: second derivation of it, so an experiment finds its group by a
        #: lookup rather than by recomputing the `(model, simulation)` key.
        self.group_indices: dict[int, int] = {
            k: g for g, group in enumerate(problem.mapping_groups) for k in group
        }
        #: the formula of the observable of every fit mapping, see
        #: `_observable_formula`
        self._formulas: dict[int, str] = {}
        #: the networks of the problem, `None` for a problem without them
        self.sciml: Any = None
        if problem.hybridizations:
            # the import needs the extra `sciml`
            from sbmlsim.fit.petab_v2.sciml_export import SciMLExporter

            self.sciml = SciMLExporter(
                problem,
                problem.hybridizations,
                simulation_ids=self._simulation_ids(),
                parameter_set=parameter_set,
            )

    def _experiment_groups(self) -> Iterator[tuple[int, str, list[int]]]:
        """Get the fit mappings which are one experiment of PEtab.

        A `FitMappingCollection` is the unit a fit is defined in, so it is the
        unit the problem is built from: the mappings of a collection which
        share a model and a simulation are one experiment. A collection whose
        mappings all share one simulation is exactly one experiment and
        carries its id; a collection over several simulations, e.g. the doses
        of a study, is one experiment per simulation, numbered after the
        collection.

        Yields:
            The index of the collection, the id of the experiment and the
            indices of its fit mappings, in the order of the collections.
        """
        problem = self.problem
        exported = sorted(self.indices)
        for collection_index, collection in enumerate(problem.mapping_collections):
            groups: dict[tuple[int, int], list[int]] = defaultdict(list)
            for k in exported:
                if problem.collection_indices[k] == collection_index:
                    key = (id(problem.models[k]), id(problem.simulations[k]))
                    groups[key].append(k)
            for n, group in enumerate(groups.values()):
                experiment_id = (
                    petab_id(collection.sid)
                    if len(groups) == 1
                    else petab_id(collection.sid, f"sim{n}")
                )
                yield collection_index, experiment_id, group

    def _simulation_ids(self) -> dict[str, list[str]]:
        """Get the ids of the PEtab experiments of every simulation of the problem.

        The ids are the ones `_add_experiments` gives the experiments, computed
        ahead of it: the networks key the arrays of their inputs by them. The
        key of a simulation is unique in its simulation experiment only, and
        a simulation whose fit mappings are in several collections is several
        experiments: the networks read their inputs by the key of the
        simulation, so every one of these experiments has the same inputs.

        Returns:
            id of the simulation of a fit mapping -> ids of its experiments.
        """
        ids: dict[str, list[str]] = defaultdict(list)
        for _, experiment_id, group in self._experiment_groups():
            key = self.problem.simulation_keys[group[0]]
            if experiment_id not in ids[key]:
                ids[key].append(experiment_id)
        return dict(ids)

    def check(self) -> None:
        """Check that the problem can be written.

        Raises:
            ValueError: for a gap which has no representation in PEtab v2, i.e.
                an observable which is a python function or a mapping over
                something else than time; or for a selector without its own
                id, see `_check_unnamed_versions`.
        """
        unsupported = [gap for gap in self.gaps if gap.kind == GapKind.UNSUPPORTED]
        if unsupported:
            mappings = gap_mappings(self.problem)
            lines = []
            for gap in unsupported:
                lines.append(f"  - {gap.id}: {gap.detail}")
                if gap.id in mappings:
                    lines.append(f"    mappings: {', '.join(mappings[gap.id])}")
            details = "\n".join(lines)
            raise ValueError(
                f"'{self.problem.opid}': the problem uses features which PEtab v2 "
                f"cannot express:\n{details}"
            )
        self._check_unnamed_versions()

    def _check_unnamed_versions(self) -> None:
        """Refuse a selector whose parameter writes its own id.

        `FitParameter(target=None, mappings=...)` is legal and estimates the
        entity from part of the data while leaving the model's value where
        the selector does not match: `ParameterMapping` honours it, but PEtab
        has no id for "this entity" distinct from the entity itself, so a
        condition cannot say "assign this estimated parameter" without
        naming a version that is not the target. Writing nothing for such a
        parameter would estimate it everywhere instead, silently, which is
        why this is refused rather than degraded.

        Raises:
            ValueError: if a versioned parameter writes its own id.
        """
        unnamed = [
            p
            for p in self.problem.parameters
            if p.is_versioned and p.target_id == p.pid
        ]
        if unnamed:
            details = "\n".join(f"  - {p.pid}" for p in unnamed)
            raise ValueError(
                f"'{self.problem.opid}': the following parameters have a "
                f"selector (`mappings`) but no `target` of their own, so "
                f"PEtab has no id to write the condition with -- it would "
                f"estimate the entity everywhere instead of only where the "
                f"selector matches. Give the version its own `pid` and set "
                f"`target` to the entity it writes:\n{details}"
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
        if self.sciml is not None:
            self.sciml.place_inputs(
                {model.model_id: model for model in petab_problem.models},
                self._model_changes(),
            )
        self._add_experiments(petab_problem)
        self._add_observables_and_measurements(petab_problem)
        self._add_parameters(petab_problem)
        self._check_condition_targets(petab_problem)
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
        #: the file of every model which is written -> its content
        files: dict[Path, str] = {}
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
                # the id the experiment gives the model, which is what the
                # hybridizations of the problem name
                sid = petab_id(self.problem.model_keys[k])
                if sid in by_source.values():
                    sid = petab_id(sid, f"model{len(by_source)}")
                # `petab.v2.Model` is the abstract base, the SBML model is the
                # concrete class which reads a file. A model which is derived
                # from the model of the problem, i.e. carries compiled
                # networks, is written as its source
                document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(path))
                name = path.name
                sbml_model = document.getModel()
                if sbml_model is not None and derivation_of(sbml_model) is not None:
                    document, derivation = strip_derivation(path)
                    name = derivation.source
                model_file = SbmlModel(sbml_document=document, model_id=sid)
                model_file.rel_path = self._model_file(
                    files, Path(name), sid, libsbml.writeSBMLToString(document)
                )
                petab_problem.models.append(model_file)
                self.sbml_models[sid] = model_file.sbml_model
                by_source[path] = sid
            self.model_ids[id(model)] = sid

    def _model_file(
        self, files: dict[Path, str], name: Path, sid: str, content: str
    ) -> Path:
        """Get the file a model is written to.

        A model is written to the file of its source. Two models whose sources
        have the same name, e.g. two models of different directories or two
        models derived from one source, are different files: the second is
        named after its model id.

        Args:
            files: the files which are written -> their content, the file of
                the model is added.
            name: name of the file of the source of the model.
            sid: id of the model.
            content: the SBML the model is written as.

        Returns:
            The path of the file, relative to the problem.

        Raises:
            ValueError: if the file named after the model id is taken, too.
        """
        for path in (name, Path(f"{sid}.xml")):
            if files.setdefault(path, content) == content:
                return path
        raise ValueError(
            f"'{self.problem.opid}': the model '{sid}' can be written neither as "
            f"'{name}' nor as '{sid}.xml', other models of the problem are "
            f"written to these files."
        )

    def _model_changes(self) -> dict[str, dict[str, float]]:
        """Get the changes of every model which is written, id -> value.

        Returns:
            id of the model -> the changes the fit applies to it, in the
            units of the model.
        """
        changes: dict[str, dict[str, float]] = {}
        for k in self.indices:
            model = self.problem.models[k]
            changes.setdefault(
                self.model_ids[id(model)],
                {
                    target: _magnitude(value)
                    for target, value in (model.changes or {}).items()
                },
            )
        return changes

    # --- EXPERIMENTS AND CONDITIONS ---

    def _add_experiments(self, petab_problem: PetabProblem) -> None:
        """Add the experiments of the fit mapping collections.

        The experiments are the ones of `_experiment_groups`: the periods of
        an experiment are the times of the changes of its simulation, see
        `_periods`.
        """
        problem = self.problem
        for collection_index, experiment_id, group in self._experiment_groups():
            k0 = group[0]
            periods = self._periods(
                petab_problem,
                experiment_id=experiment_id,
                group_index=self.group_indices[k0],
                simulation_key=problem.simulation_keys[k0],
                sbml_model=self.sbml_models.get(self.model_ids[id(problem.models[k0])]),
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
        group_index: int,
        simulation_key: str,
        sbml_model: Any = None,
    ) -> list[petab_v2.ExperimentPeriod]:
        """Get the periods of an experiment and add their conditions.

        The periods are the ones of the compiled simulation of the group, in
        the units of the model, which are the units of PEtab: the first period
        is the start of the simulation, its condition the changes before the
        initialization; every other time of a change is a period. A
        `SteadyState` is the period at `time=-inf`, whose condition holds the
        changes before the initialization, and the changes at the start are
        then the condition of the first period (PEtab v2, initialization). The
        times are shifted by the time shift of the simulation, as the
        measurements are.

        Args:
            petab_problem: problem which is built.
            experiment_id: id of the experiment the periods belong to.
            group_index: index into `problem.mapping_groups` of the simulation
                group this experiment was built from, i.e. `self.group_indices`
                of the fit mapping the experiment groups around. It is what
                `ParameterMapping.indices_for` resolves the binding for.
            simulation_key: id of the simulation in its experiment, which is
                the condition of the inputs of the networks.
            sbml_model: `libsbml.Model` of the problem, for the math of the
                selections.

        Returns:
            The periods of the experiment, with their conditions added to the
            problem.
        """
        # a versioned parameter is written as a condition: PEtab assigns the
        # entity of the model the value of the estimated parameter, which is
        # how one entity is estimated separately for parts of the data. The
        # binding of a target to the parameter which writes it in this
        # simulation group is `ParameterMapping`'s, read here rather than
        # rederived from the parameters' selectors.
        mapping = self.problem.parameter_mapping
        version_changes: list[petab_v2.Change] = []
        if mapping is not None:
            for target, index in sorted(mapping.indices_for(group_index).items()):
                parameter = self.problem.parameters[index]
                if parameter.is_external:
                    # not an entity of the model, the networks read it
                    continue
                if has_renamed_targets([parameter]):
                    version_changes.append(
                        petab_v2.Change(
                            target_id=condition_target(target, sbml_model),
                            target_value=parameter.pid,
                        )
                    )

        # the inputs of the networks which differ between the conditions
        # are changes of the condition of the first period, and the arrays
        # of such inputs are keyed by its id
        input_changes: list[petab_v2.Change] = []
        needs_condition = False
        if self.sciml is not None:
            input_changes = self.sciml.input_changes(simulation_key)
            needs_condition = self.sciml.needs_condition()

        plan = self.problem.plans[group_index]

        def changes_of(assignments: Sequence[Assignment]) -> list[petab_v2.Change]:
            # a formula is written in what the identifiers of the model mean,
            # which is what a condition of PEtab evaluates
            return [
                petab_v2.Change(
                    target_id=condition_target(a.target, sbml_model),
                    target_value=a.value
                    if a.value is not None
                    else formula_of_selections(str(a.formula), sbml_model),
                )
                for a in assignments
            ]

        # (time, changes) of every period, the first one is the one of the
        # changes before the initialization
        events = {event.time: event.assignments for event in plan.events}
        first: list[petab_v2.Change] = changes_of(plan.preinit)
        timeline: list[tuple[float, list[petab_v2.Change]]] = []
        if plan.steady_state is not None:
            first = changes_of(plan.steady_state.preinit) + first
            timeline.append((float("-inf"), first + version_changes + input_changes))
            timeline.append((plan.start, changes_of(events.pop(plan.start, ()))))
        else:
            timeline.append(
                (
                    plan.start,
                    first
                    + changes_of(events.pop(plan.start, ()))
                    + version_changes
                    + input_changes,
                )
            )
        for time in sorted(events):
            timeline.append((time, changes_of(events[time])))

        periods: list[petab_v2.ExperimentPeriod] = []
        for k, (time, changes) in enumerate(timeline):
            condition_ids: list[str] = []
            if changes or (k == 0 and needs_condition):
                condition_id = period_condition_id(experiment_id, k)
                if changes:
                    _table(petab_problem, "condition_tables").conditions.append(
                        petab_v2.Condition(id=condition_id, changes=changes)
                    )
                condition_ids.append(condition_id)
            periods.append(
                petab_v2.ExperimentPeriod(
                    time=time + plan.time_shift, condition_ids=condition_ids
                )
            )
        return periods

    # --- OBSERVABLES AND MEASUREMENTS ---

    def _add_observables_and_measurements(self, petab_problem: PetabProblem) -> None:
        """Add one observable per fit mapping with its reference data."""
        problem = self.problem
        self._name_observables()
        written: set[str] = set()
        for k in self.indices:
            observable_id = self.observable_ids[k]
            # the noise model of the mapping, which is the one it was read
            # with or the standard deviation of its data: the noise parameter
            # of a measurement fills in the placeholder the observable declares
            noise = self._noise_model(k, observable_id)
            observable = problem.observable_models[k]
            observable_values = self._observable_values(k)
            if observable_id not in written:
                written.add(observable_id)
                _table(petab_problem, "observable_tables").observables.append(
                    petab_v2.Observable(
                        id=observable_id,
                        name=f"{problem.experiment_keys[k]}.{problem.mapping_keys[k]}",
                        formula=self._observable_formula(k),
                        noise_formula=noise.formula,
                        noise_distribution=noise.distribution.value,
                        noise_placeholders=list(noise.placeholders),
                        observable_placeholders=list(observable.placeholders)
                        if observable is not None
                        else [],
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
                        observable_parameters=list(observable_values[i])
                        if observable_values
                        else [],
                        noise_parameters=list(noise.placeholder_values[i])
                        if noise.placeholders
                        else [],
                    )
                )

    def _name_observables(self) -> None:
        """Name the observable of every fit mapping which is written.

        The id of an observable is the key of its fit mapping, and the key
        with its experiment where two experiments share a key. Fit mappings
        which observe the same thing with the same noise in different
        experiments are one observable measured in several experiments,
        which is what the reader splits into one fit mapping per experiment
        (`<observable>_<experiment>`): they are written as one observable
        again, so that the problem which was read keeps its observables. An
        observable which would shadow an entity of the model is prefixed.

        Raises:
            ValueError: if fit mappings which observe different things, or
                one thing in the same experiment, get the id of one
                observable, e.g. the mappings `a.b` and `a_b`.
        """
        problem = self.problem
        keys = [problem.mapping_keys[k] for k in self.indices]
        unique = len(set(keys)) == len(keys)
        #: what the observable of a fit mapping is: its model, its formula and
        #: its noise
        contents: dict[int, tuple[Any, ...]] = {}
        content: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for k in self.indices:
            observable_id = (
                petab_id(problem.mapping_keys[k])
                if unique
                else petab_id(problem.experiment_keys[k], problem.mapping_keys[k])
            )
            self.observable_ids[k] = observable_id
            noise = noise_model_of(problem, k)
            observable = problem.observable_models[k]
            contents[k] = (
                self.model_ids[id(problem.models[k])],
                self._observable_formula(k),
                observable.placeholders if observable is not None else (),
                noise.formula,
                noise.distribution,
                tuple(noise.placeholders),
            )
            content[contents[k]].append(k)
        for group in content.values():
            if len(group) < 2:
                continue
            experiments = [self.experiment_ids.get(k) for k in group]
            if len(set(experiments)) != len(group) or None in experiments:
                continue
            stems: set[str] = set()
            for k, experiment in zip(group, experiments, strict=True):
                key, suffix = problem.mapping_keys[k], f"_{experiment}"
                if not key.endswith(suffix):
                    stems.clear()
                    break
                stems.add(key[: -len(suffix)])
            if len(stems) != 1:
                continue
            stem = petab_id(next(iter(stems)))
            for k in group:
                self.observable_ids[k] = stem
        # an observable must not shadow an entity of the model
        for k in self.indices:
            sbml_model = self.sbml_models.get(self.model_ids[id(problem.models[k])])
            observable_id = self.observable_ids[k]
            if sbml_model is not None and sbml_model.getElementBySId(observable_id):
                self.observable_ids[k] = petab_id("observable", observable_id)
        # the key of a fit mapping in the extension is the key the reader
        # gives it: the observable, and `<observable>_<experiment>` for an
        # observable which is measured in several experiments
        experiments_of: dict[str, set[str | None]] = defaultdict(set)
        for k in self.indices:
            experiments_of[self.observable_ids[k]].add(self.experiment_ids.get(k))
        for k in self.indices:
            observable_id = self.observable_ids[k]
            self.info_keys[k] = (
                observable_id
                if len(experiments_of[observable_id]) == 1
                else f"{observable_id}_{self.experiment_ids.get(k)}"
            )
        # the key of a fit mapping in the extension is one fit mapping, e.g.
        # `a` in the experiment `b_c` and `a_b` in `c` are both `a_b_c`
        by_info_key: dict[str, list[int]] = defaultdict(list)
        for k in self.indices:
            by_info_key[self.info_keys[k]].append(k)
        for info_key, group in by_info_key.items():
            if len(group) < 2:
                continue
            mappings = ", ".join(
                f"'{problem.mapping_keys[k]}' ({problem.experiment_keys[k]}, "
                f"experiment '{self.experiment_ids.get(k)}')"
                for k in group
            )
            raise ValueError(
                f"'{problem.opid}': the fit mappings {mappings} are the fit "
                f"mapping '{info_key}' of the `sbmlsim` extension, the key of "
                f"an observable and its experiment. Give the mappings keys "
                f"which differ in PEtab, i.e. in more than the characters "
                f"which are no letters, digits or `_`."
            )
        # one observable is one thing, measured once per experiment: the ids
        # of PEtab may join mappings whose keys differ
        by_id: dict[str, list[int]] = defaultdict(list)
        for k in self.indices:
            by_id[self.observable_ids[k]].append(k)
        for observable_id, group in by_id.items():
            info_keys = [self.info_keys[k] for k in group]
            if len({contents[k] for k in group}) == 1 and len(set(info_keys)) == len(
                info_keys
            ):
                continue
            mappings = ", ".join(
                f"'{problem.mapping_keys[k]}' ({problem.experiment_keys[k]})"
                for k in group
            )
            raise ValueError(
                f"'{problem.opid}': the fit mappings {mappings} are the "
                f"observable '{observable_id}' of PEtab, but they observe "
                f"different things or one thing in the same experiment. Give "
                f"the mappings keys which differ in PEtab, i.e. in more than "
                f"the characters which are no letters, digits or `_`."
            )

    def _observable_formula(self, k: int) -> str:
        """Get the formula of the observable of a fit mapping.

        An observable model is written as its formula, a selection as its
        math, both in what the identifiers of the model mean in PEtab, see
        `sbmlsim.fit.petab_v2.symbols`.

        Args:
            k: index of the fit mapping.

        Returns:
            The math of PEtab of the observable.
        """
        if k in self._formulas:
            return self._formulas[k]
        problem = self.problem
        sbml_model = self.sbml_models.get(self.model_ids[id(problem.models[k])])
        observable = problem.observable_models[k]
        if observable is not None:
            formula = petab_math_str(
                sympify_petab(formula_of_selections(observable.formula, sbml_model))
            )
        else:
            formula = observable_formula(problem.yid_observable[k], sbml_model)
        self._formulas[k] = formula
        return formula

    def _observable_values(self, k: int) -> list[list[float | str]]:
        """Get the values of the placeholders of the observable of a mapping.

        Args:
            k: index of the fit mapping.

        Returns:
            The values of every measurement in the math of PEtab, empty for an
            observable without placeholders.

        Raises:
            ValueError: if the observable model does not have the values of
                its placeholders for every measurement which is written.
        """
        problem = self.problem
        observable = problem.observable_models[k]
        if observable is None or not observable.placeholders:
            return []
        size = len(problem.y_references[k])
        if len(observable.placeholder_values) != size:
            raise ValueError(
                f"'{problem.opid}': the observable model of the fit mapping "
                f"'{problem.mapping_keys[k]}' has '{len(observable.placeholder_values)}' "
                f"values of its placeholders '{list(observable.placeholders)}' for "
                f"'{size}' measurements."
            )
        sbml_model = self.sbml_models.get(self.model_ids[id(problem.models[k])])
        return [
            [
                formula_of_selections(value, sbml_model)
                if isinstance(value, str)
                else value
                for value in values
            ]
            for values in observable.placeholder_values
        ]

    def _observable_unit(self, k: int) -> str:
        """Get the unit of the observable of a fit mapping."""
        problem = self.problem
        observable = problem.observable_models[k]
        if observable is not None:
            return observable.unit
        return str(problem.models[k].uinfo[problem.yid_observable[k]])

    def _noise_model(self, k: int, observable_id: str) -> NoiseModel:
        """Get the noise model a fit mapping is written with.

        Args:
            k: index of the fit mapping.
            observable_id: id of the observable the mapping is written as. The
                symbol of the noise formula which stands for the simulation is
                the id of the observable, so it is renamed to this id.

        Returns:
            The noise model of the mapping, see
            `sbmlsim.fit.petab_v2.likelihood.noise_model_of`, in the math of
            PEtab, i.e. the formula and the placeholder values in what the
            identifiers of the model mean.

        Raises:
            ValueError: if the noise model does not have the values of its
                placeholders for every measurement which is written.
        """
        problem = self.problem
        noise = noise_model_of(problem, k)
        size = len(problem.y_references[k])
        if noise.placeholders and len(noise.placeholder_values) != size:
            raise ValueError(
                f"'{problem.opid}': the noise model of the fit mapping "
                f"'{problem.mapping_keys[k]}' has '{len(noise.placeholder_values)}' "
                f"values of its placeholders '{list(noise.placeholders)}' for "
                f"'{size}' measurements."
            )
        sbml_model = self.sbml_models.get(self.model_ids[id(problem.models[k])])
        expression = sympify_petab(formula_of_selections(noise.formula, sbml_model))
        if noise.observable is not None and noise.observable != observable_id:
            expression = expression.subs(
                sp.Symbol(noise.observable, real=True),
                sp.Symbol(observable_id, real=True),
            )
        return dataclasses.replace(
            noise,
            formula=petab_math_str(expression),
            placeholder_values=tuple(
                tuple(
                    formula_of_selections(value, sbml_model)
                    if isinstance(value, str)
                    else value
                    for value in values
                )
                for values in noise.placeholder_values
            ),
            observable=observable_id if noise.observable is not None else None,
        )

    def _check_condition_targets(self, petab_problem: PetabProblem) -> None:
        """Refuse a condition which sets a parameter of the parameter table.

        PEtab does not allow an id in both tables: the parameter table gives
        the value of a parameter for the whole simulation, a condition sets
        it at the start of a period.

        Raises:
            ValueError: if a change of a simulation sets a parameter which is
                estimated or which the parameter table fixes, naming the
                parameter, the experiment and the simulation.
        """
        parameters = {p.id for p in petab_problem.parameters}
        conditions = {c.id: c for c in petab_problem.conditions}
        simulations = {
            experiment_id: simulation
            for simulation, ids in self._simulation_ids().items()
            for experiment_id in ids
        }
        for experiment in petab_problem.experiments:
            for period in experiment.periods:
                for condition_id in period.condition_ids:
                    condition = conditions.get(condition_id)
                    if condition is None:
                        continue
                    for change in condition.changes:
                        if change.target_id not in parameters:
                            continue
                        raise ValueError(
                            f"'{self.problem.opid}': the parameter "
                            f"'{change.target_id}' is a row of the parameter "
                            f"table and is set by the condition '{condition_id}' "
                            f"of the experiment '{experiment.id}' (simulation "
                            f"'{simulations.get(experiment.id)}'), PEtab does not "
                            f"allow a parameter in both tables. Do not change a "
                            f"row of the parameter table (a parameter of the fit "
                            f"or a parameter an input of a network uses) in the "
                            f"simulation."
                        )

    # --- PARAMETERS ---

    def _add_parameters(self, petab_problem: PetabProblem) -> None:
        """Add the parameters which are estimated.

        The nominal value of a plain parameter is its value, i.e. its start
        value or the value of the parameter set of the export. The
        nominal value of a version is the value its target has in the model
        rather than the version's own start value, so that a tool which does
        not estimate it still simulates the model as it is today, i.e. with
        no version applied.
        """
        elements = self.sciml.element_ids if self.sciml is not None else set()
        for index, parameter in enumerate(self.problem.parameters):
            if parameter.pid in elements:
                # the rows of the networks describe the elements
                continue
            nominal_value = (
                float(self.problem.xmodel[index])
                if has_renamed_targets([parameter])
                else self.values[parameter.pid]
            )
            _table(petab_problem, "parameter_tables").parameters.append(
                petab_v2.Parameter(
                    id=parameter.pid,
                    lb=parameter.lower_bound,
                    ub=parameter.upper_bound,
                    nominal_value=nominal_value,
                    estimate=True,
                    prior_distribution=None
                    if parameter.prior is None
                    else parameter.prior.distribution.value,
                    prior_parameters=[]
                    if parameter.prior is None
                    else list(parameter.prior.parameters),
                )
            )
        self._add_noise_parameters(petab_problem)
        if self.sciml is not None:
            _table(petab_problem, "parameter_tables").parameters.extend(
                self.sciml.parameter_rows()
            )
            _table(petab_problem, "mapping_tables").mappings.extend(
                self.sciml.mapping_rows()
            )

    def _add_noise_parameters(self, petab_problem: PetabProblem) -> None:
        """Add the parameters of the noise models, every one of them once.

        A parameter of a noise formula is a row of the parameter table, with
        the nominal value the log-likelihood uses. It is written as estimated
        if the problem it was read from estimates it, which `sbmlsim` does not
        do, see the `noise-parameters` gap.

        Raises:
            ValueError: if two noise models do not agree on a parameter, i.e.
                on its value, its bounds or whether it is estimated.
        """
        written: dict[str, NoiseParameter] = {}
        for k in self.indices:
            for parameter in noise_model_of(self.problem, k).parameters:
                if parameter.pid in self.problem.pids:
                    # a parameter of the fit, which is written already
                    continue
                if parameter.pid in written:
                    if written[parameter.pid] != parameter:
                        raise ValueError(
                            f"'{self.problem.opid}': the parameter "
                            f"'{parameter.pid}' of the noise is "
                            f"'{written[parameter.pid]}' and '{parameter}' in "
                            f"the noise models of two fit mappings."
                        )
                    continue
                written[parameter.pid] = parameter
                _table(petab_problem, "parameter_tables").parameters.append(
                    petab_v2.Parameter(
                        id=parameter.pid,
                        lb=parameter.lower_bound,
                        ub=parameter.upper_bound,
                        nominal_value=parameter.value,
                        estimate=parameter.estimate,
                    )
                )

    # --- EXTENSION ---

    def _add_extension(self, petab_problem: PetabProblem) -> None:
        """Add what the tables of PEtab do not hold."""
        problem = self.problem
        settings = problem.settings

        elements = self.sciml.element_ids if self.sciml is not None else set()
        parameters = {
            parameter.pid: {
                "unit": parameter.unit,
                "start_value": self.values[parameter.pid],
                **(
                    {"scale": parameter.scale.name}
                    if parameter.scale is not None
                    else {}
                ),
            }
            for parameter in problem.parameters
            if parameter.pid not in elements
        }

        observables: dict[str, dict[str, Any]] = {}
        for k in self.indices:
            model = problem.models[k]
            # keyed by the fit mapping, several of which may share an
            # observable, see `_name_observables`
            observables[self.info_keys[k]] = {
                "observable": self.observable_ids[k],
                "experiment": problem.experiment_keys[k],
                "mapping": problem.mapping_keys[k],
                "kind": problem.mapping_kinds[k].value,
                "collection": problem.mapping_collections[
                    problem.collection_indices[k]
                ].sid,
                "xid_observable": problem.xid_observable[k],
                # an observable model is written as its formula, which the
                # reader reads as an observable model again
                "yid_observable": None
                if problem.observable_models[k] is not None
                else problem.yid_observable[k],
                "x_unit": str(model.uinfo[problem.xid_observable[k]]),
                "y_unit": self._observable_unit(k),
                "error_type": problem.y_errors_type[k],
                "weight_curve": float(problem.weights_curves[k]),
                "weight_mapping": self._weight_mapping(k),
                "model": self.model_ids[id(model)],
            }

        experiments: dict[str, dict[str, Any]] = {}
        for k, experiment_id in self.experiment_ids.items():
            if experiment_id in experiments:
                continue
            collection_index = self.experiment_collections[experiment_id]
            experiments[experiment_id] = {
                "collection": problem.mapping_collections[collection_index].sid,
                # the simulation as the experiment defines it, with units
                "simulation": problem.simulations[k].to_dict(),
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
            inputs=dict(self.sciml.fallbacks) if self.sciml is not None else {},
            gaps=gaps_dict(self.gaps),
        )
        if petab_problem.config is None:
            petab_problem.config = petab_v2.ProblemConfig(format_version="2.0.0")
        petab_problem.config.extensions[EXTENSION_ID] = extension
        if self.sciml is not None:
            petab_problem.config.extensions[SCIML_EXTENSION_ID] = self.sciml.config()

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


def to_petab(
    problem: OptimizationProblem,
    output_dir: Path,
    settings: FitSettings | None = None,
    kinds: set[MappingKind] | None = None,
    required_extension: bool = True,
    parameter_set: ParameterSet | None = None,
) -> Path:
    """Write an optimization problem as a PEtab v2 problem.

    Args:
        problem: problem to write, it is initialized if it is not.
        output_dir: directory the problem is written to, created if it does not
            exist.
        settings: settings of the fit, required if the problem is not
            initialized.
        kinds: kinds of fit mappings to write, everything the problem resolved
            by default, i.e. the training data, the validation data and the
            outliers.
        required_extension: mark the `sbmlsim` extension as required, which it
            is: the settings it carries are the objective of the fit. `False`
            writes a problem other tools fit with the objective of PEtab.
        parameter_set: values of the parameters of the fit to write, e.g. the
            result of a fit: the exported problem starts from them, see
            `PetabExporter`. The start values of the problem are written if
            there is none.

    Returns:
        Path of the YAML file of the problem.

    Raises:
        ValueError: if the problem uses features PEtab v2 cannot express, or if
            the parameter set has a value for an element of a network which is
            no parameter of the fit.
        KeyError: if the parameter set lacks a parameter of the problem.
    """
    exporter = PetabExporter(
        problem,
        settings=settings,
        kinds=kinds,
        required_extension=required_extension,
        parameter_set=parameter_set,
    )
    petab_problem = exporter.to_problem()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # `to_files` writes every model from its libsbml document, a derived model
    # as its source, see `_add_models`
    _set_table_paths(petab_problem)
    petab_problem.to_files(base_path=output_dir)
    if exporter.sciml is not None:
        exporter.sciml.write(output_dir)
    yaml_file = output_dir / YAML_FILE
    logger.info("PEtab v2 problem written: %s", yaml_file.resolve().as_uri())
    return yaml_file
