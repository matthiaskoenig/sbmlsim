"""Tests that the export writes the definition of a fit, not its state."""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import petab.v2 as petab_v2
import pytest
import yaml
from test_petab_v2_reader import write_problem

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.objects import FitParameter
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import ParameterScaleType
from sbmlsim.fit.petab_v2 import PetabExporter, to_petab
from sbmlsim.fit.petab_v2.extension import (
    EXTENSION_VERSION,
    SbmlsimExtension,
    extension_of,
)
from sbmlsim.fit.petab_v2.likelihood import log_likelihood
from sbmlsim.fit.petab_v2.reader import DEFAULT_EXPERIMENT, PetabReader, from_petab
from tests.fit.hooks import Scaling, factor_parameter


def _extension(config: Any) -> SbmlsimExtension:
    extension = extension_of(config)
    assert extension is not None
    return extension


def _hctz(settings: FitSettings, **replaced: Any) -> OptimizationProblem:
    definition = dataclasses.replace(FIT_DEFINITIONS["PK"], **replaced)
    problem = definition.problem(opid="hctz_definition")
    problem.initialize(settings)
    return problem


def test_export_after_an_evaluation_writes_the_definition(
    fit_settings: FitSettings,
) -> None:
    """The values an evaluation writes into the timecourses are not conditions."""
    problem = _hctz(fit_settings)
    x = np.asarray(problem.x0, dtype=float)
    problem.cost_least_square(problem.to_scale(x))
    # the evaluation wrote the parameters into the first timecourses
    assert any(
        set(problem.pids) & set(simulation.timecourses[0].changes)
        for simulation in problem.simulations
    )
    assert not any(
        set(problem.pids) & set(changes) for changes in problem.defined_changes
    )

    petab_problem = PetabExporter(problem).to_problem()
    targets = {
        change.target_id
        for condition in petab_problem.conditions
        for change in condition.changes
    }
    assert not targets & set(problem.pids)


def test_the_observables_are_named_after_the_mappings(
    fit_settings: FitSettings,
) -> None:
    """Unique mapping keys are the observable ids, without the experiment."""
    problem = _hctz(fit_settings)
    exporter = PetabExporter(problem)
    petab_problem = exporter.to_problem()
    ids = {observable.id for observable in petab_problem.observables}
    assert ids == set(problem.mapping_keys)
    assert _extension(petab_problem.config).version == EXTENSION_VERSION
    info = _extension(petab_problem.config).observables
    assert set(info) == set(problem.mapping_keys)
    assert all(entry["observable"] == key for key, entry in info.items())


def test_an_observable_measured_in_two_experiments_is_written_once(
    tmp_path: Path,
) -> None:
    """The reader splits it into two mappings, the export joins them again."""
    path = write_problem(
        tmp_path / "problem", {"prey_o": "prey"}, {"e1": None, "e2": None}
    )
    problem, _ = from_petab(path)
    # the bounds of the problem start at zero, which the logarithmic scale
    # of the default settings refuses
    settings = FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    problem.initialize(settings)
    assert sorted(problem.mapping_keys) == ["prey_o_e1", "prey_o_e2"]

    yaml_file = to_petab(problem, tmp_path / "export")
    petab_problem = petab_v2.Problem.from_yaml(yaml_file)
    assert [observable.id for observable in petab_problem.observables] == ["prey_o"]
    assert {m.experiment_id for m in petab_problem.measurements} == {"e1", "e2"}

    restored, _ = from_petab(yaml_file)
    restored.initialize(settings)
    reader = PetabReader.from_yaml(yaml_file)
    assert sorted(restored.mapping_keys) == ["prey_o_e1", "prey_o_e2"]
    assert reader.observable_info("prey_o_e1")["mapping"] == "prey_o_e1"
    assert log_likelihood(restored) == pytest.approx(log_likelihood(problem), rel=1e-9)


def test_a_formula_observable_is_written_as_its_formula(tmp_path: Path) -> None:
    """The model of the problem is written, not the one with the observable."""
    path = write_problem(
        tmp_path / "problem", {"total": "prey + predator"}, {"e1": None}
    )
    reader = PetabReader.from_yaml(path)
    reader.derived_dir = tmp_path / "derived"
    problem = reader.to_optimization_problem(opid="formula")
    settings = FitSettings(parameter_scale=ParameterScaleType.LINEAR)
    problem.initialize(settings)
    assert problem.yid_observable == ["observable_total"]

    yaml_file = to_petab(problem, tmp_path / "export")
    petab_problem = petab_v2.Problem.from_yaml(yaml_file)
    assert (tmp_path / "export" / "lv.xml").is_file()
    assert not (tmp_path / "export" / "lv_observables.xml").exists()
    assert not petab_problem.models[0].has_entity_with_id("observable_total")
    (observable,) = petab_problem.observables
    assert str(observable.formula) == "predator + prey"
    info = _extension(petab_problem.config).observables["total"]
    assert info["yid_observable"] is None

    restored, _ = from_petab(yaml_file)
    restored.initialize(settings)
    assert restored.yid_observable == ["observable_total"]
    assert log_likelihood(restored) == pytest.approx(log_likelihood(problem), rel=1e-9)


def test_a_model_named_model(tmp_path: Path) -> None:
    """The model id `model` of the PEtab documentation is not the default experiment."""
    path = write_problem(tmp_path / "problem", {"prey_o": "prey"}, {"e1": None})
    # the measurements name no experiment, and the model is named `model`
    measurements = pd.read_csv(tmp_path / "problem" / "measurements.tsv", sep="\t")
    measurements.drop(columns=["experimentId"]).to_csv(
        tmp_path / "problem" / "measurements.tsv", sep="\t", index=False
    )
    config = yaml.safe_load(path.read_text())
    config.pop("experiment_files")
    config["model_files"] = {"model": config["model_files"]["lv"]}
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    problem, _ = from_petab(path)
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    assert problem.simulation_keys == [DEFAULT_EXPERIMENT] * 1
    assert DEFAULT_EXPERIMENT != "model"


def test_the_scale_of_a_parameter_survives_the_round_trip(
    fit_settings: FitSettings, tmp_path: Path
) -> None:
    first = FIT_DEFINITIONS["PK"].parameters[0]
    parameters = [
        FitParameter(
            first.pid,
            first.start_value,
            first.lower_bound,
            first.upper_bound,
            unit=first.unit,
            target=first.target,
            mappings=first.mappings,
            scale=ParameterScaleType.LINEAR,
        ),
        *FIT_DEFINITIONS["PK"].parameters[1:],
    ]
    problem = _hctz(fit_settings, parameters=parameters)
    yaml_file = to_petab(problem, tmp_path / "export")
    extension = _extension(petab_v2.Problem.from_yaml(yaml_file).config)
    assert extension.parameters[parameters[0].pid]["scale"] == "LINEAR"
    restored, _ = from_petab(yaml_file)
    scales = {p.pid: p.scale for p in restored.parameters}
    assert scales[parameters[0].pid] is ParameterScaleType.LINEAR
    assert all(
        scale is None for pid, scale in scales.items() if pid != parameters[0].pid
    )


def test_a_problem_with_a_hook_which_is_no_network_is_refused(
    definition_hctz_iv: FitDefinition, fit_settings: FitSettings
) -> None:
    """A hook which is no network has no representation in PEtab SciML."""
    definition = dataclasses.replace(
        definition_hctz_iv, parameters=[factor_parameter()], hybridizations=[Scaling()]
    )
    problem = definition.problem(opid="external")
    problem.initialize(fit_settings)
    with pytest.raises(ValueError, match="is not a `Hybridization`"):
        PetabExporter(problem)
