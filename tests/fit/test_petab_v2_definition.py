"""Tests that the export writes the definition of a fit, not its state."""

import copy
import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Any

import libsbml
import numpy as np
import pandas as pd
import petab.v2 as petab_v2
import pytest
import yaml

from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.fit import FitSettings
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
from tests.fit.test_petab_v2_reader import write_problem


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


def test_an_export_after_an_evaluation_equals_one_before(
    fit_settings: FitSettings, tmp_path: Path
) -> None:
    """Every file, the extension included, is the definition, and valid PEtab."""
    problem = _hctz(fit_settings)
    before = to_petab(problem, tmp_path / "before").parent
    problem.cost_least_square(problem.to_scale(np.asarray(problem.x0, dtype=float)))
    after = to_petab(problem, tmp_path / "after").parent

    names = sorted(path.name for path in before.iterdir())
    assert names == sorted(path.name for path in after.iterdir())
    for name in names:
        assert (before / name).read_bytes() == (after / name).read_bytes(), name

    # the problem which is read from it is written again as valid PEtab
    restored, settings = from_petab(after / "problem.yaml")
    restored.initialize(settings)
    again = to_petab(restored, tmp_path / "again")
    issues = petab_v2.Problem.from_yaml(again).validate()
    assert not issues.has_errors(), str(issues)


def test_the_ids_of_the_simulations_are_the_ids_of_the_experiments(
    fit_settings: FitSettings,
) -> None:
    """The networks key their arrays by the ids `_add_experiments` gives."""
    problem = _hctz(fit_settings)
    exporter = PetabExporter(problem)
    ids = exporter._simulation_ids()
    exporter.to_problem()
    expected: dict[str, list[str]] = defaultdict(list)
    for k in sorted(exporter.experiment_ids):
        experiment_id = exporter.experiment_ids[k]
        if experiment_id not in expected[problem.simulation_keys[k]]:
            expected[problem.simulation_keys[k]].append(experiment_id)
    assert {key: sorted(value) for key, value in ids.items()} == {
        key: sorted(value) for key, value in expected.items()
    }
    # a simulation of several experiments has the id of each of them
    assert any(len(value) > 1 for value in ids.values())


def test_two_observables_whose_ids_collide_are_refused(tmp_path: Path) -> None:
    """A mapping `prey.o` is the observable `prey_o` of PEtab, which is taken."""
    path = write_problem(
        tmp_path / "problem", {"prey_o": "prey", "q": "predator"}, {"e1": None}
    )
    problem, _ = from_petab(path)
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    problem.mapping_keys[problem.mapping_keys.index("q")] = "prey.o"
    with pytest.raises(ValueError, match=r"'prey_o'.*'prey\.o'"):
        PetabExporter(problem).to_problem()


def test_two_mappings_whose_keys_in_the_extension_collide_are_refused(
    tmp_path: Path,
) -> None:
    """`a` in the experiment `b_c` and `a_b` in `c` are both `a_b_c`."""
    path = write_problem(
        tmp_path / "problem",
        {"a": "prey", "z": "predator"},
        {"b_c": None, "d": None, "c": None, "e": None},
        measured={"a": ["b_c", "d"], "z": ["c", "e"]},
    )
    problem, _ = from_petab(path)
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    # the observable `a_b` measured in the experiments `c` and `e`
    for old, new in [("z_c", "a_b_c"), ("z_e", "a_b_e")]:
        problem.mapping_keys[problem.mapping_keys.index(old)] = new
    with pytest.raises(
        ValueError, match=r"'a_b_c'.*experiment 'b_c'.*'a_b_c'.*experiment 'c'"
    ):
        PetabExporter(problem).to_problem()


def test_two_models_of_one_file_name_are_two_files(
    fit_settings: FitSettings, tmp_path: Path
) -> None:
    """A model is not overwritten by another model of the same file name."""
    problem = _hctz(fit_settings)
    source = Path(problem.models[0].source.path)
    other = tmp_path / "other" / source.name
    other.parent.mkdir()
    document: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(source))
    document.getModel().setName("other")
    libsbml.writeSBMLToFile(document, str(other))
    # the experiments share one model, the last collection gets another file
    # of the same name
    model = copy.copy(problem.models[-1])
    model.source = dataclasses.replace(model.source, path=other)
    collection = problem.collection_indices[-1]
    for k in range(len(problem.models)):
        if problem.collection_indices[k] == collection:
            problem.models[k] = model

    yaml_file = to_petab(problem, tmp_path / "export")
    config = yaml.safe_load(yaml_file.read_text())
    locations = [entry["location"] for entry in config["model_files"].values()]
    assert len(locations) == len(set(locations)) == 2
    names = {
        libsbml.readSBMLFromFile(str(tmp_path / "export" / location))
        .getModel()
        .getName()
        for location in locations
    }
    assert "other" in names and len(names) == 2


def test_the_info_of_a_mapping_falls_back_to_the_observable_only_before_0_2(
    tmp_path: Path,
) -> None:
    """Version 0.1.0 keyed the block by the observable, 0.2.0 by the mapping."""
    path = write_problem(
        tmp_path / "problem", {"prey_o": "prey"}, {"e1": None, "e2": None}
    )
    problem, _ = from_petab(path)
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    yaml_file = to_petab(problem, tmp_path / "export")
    config = yaml.safe_load(yaml_file.read_text())
    block = config["extensions"]["sbmlsim"]
    block["observables"] = {"prey_o": block["observables"]["prey_o_e1"]}
    yaml_file.write_text(yaml.safe_dump(config, sort_keys=False))
    assert PetabReader.from_yaml(yaml_file).observable_info("prey_o_e1") == {}

    block["version"] = "0.1.0"
    yaml_file.write_text(yaml.safe_dump(config, sort_keys=False))
    info = PetabReader.from_yaml(yaml_file).observable_info("prey_o_e1")
    assert info["mapping"] == "prey_o_e1"


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
    # the default experiment and the model do not share the key `model`
    problem, _ = from_petab(path)
    problem.initialize(FitSettings(parameter_scale=ParameterScaleType.LINEAR))
    assert problem.simulation_keys == [DEFAULT_EXPERIMENT]
    assert problem.model_keys == ["model"]


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
