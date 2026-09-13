"""Tests of the PEtab v2 layer."""

import dataclasses
from pathlib import Path

import numpy as np
import petab.v2 as petab_v2
import pytest

from conftest import is_intravenous, is_oral  # ty: ignore[unresolved-import]
from examples.hctz_fitting.fitting.fitting import FIT_DEFINITIONS
from sbmlsim.experiment import SimulationExperiment
from sbmlsim.fit import FitSettings
from sbmlsim.fit.cli import FitDefinition
from sbmlsim.fit.objects import FitParameter, MappingKind
from sbmlsim.fit.optimization import OptimizationProblem
from sbmlsim.fit.options import (
    ResidualType,
    WeightingCurvesType,
    WeightingPointsType,
)
from sbmlsim.fit.petab_v2 import (
    GAPS,
    GapKind,
    PetabExporter,
    gaps_of_problem,
    gaps_table,
    to_petab,
)
from sbmlsim.fit.petab_v2.export import NOISE_PLACEHOLDER, petab_id
from sbmlsim.fit.petab_v2.extension import EXTENSION_ID, extension_of
from sbmlsim.fit.petab_v2.reader import PetabReader, from_petab


@pytest.fixture(scope="module")
def fit_settings_module() -> FitSettings:
    """Get the settings of the fit, for the whole module.

    The same settings as the `fit_settings` fixture, which is per test and
    cannot be used by the module scoped problem.
    """
    return FitSettings(
        residual=ResidualType.NORMALIZED,
        weighting_curves=(WeightingCurvesType.POINTS,),
        weighting_points=WeightingPointsType.ERROR_WEIGHTING,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )


@pytest.fixture(scope="module")
def op_hctz_pk_module(fit_settings_module: FitSettings) -> OptimizationProblem:
    """Get the initialized problem of the pharmacokinetics, for the module."""
    problem = FIT_DEFINITIONS["PK"].problem(opid="hctz_pk_petab")
    problem.initialize(fit_settings_module)
    return problem


@pytest.fixture(scope="module")
def petab_dir(
    tmp_path_factory: pytest.TempPathFactory,
    op_hctz_pk_module: OptimizationProblem,
    fit_settings_module: FitSettings,
) -> Path:
    """Write the HCTZ PK problem as a PEtab v2 problem, once for the module."""
    output_dir = tmp_path_factory.mktemp("petab")
    to_petab(op_hctz_pk_module, output_dir, settings=fit_settings_module)
    return output_dir


def test_petab_id() -> None:
    """A PEtab identifier has letters, digits and underscores."""
    assert petab_id("Beermann1976", "fm_1") == "Beermann1976__fm_1"
    assert petab_id("a.b", "c d") == "a_b__c_d"
    assert petab_id("1976") == "_1976"


def test_noise_placeholder_is_declared(petab_dir: Path) -> None:
    """PEtab v2 declares the placeholders of an observable.

    The `noiseParameter${n}_${observableId}` names of v1 are gone.
    """
    problem = petab_v2.Problem.from_yaml(petab_dir / "problem.yaml")
    with_noise = [
        observable
        for observable in problem.observables
        if observable.noise_placeholders
    ]
    assert with_noise
    for observable in with_noise:
        assert [str(p) for p in observable.noise_placeholders] == [NOISE_PLACEHOLDER]
        assert str(observable.noise_formula) == NOISE_PLACEHOLDER


def test_gaps_are_documented() -> None:
    """Every gap says what it is and what the layer does about it."""
    assert GAPS
    assert len({gap.id for gap in GAPS}) == len(GAPS)
    for gap in GAPS:
        assert gap.kind in set(GapKind)
        assert gap.sbmlsim and gap.petab and gap.detail


def test_gaps_of_problem(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The gaps of a problem are the ones it runs into."""
    op_hctz_pk.initialize(fit_settings)
    gaps = gaps_of_problem(op_hctz_pk)
    ids = {gap.id for gap in gaps}

    # the HCTZ problem has units, settings and an output grid, and it is fitted
    # on training data with validation data; Beermann1976's PO single dose
    # also shares one simulation between training and outlier mappings
    assert {
        "units",
        "fit-settings",
        "output-times",
        "mapping-kind",
        "experiment-split",
    } <= ids
    # and it does not use what PEtab cannot express at all
    assert not [gap for gap in gaps if gap.kind == GapKind.UNSUPPORTED]
    assert gaps_table(gaps).row_count == len(gaps)


def test_gaps_require_an_initialized_problem(
    op_hctz_pk: OptimizationProblem,
) -> None:
    """The gaps of the data are only known when the data is resolved."""
    with pytest.raises(ValueError, match="initialize"):
        gaps_of_problem(op_hctz_pk)


def test_export_writes_the_problem(petab_dir: Path) -> None:
    """The export writes the YAML, the tables and the model."""
    files = {path.name for path in petab_dir.iterdir()}
    assert {
        "problem.yaml",
        "conditions.tsv",
        "experiments.tsv",
        "observables.tsv",
        "measurements.tsv",
        "parameters.tsv",
    } <= files
    assert any(name.endswith(".xml") for name in files)


def test_exported_problem_is_valid_petab(petab_dir: Path) -> None:
    """PEtab reads and validates what the export writes."""
    problem = petab_v2.Problem.from_yaml(petab_dir / "problem.yaml")
    assert len(problem.models) == 1
    assert problem.measurements
    assert problem.observables
    assert len(problem.parameters) == 3

    issues = problem.validate()
    # the only message is that nothing validates the `sbmlsim` extension
    errors = [issue for issue in issues if "sbmlsim" not in str(issue)]
    assert not errors, f"validation of the exported problem failed: {errors}"


def test_extension_carries_the_fit(
    petab_dir: Path, fit_settings_module: FitSettings
) -> None:
    """What PEtab does not express is in the extension."""
    problem = petab_v2.Problem.from_yaml(petab_dir / "problem.yaml")
    assert EXTENSION_ID in problem.config.extensions

    extension = extension_of(problem.config)
    assert extension is not None
    assert FitSettings.from_dict(extension.settings) == fit_settings_module
    assert extension.parameters
    assert extension.observables
    assert extension.gaps

    # the units of the data and the kind of every mapping are kept
    for info in extension.observables.values():
        assert info["y_unit"]
        assert info["kind"] in {kind.value for kind in MappingKind}


def test_the_extension_is_required(petab_dir: Path) -> None:
    """The settings of the extension are the objective, so it is required.

    PEtab asks an extension which changes the mathematical interpretation of a
    problem to be `required`, i.e. a tool which does not know it rejects the
    problem rather than fitting the same data with another objective.
    """
    problem = petab_v2.Problem.from_yaml(petab_dir / "problem.yaml")
    extension = extension_of(problem.config)
    assert extension is not None
    assert extension.required is True


def test_a_problem_for_other_tools(
    tmp_path: Path,
    op_hctz_pk: OptimizationProblem,
    fit_settings: FitSettings,
) -> None:
    """A problem which other tools should fit is written as not required."""
    to_petab(
        op_hctz_pk,
        tmp_path,
        settings=fit_settings,
        required_extension=False,
    )
    problem = petab_v2.Problem.from_yaml(tmp_path / "problem.yaml")
    extension = extension_of(problem.config)
    assert extension is not None
    assert extension.required is False
    # and it is still the fit, i.e. the settings are there to read
    assert FitSettings.from_dict(extension.settings) == fit_settings


def test_outliers_are_exported_with_their_kind(
    op_hctz_pk: OptimizationProblem, fit_settings: FitSettings
) -> None:
    """The outliers are written, the extension says that a fit drops them.

    They are measurements like the validation data, so a round trip keeps
    them; what a fit does with them is the kind in the extension, which is
    why the extension is required.
    """
    exporter = PetabExporter(op_hctz_pk, settings=fit_settings)
    kinds = {op_hctz_pk.mapping_kinds[k] for k in exporter.indices}
    assert MappingKind.OUTLIER in kinds
    assert MappingKind.EXCLUDED not in kinds


def test_experiments_are_named_after_the_collections(petab_dir: Path) -> None:
    """A PEtab problem is built from the fit mapping collections.

    A collection whose mappings share a simulation is exactly one experiment
    and carries its id; a collection over several simulations, e.g. the doses
    of a study, is one experiment per simulation, numbered after it.
    """
    problem = petab_v2.Problem.from_yaml(petab_dir / "problem.yaml")
    extension = extension_of(problem.config)
    assert extension is not None

    experiment_ids = {experiment.id for experiment in problem.experiments}
    assert experiment_ids
    for experiment_id in experiment_ids:
        collection = extension.experiments[experiment_id]["collection"]
        assert collection in extension.collections
        assert experiment_id.startswith(collection)

    # the validation data of the example is one simulation, i.e. one experiment
    assert "Patel1984_validation" in experiment_ids


def test_one_collection_per_experiment(petab_dir: Path) -> None:
    """The reader gives one collection back for every experiment."""
    reader = PetabReader.from_yaml(petab_dir / "problem.yaml")
    collections = reader.mapping_collections(reader.experiment_class())

    assert len(collections) == len(reader.petab_problem.experiments)
    ids = {collection.sid for collection in collections}
    assert ids == {experiment.id for experiment in reader.petab_problem.experiments}
    # and every mapping of the problem is in exactly one collection
    mappings = [m for collection in collections for m in collection.mappings]
    assert len(mappings) == len(set(mappings))
    assert set(mappings) == {
        observable.id for observable in reader.petab_problem.observables
    }


def test_reader_builds_an_experiment(petab_dir: Path) -> None:
    """The problem is read as a simulation experiment."""
    reader = PetabReader.from_yaml(petab_dir / "problem.yaml")
    experiment_class = reader.experiment_class()
    assert issubclass(experiment_class, SimulationExperiment)
    assert reader.models()
    assert reader.simulations()
    assert reader.tasks()
    assert reader.datasets()
    assert len(reader.fit_parameters()) == 3


def test_round_trip_keeps_the_fit(
    petab_dir: Path,
    op_hctz_pk_module: OptimizationProblem,
    fit_settings_module: FitSettings,
) -> None:
    """A problem which is written and read again is the fit it started from."""
    problem, settings = from_petab(petab_dir / "problem.yaml", opid="hctz_petab")
    assert settings == fit_settings_module

    problem.initialize(settings)
    original = op_hctz_pk_module

    # the same mappings, with their kinds and their units
    assert len(problem.mapping_keys) == len(original.mapping_keys)
    assert set(problem.mapping_kinds) == set(original.mapping_kinds)
    assert [p.unit for p in problem.parameters] == [p.unit for p in original.parameters]
    assert [p.start_value for p in problem.parameters] == [
        p.start_value for p in original.parameters
    ]

    # and the same objective, up to the selections of the tasks, see the
    # `selections` gap
    x = np.log10(np.asarray(original.x0, dtype=float))
    cost_original = original.cost_least_square(x)
    cost_petab = problem.cost_least_square(x)
    assert cost_petab == pytest.approx(cost_original, rel=1e-4)


def test_round_trip_keeps_the_data(
    petab_dir: Path, op_hctz_pk_module: OptimizationProblem
) -> None:
    """The reference data of every mapping survives the tables."""
    problem, settings = from_petab(petab_dir / "problem.yaml")
    problem.initialize(settings)
    original = op_hctz_pk_module

    keys = {
        f"{original.experiment_keys[k]}__{original.mapping_keys[k]}": k
        for k in range(len(original.mapping_keys))
    }
    for i, key in enumerate(problem.mapping_keys):
        k = keys[key]
        assert np.array_equal(
            np.asarray(problem.x_references[i]),
            np.asarray(original.x_references[k]),
        )
        assert np.allclose(
            np.asarray(problem.y_references[i]),
            np.asarray(original.y_references[k]),
            rtol=1e-10,
        )
        assert problem.weights_curves[i] == pytest.approx(original.weights_curves[k])


def test_a_versioned_parameter_is_written_as_a_condition(
    tmp_path: Path,
    definition_hctz_pk: FitDefinition,
    fit_settings: FitSettings,
) -> None:
    """PEtab says `Ka_dis_hctz = Ka_po` in the experiments of the version."""
    definition = dataclasses.replace(
        definition_hctz_pk,
        parameters=[
            FitParameter(
                "Ka_po",
                0.35,
                0.01,
                10.0,
                "1/hr",
                target="Ka_dis_hctz",
                mappings=is_oral,
            ),
        ],
    )
    problem = definition.problem(opid="hctz_pk_versioned")
    to_petab(problem, tmp_path, settings=fit_settings)

    petab_problem = petab_v2.Problem.from_yaml(tmp_path / "problem.yaml")
    assert [p.id for p in petab_problem.parameters] == ["Ka_po"]

    changes = [
        change
        for condition in petab_problem.conditions
        for change in condition.changes
        if change.target_id == "Ka_dis_hctz"
    ]
    assert changes, "no condition writes the target of the version"
    assert all(str(change.target_value) == "Ka_po" for change in changes)

    issues = petab_problem.validate()
    errors = [issue for issue in issues if "sbmlsim" not in str(issue)]
    assert not errors, f"validation failed: {errors}"


def test_the_round_trip_keeps_a_versioned_parameter(
    tmp_path: Path,
    definition_hctz_pk: FitDefinition,
    fit_settings: FitSettings,
) -> None:
    """A version survives being written and read, as a set of ids.

    A selector is a callable and cannot be written to a TSV, so PEtab stores
    the resolution. The parameter which comes back selects the same mappings
    by their id, which is the same fit.

    `is_intravenous` selects only training data of one route, none of it
    shares a simulation with an outlier: a simulation whose mappings are of
    several kinds is written as one PEtab experiment per kind, because
    `select_mapping_collections` gives an experiment one `FitMappingCollection`
    per kind it holds and `PetabExporter._add_experiments` writes one PEtab
    experiment per collection (see the `experiment-split` gap). A selector
    reaching across such a split would count more covered simulation groups
    after the round trip than before, which is that unrelated gap and not a
    versioning concern; the binding itself is unaffected by the split, since
    every experiment it produces carries the same version.
    """
    definition = dataclasses.replace(
        definition_hctz_pk,
        parameters=[
            FitParameter(
                "Ka_iv",
                0.35,
                0.01,
                10.0,
                "1/hr",
                target="Ka_dis_hctz",
                mappings=is_intravenous,
            ),
        ],
    )
    problem = definition.problem(opid="hctz_pk_versioned_rt")
    problem.initialize(fit_settings)
    to_petab(problem, tmp_path, settings=fit_settings)

    read_problem, settings = from_petab(tmp_path / "problem.yaml", opid="read")
    read_problem.initialize(settings)

    (parameter,) = read_problem.parameters
    assert parameter.pid == "Ka_iv"
    assert parameter.target_id == "Ka_dis_hctz"
    assert parameter.is_versioned

    # the same simulations are covered as before
    before = problem.parameter_mapping_initialized.coverage()[0]
    after = read_problem.parameter_mapping_initialized.coverage()[0]
    assert after.n_covered == before.n_covered

    # and it is the same fit: the cost of the model values agrees
    x_before = problem.to_scale(problem.xmodel)
    x_after = read_problem.to_scale(read_problem.xmodel)
    assert read_problem.cost_least_square(x_after) == pytest.approx(
        problem.cost_least_square(x_before), rel=1e-4
    )


def test_a_condition_of_an_unsupported_value_raises(petab_dir: Path) -> None:
    """A condition value which is neither a number nor an estimated id raises.

    `_is_number` alone is too wide a net for "is a versioned parameter's
    binding": it is also `False` for a fixed parameter's id, for an
    expression such as `2*k1` and for `nan`/`inf`. None of those are a version
    `_versions` would recognise either, so the target would silently keep its
    model value with no versioned parameter to explain why. The reader raises
    instead, the same way it already raises for a period which names a
    condition the problem does not define.

    The `sbmlsim` extension keeps the exact timecourses it was written with,
    which is what `simulations()` uses when it is there, so the tables are not
    read at all; the extension is dropped here to fall back on them, the path
    a foreign PEtab problem takes.
    """
    petab_problem = petab_v2.Problem.from_yaml(petab_dir / "problem.yaml")
    petab_problem.config.extensions = {}
    condition = next(c for c in petab_problem.conditions if c.changes)
    change = condition.changes[0]
    change.target_value = "not_a_number_and_not_an_estimated_parameter"  # ty: ignore[invalid-assignment]

    reader = PetabReader(petab_problem, base_path=petab_dir)
    assert reader.extension is None
    with pytest.raises(ValueError, match=condition.id):
        reader.simulations()
