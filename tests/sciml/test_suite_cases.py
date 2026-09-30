"""Tests of reading and running a case of the PEtab SciML test suite.

The cases are written by the tests, nothing is downloaded.
"""

import zipfile
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml
from petab_sciml import Input, Layer, NNModel, NNModelStandard, Node

from sbmlsim.fit.petab_v2.likelihood import (
    gradient,
    log_likelihood,
)
from sbmlsim.fit.petab_v2.likelihood import (
    nominal_parameters as nominal_parameter_set,
)
from sbmlsim.fit.petab_v2.reader import DEFAULT_EXPERIMENT, PetabReader
from sbmlsim.sciml.testsuite import (
    GRADIENT_ORDER,
    GRADIENT_STEP,
    SCIML_SUITE_COMMIT,
    CaseStatus,
    InitializationCase,
    ModelImportCase,
    ProblemImportCase,
    SciMLSuite,
    compare_arrays,
    parameter_key,
)
from sbmlsim.testsuite import cache
from tests.sciml.hybrid import feed_forward
from tests.sciml.petab import write_problem

WEIGHT = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
BIAS = np.array([0.1, 0.2, 0.3])


def _model(sid: str = "net0", layer_type: str = "Linear") -> NNModel:
    """Build `tanh(layer1(x))` with 2 inputs and 3 outputs."""
    return NNModel(
        nn_model_id=sid,
        inputs=[Input(input_id="input0")],
        layers=[
            Layer(
                layer_id="layer1",
                layer_type=layer_type,
                args={"in_features": 2, "out_features": 3, "bias": True},
            )
        ],
        forward=[
            Node(
                name="net_input",
                op="placeholder",
                target="net_input",
                args=[],
                kwargs={},
            ),
            Node(
                name="layer1",
                op="call_module",
                target="layer1",
                args=["net_input"],
                kwargs={},
            ),
            Node(
                name="tanh", op="call_method", target="tanh", args=["layer1"], kwargs={}
            ),
            Node(name="output", op="output", target="output", args=["tanh"], kwargs={}),
        ],
    )


def _h5(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an HDF5 file of the suite, the keys are the paths of the arrays."""
    with h5py.File(path, "w") as f:
        f.create_group("metadata")["pytorch_format"] = True
        for key, array in arrays.items():
            f[key] = array


def _model_import_case(
    path: Path, shift: float = 0.0, layer_type: str = "Linear", **solutions: object
) -> Path:
    """Write a case of the group `ml_model_import` with two combinations."""
    directory = path / "ml_model_import" / "001"
    directory.mkdir(parents=True)
    NNModelStandard.save_data(
        data=_model(layer_type=layer_type), filename=str(directory / "net.yaml")
    )
    for i in (1, 2):
        x = np.array([0.5 * i, -0.25])
        _h5(directory / f"net_input_{i}.hdf5", {"inputs/input0/data": x.astype("f4")})
        _h5(
            directory / f"net_ps_{i}.hdf5",
            {
                "parameters/net0/layer1/weight": (i * WEIGHT).astype("f4"),
                "parameters/net0/layer1/bias": BIAS.astype("f4"),
            },
        )
        y = np.tanh(i * WEIGHT @ x + BIAS) + shift
        _h5(
            directory / f"net_output_{i}.hdf5", {"outputs/output0/data": y.astype("f4")}
        )
    content = {
        "net_file": "net.yaml",
        "net_input": ["net_input_1.hdf5", "net_input_2.hdf5"],
        "net_ps": ["net_ps_1.hdf5", "net_ps_2.hdf5"],
        "net_output": ["net_output_1.hdf5", "net_output_2.hdf5"],
        "input_order_py": ["W"],
        "output_order_py": ["W"],
        **solutions,
    }
    (directory / "solutions.yaml").write_text(yaml.safe_dump(content))
    return directory


def _initialization_case(
    path: Path, entity: str, reference: dict[str, np.ndarray]
) -> Path:
    """Write a case of the group `initialization`.

    The array file holds `WEIGHT` and `BIAS`, the parameter table sets the
    entity of the mapping table to zero.
    """
    directory = path / "initialization" / "001"
    petab = directory / "petab"
    petab.mkdir(parents=True)
    NNModelStandard.save_data(
        data=_model(sid="other"), filename=str(petab / "net1.yaml")
    )
    _h5(
        petab / "net1_ps.hdf5",
        {
            "parameters/net1/layer1/weight": WEIGHT,
            "parameters/net1/layer1/bias": BIAS,
        },
    )
    (petab / "problem.yaml").write_text(
        yaml.safe_dump(
            {
                "format_version": "2.0.0",
                "model_files": {"lv": {"location": "lv.xml", "language": "sbml"}},
                "parameter_files": ["parameters.tsv"],
                "mapping_files": ["mapping.tsv"],
                "measurement_files": [],
                "observable_files": [],
                "extensions": {
                    "sciml": {
                        "version": "0.1.0",
                        "required": True,
                        "array_files": ["net1_ps.hdf5"],
                        "hybridization_files": [],
                        "neural_networks": {
                            "net1": {
                                "location": "net1.yaml",
                                "pre_initialization": False,
                                "format": "YAML",
                            }
                        },
                    }
                },
            }
        )
    )
    (petab / "parameters.tsv").write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "alpha\tlin\t0.0\t15.0\t1.3\ttrue\n"
        "net1_part\tlin\t-inf\tinf\t0.0\ttrue\n"
        "net1_ps\tlin\t-inf\tinf\tarray\ttrue\n"
    )
    (petab / "mapping.tsv").write_text(
        "petabEntityId\tmodelEntityId\n"
        "net1_input1\tnet1.inputs[0][0]\n"
        "net1_output1\tnet1.outputs[0][0]\n"
        "net1_ps\tnet1.parameters\n"
        f"net1_part\t{entity}\n"
    )
    _h5(
        directory / "net1_ref.hdf5",
        {f"parameters/net1/layer1/{name}": array for name, array in reference.items()},
    )
    (directory / "solutions.yaml").write_text(
        yaml.safe_dump({"tol": 0.001, "parameter_files": {"net1": "net1_ref.hdf5"}})
    )
    return directory


# ---------------------------------------------------------------------------
# ml_model_import
# ---------------------------------------------------------------------------
def test_a_model_import_case_is_read(tmp_path: Path) -> None:
    """The combinations, the axis orders and the tolerance of a case."""
    case = ModelImportCase.from_directory(_model_import_case(tmp_path))

    assert case.cid == "001"
    assert case.net_file.name == "net.yaml"
    assert [[p.name for p in row] for row in case.inputs] == [
        ["net_input_1.hdf5"],
        ["net_input_2.hdf5"],
    ]
    assert [p.name for p in case.parameters] == ["net_ps_1.hdf5", "net_ps_2.hdf5"]
    assert [p.name for p in case.outputs] == ["net_output_1.hdf5", "net_output_2.hdf5"]
    assert case.input_order == ["W"]
    assert case.output_order == ["W"]
    assert case.dropout is None
    assert case.tolerance == 1e-3


def test_a_case_with_several_inputs(tmp_path: Path) -> None:
    """The inputs of a network with two inputs are listed per argument."""
    directory = _model_import_case(tmp_path)
    solutions = yaml.safe_load((directory / "solutions.yaml").read_text())
    del solutions["net_input"]
    solutions["net_input_arg1"] = ["b_1.hdf5", "b_2.hdf5"]
    solutions["net_input_arg0"] = ["a_1.hdf5", "a_2.hdf5"]
    (directory / "solutions.yaml").write_text(yaml.safe_dump(solutions))

    case = ModelImportCase.from_directory(directory)
    assert [[p.name for p in row] for row in case.inputs] == [
        ["a_1.hdf5", "b_1.hdf5"],
        ["a_2.hdf5", "b_2.hdf5"],
    ]


def test_a_case_with_dropout_has_a_wider_tolerance(tmp_path: Path) -> None:
    """The reference values of a dropout case are a mean of random passes."""
    case = ModelImportCase.from_directory(_model_import_case(tmp_path, dropout=40000))
    assert case.dropout == 40000
    assert case.tolerance == 1e-2


def test_a_model_import_case_passes(tmp_path: Path) -> None:
    """The forward pass reproduces the outputs of every combination."""
    result = ModelImportCase.from_directory(_model_import_case(tmp_path)).run()
    assert result.passed
    assert result.key == "ml_model_import/001"
    assert result.max_difference is not None
    assert result.max_difference < 1e-6


def test_an_output_outside_of_the_tolerance(tmp_path: Path) -> None:
    """A difference above the tolerance fails and is reported."""
    case = ModelImportCase.from_directory(_model_import_case(tmp_path, shift=0.002))
    result = case.run()
    assert result.status == CaseStatus.TOLERANCE
    assert result.max_difference == pytest.approx(0.002, rel=1e-3)
    assert "above the tolerance 0.001" in result.message


def test_a_case_with_a_layer_which_is_not_implemented(tmp_path: Path) -> None:
    """A case does not raise, the outcome names the layer."""
    case = ModelImportCase.from_directory(
        _model_import_case(tmp_path, layer_type="LSTM")
    )
    result = case.run()
    assert result.status == CaseStatus.UNSUPPORTED
    assert "'LSTM'" in result.message


def test_a_case_with_a_missing_file(tmp_path: Path) -> None:
    """A file which does not exist is an error of the case, not of the run."""
    directory = _model_import_case(tmp_path)
    (directory / "net_output_2.hdf5").unlink()
    result = ModelImportCase.from_directory(directory).run()
    assert result.status == CaseStatus.ERROR
    assert "net_output_2.hdf5" in result.message


def test_an_axis_order_which_does_not_fit(tmp_path: Path) -> None:
    """The axis order names the axes of the array."""
    directory = _model_import_case(tmp_path, input_order_py=["C", "W"])
    result = ModelImportCase.from_directory(directory).run()
    assert result.status == CaseStatus.ERROR
    assert "['C', 'W']" in result.message


def test_a_directory_which_is_not_a_case(tmp_path: Path) -> None:
    """A directory without `solutions.yaml` is not read."""
    with pytest.raises(ValueError, match=r"no 'solutions.yaml'"):
        ModelImportCase.from_directory(tmp_path)
    (tmp_path / "solutions.yaml").write_text("net_file: net.yaml\n")
    with pytest.raises(ValueError, match=r"no inputs or no outputs"):
        ModelImportCase.from_directory(tmp_path)


def test_the_comparison_of_arrays() -> None:
    """The shape, the values and the undefined values are compared."""
    a = np.array([1.0, 2.0])
    assert compare_arrays(a, a + 5e-4, 1e-3)[0] == CaseStatus.PASS
    assert compare_arrays(a, a + 2e-3, 1e-3)[0] == CaseStatus.TOLERANCE
    assert compare_arrays(a, a.reshape(1, 2), 1e-3)[0] == CaseStatus.SHAPE
    assert compare_arrays(np.array([1.0, np.nan]), a, 1e-3)[0] == CaseStatus.TOLERANCE
    assert compare_arrays(np.array([]), np.array([]), 1e-3)[0] == CaseStatus.PASS


@pytest.mark.parametrize("expected", [[1.0, np.nan], [np.inf, 2.0]])
def test_a_reference_value_which_is_not_finite(expected: list[float]) -> None:
    """A reference value which is not finite does not pass."""
    status, message, _ = compare_arrays(np.array([1.0, 2.0]), np.array(expected), 1e-3)
    assert status == CaseStatus.TOLERANCE
    assert "not finite" in message


def test_a_case_which_compares_nothing(tmp_path: Path) -> None:
    """A case without a comparison is an error, not a pass."""
    model_import = ModelImportCase(
        cid="001",
        path=tmp_path,
        net_file=tmp_path / "net.yaml",
        inputs=[],
        parameters=[],
        outputs=[],
    )
    directory = _initialization_case(
        tmp_path, "net1.parameters[layer1]", {"weight": WEIGHT, "bias": BIAS}
    )
    initialization = replace(
        InitializationCase.from_directory(directory), parameter_files={}
    )
    for result in (model_import.run(), initialization.run()):
        assert result.status == CaseStatus.ERROR, result
        assert result.message == "nothing was compared"


@pytest.mark.parametrize(
    "change",
    [
        {"net_output": []},
        {"net_input": []},
        {"net_output": ["net_output_1.hdf5"]},
        {"net_ps": ["net_ps_1.hdf5"]},
    ],
)
def test_the_combinations_of_a_case(tmp_path: Path, change: dict) -> None:
    """Inputs, arrays and outputs are listed for every combination."""
    with pytest.raises(ValueError, match=r"001'.*combinations"):
        ModelImportCase.from_directory(_model_import_case(tmp_path, **change))


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("entity", "key"),
    [
        ("net1.parameters", "net1"),
        ("net1.parameters[layer1]", "net1.layer1"),
        ("net1.parameters[layer1].weight", "net1.layer1.weight"),
        ("net1.parameters[block.0].bias", "net1.block.0.bias"),
        ("net1.inputs[0][1]", None),
        ("net1.outputs[0][0]", None),
        ("net1.parametersX", None),
        ("net1.parameters[layer1", None),
        ("net1.parameters[].weight", None),
        ("net1.parameters[layer1].", None),
        (".parameters", None),
        ("alpha", None),
        ("", None),
    ],
)
def test_the_key_of_an_entity(entity: str, key: str | None) -> None:
    """The entities of the mapping table which are parameters of a network."""
    assert parameter_key(entity) == key


@pytest.mark.parametrize(
    ("entity", "reference"),
    [
        ("net1.parameters[layer1]", {"weight": 0 * WEIGHT, "bias": 0 * BIAS}),
        ("net1.parameters[layer1].weight", {"weight": 0 * WEIGHT, "bias": BIAS}),
        ("net1.parameters[layer1].bias", {"weight": WEIGHT, "bias": 0 * BIAS}),
    ],
)
def test_an_initialization_case_passes(
    tmp_path: Path, entity: str, reference: dict[str, np.ndarray]
) -> None:
    """The row of the parameter table replaces the values of the array file."""
    case = InitializationCase.from_directory(
        _initialization_case(tmp_path, entity, reference)
    )
    assert case.tolerance == 0.001
    assert list(case.parameter_files) == ["net1"]

    nominal = case.nominal()
    np.testing.assert_array_equal(
        nominal["net1"]["layer1"]["weight"], reference["weight"]
    )
    np.testing.assert_array_equal(nominal["net1"]["layer1"]["bias"], reference["bias"])
    result = case.run()
    assert result.passed, result.message
    assert result.key == "initialization/001"


def test_an_initialization_which_differs(tmp_path: Path) -> None:
    """A nominal value which is not the reference value fails and is named."""
    directory = _initialization_case(
        tmp_path, "net1.parameters[layer1].bias", {"weight": WEIGHT, "bias": BIAS}
    )
    result = InitializationCase.from_directory(directory).run()
    assert result.status == CaseStatus.TOLERANCE
    assert "'net1.layer1.bias'" in result.message
    assert result.max_difference == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("solutions", "message"),
    [
        ({"parameter_files": {"net1": "ref.hdf5"}}, r"001'.*'tol'"),
        ({"tol": 0.001}, r"001'.*'parameter_files'"),
        ({"tol": 0.001, "parameter_files": {}}, r"001'.*no reference files"),
    ],
)
def test_the_solutions_of_an_initialization_case(
    tmp_path: Path, solutions: dict, message: str
) -> None:
    """A key which is missing is named."""
    directory = tmp_path / "001"
    directory.mkdir()
    (directory / "solutions.yaml").write_text(yaml.safe_dump(solutions))
    with pytest.raises(ValueError, match=message):
        InitializationCase.from_directory(directory)


def test_a_reference_file_without_the_network(tmp_path: Path) -> None:
    """The reference file names the network."""
    directory = _initialization_case(
        tmp_path, "net1.parameters[layer1]", {"weight": WEIGHT, "bias": BIAS}
    )
    _h5(directory / "net1_ref.hdf5", {"parameters/net2/layer1/bias": BIAS})
    result = InitializationCase.from_directory(directory).run()
    assert result.status == CaseStatus.ERROR
    assert "has no arrays of the network 'net1'" in result.message


def test_a_network_in_another_format(tmp_path: Path) -> None:
    """Only the format `YAML` is read."""
    directory = _initialization_case(
        tmp_path, "net1.parameters[layer1]", {"weight": WEIGHT, "bias": BIAS}
    )
    problem = directory / "petab" / "problem.yaml"
    problem.write_text(problem.read_text().replace("format: YAML", "format: equinox"))
    result = InitializationCase.from_directory(directory).run()
    assert result.status == CaseStatus.ERROR
    assert "'equinox' is not supported" in result.message


# ---------------------------------------------------------------------------
# sciml_problem_import
# ---------------------------------------------------------------------------
def test_a_problem_import_case_is_read(tmp_path: Path) -> None:
    """The reference values and the tolerances of a case are read."""
    directory = tmp_path / "sciml_problem_import" / "001"
    directory.mkdir(parents=True)
    (directory / "solutions.yaml").write_text(
        yaml.safe_dump(
            {
                "llh": 33.5,
                "tol_llh": 0.001,
                "tol_simulations": 0.002,
                "tol_grad": 0.1,
                "simulation_files": ["simulations.tsv"],
                "grad_files": {"mech": "grad_mech.tsv", "net1": "grad_net1.hdf5"},
            }
        )
    )
    case = ProblemImportCase.from_directory(directory)
    assert (case.llh, case.log_posterior) == (33.5, None)
    assert (case.tol_llh, case.tol_simulations, case.tol_grad) == (0.001, 0.002, 0.1)
    assert case.problem_path == directory / "petab" / "problem.yaml"
    assert [p.name for p in case.simulation_files] == ["simulations.tsv"]
    assert {k: p.name for k, p in case.gradient_files.items()} == {
        "mech": "grad_mech.tsv",
        "net1": "grad_net1.hdf5",
    }


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        (["tol_llh"], r"'tol_llh' or 'tol_log_posterior'"),
        (["tol_simulations"], r"'tol_simulations'"),
        (["tol_grad"], r"'tol_grad'"),
    ],
)
def test_the_tolerances_of_a_problem_import_case(
    tmp_path: Path, missing: list[str], message: str
) -> None:
    """A tolerance which is missing is named."""
    solutions = {"llh": 1.0, "tol_llh": 0.1, "tol_simulations": 0.1, "tol_grad": 0.1}
    for key in missing:
        del solutions[key]
    directory = tmp_path / "007"
    directory.mkdir()
    (directory / "solutions.yaml").write_text(yaml.safe_dump(solutions))
    with pytest.raises(ValueError, match=rf"007'.*{message}"):
        ProblemImportCase.from_directory(directory)


def test_a_problem_import_case_with_priors(tmp_path: Path) -> None:
    """A case with priors states the log-posterior and its tolerance."""
    (tmp_path / "solutions.yaml").write_text(
        yaml.safe_dump(
            {
                "log_posterior": -12.5,
                "tol_log_posterior": 0.01,
                "tol_simulations": 0.002,
                "tol_grad": 0.1,
                "simulation_files": [],
                "grad_files": {},
            }
        )
    )
    case = ProblemImportCase.from_directory(tmp_path)
    assert (case.llh, case.log_posterior, case.tol_llh) == (None, -12.5, 0.01)


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------
def test_the_suite_iterates_its_groups(tmp_path: Path) -> None:
    """A suite yields the cases of its groups and runs the compared ones."""
    _model_import_case(tmp_path)
    _initialization_case(
        tmp_path, "net1.parameters[layer1]", {"weight": 0 * WEIGHT, "bias": 0 * BIAS}
    )
    (tmp_path / "ml_model_import" / "README.md").write_text("not a case")
    suite = SciMLSuite(path=tmp_path, commit="0" * 40)

    assert suite.case_ids("ml_model_import") == ["001"]
    assert suite.case_ids("sciml_problem_import") == []
    assert [case.cid for case in suite.model_import_cases()] == ["001"]
    assert [case.cid for case in suite.initialization_cases()] == ["001"]
    assert [r.key for r in suite.run()] == ["ml_model_import/001", "initialization/001"]
    assert all(r.passed for r in suite.run())


def test_the_cache_is_per_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A commit is cached under its hash, and can be pointed elsewhere."""
    monkeypatch.delenv("SBMLSIM_SCIML_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache")
    assert SciMLSuite.cache_path() == Path(
        f"/tmp/cache/sbmlsim/petab-sciml-testsuite/{SCIML_SUITE_COMMIT}"
    )
    assert len(SCIML_SUITE_COMMIT) == 40

    monkeypatch.setenv("SBMLSIM_SCIML_SUITE_PATH", "/elsewhere/test_cases")
    assert SciMLSuite.cache_path() == Path("/elsewhere/test_cases")


def test_the_suite_is_loaded_from_the_archive_of_a_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory `test_cases` of the archive becomes the cache."""
    archive = tmp_path / "suite.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("petab_sciml_testsuite-abc/README.md", "suite")
        zf.writestr(
            "petab_sciml_testsuite-abc/test_cases/ml_model_import/001/solutions.yaml",
            "net_file: net.yaml\n",
        )
        zf.writestr(
            "petab_sciml_testsuite-abc/test_cases/initialization/001/solutions.yaml",
            "tol: 0.001\n",
        )
    monkeypatch.delenv("SBMLSIM_SCIML_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("sbmlsim.sciml.testsuite.SCIML_SUITE_URL", archive.as_uri())

    assert SciMLSuite.cached("abc") is None
    suite = SciMLSuite.load("abc")

    assert suite.path == tmp_path / "cache/sbmlsim/petab-sciml-testsuite/abc"
    assert suite.case_ids("ml_model_import") == ["001"]
    assert suite.case_ids("initialization") == ["001"]
    assert SciMLSuite.cached("abc") == suite

    # a fetch which was killed left its staging directory, the next load removes it
    killed = suite.path.parent / ".abc.xyz.incomplete"
    killed.mkdir()
    (killed / cache.LOCK_NAME).touch()
    assert SciMLSuite.load("abc") == suite
    assert not killed.exists()


def test_an_archive_which_is_not_the_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An archive without the groups is an error and is not cached."""
    archive = tmp_path / "suite.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("something/README.md", "not the suite")
    monkeypatch.delenv("SBMLSIM_SCIML_SUITE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("sbmlsim.sciml.testsuite.SCIML_SUITE_URL", archive.as_uri())

    with pytest.raises(OSError, match=r"No directory 'ml_model_import'"):
        SciMLSuite.load("abc")
    assert SciMLSuite.cached("abc") is None


def _problem_import_case(
    tmp_path: Path,
    llh: float | None = None,
    gradient_of: dict[str, float] | None = None,
    log_posterior: float | None = None,
    frozen_layer: bool = False,
    experiment: bool = True,
) -> ProblemImportCase:
    """Write a case of `sciml_problem_import` with the values of `sbmlsim`.

    The reference values are calculated with `sbmlsim` itself, so a case
    which is not changed passes; `llh`, `gradient_of` and `log_posterior`
    replace them. Without `experiment` the measurements name no experiment.
    """
    directory = tmp_path / "sciml_problem_import" / "001"
    network = feed_forward()
    mapping = [
        ("net1_input1", "net1.inputs[0][0]"),
        ("net1_input2", "net1.inputs[0][1]"),
        ("net1_output1", "net1.outputs[0][0]"),
    ]
    parameters = []
    if frozen_layer:
        mapping.append(("net1_layer1", "net1.parameters[layer1]"))
        parameters.append(
            {"parameterId": "net1_layer1", "nominalValue": "array", "estimate": False}
        )
    write_problem(
        directory / "petab",
        networks=[network],
        pre_initialization={"net1": False},
        mapping=mapping,
        hybridization=[
            ("net1_input1", "prey"),
            ("net1_input2", "predator"),
            ("gamma", "net1_output1"),
        ],
        parameters=parameters,
    )
    if not experiment:
        measurements = pd.read_csv(directory / "petab" / "measurements.tsv", sep="\t")
        measurements["experimentId"] = ""
        measurements.to_csv(
            directory / "petab" / "measurements.tsv", sep="\t", index=False
        )
    reader = PetabReader.from_yaml(directory / "petab" / "problem.yaml")
    reader.derived_dir = tmp_path / "derived"
    problem = reader.to_optimization_problem()
    case = ProblemImportCase(
        cid="001",
        path=directory,
        problem_path=directory / "petab" / "problem.yaml",
        llh=llh,
        log_posterior=log_posterior,
        simulation_files=[directory / "simulations.tsv"],
        gradient_files={
            "mech": directory / "grad_mech.tsv",
            "net1": directory / "grad_net1.hdf5",
        },
        tol_llh=1e-3,
        tol_simulations=1e-3,
        tol_grad=0.1,
    )
    problem.initialize(case.settings())
    parameters_nominal = nominal_parameter_set(problem)
    if llh is None and log_posterior is None:
        case = replace(case, llh=log_likelihood(problem, parameters_nominal))
    predictions = problem.predictions(parameters_nominal.x(problem.pids))
    rows = []
    for k, values in predictions.items():
        for time, value in zip(problem.x_references[k], values, strict=True):
            rows.append(
                {
                    "observableId": reader.observable_id(problem.mapping_keys[k]),
                    "experimentId": (
                        ""
                        if problem.simulation_keys[k] == DEFAULT_EXPERIMENT
                        else problem.simulation_keys[k]
                    ),
                    "simulation": value,
                    "time": time,
                }
            )
    pd.DataFrame(rows).to_csv(directory / "simulations.tsv", sep="\t", index=False)
    grad = gradient(
        problem, parameters_nominal, step=GRADIENT_STEP, order=GRADIENT_ORDER
    )
    grad.update(pd.Series(gradient_of or {}))
    pd.DataFrame(
        {
            "parameterId": [p for p in grad.index if not p.startswith("net1__")],
            "value": [grad[p] for p in grad.index if not p.startswith("net1__")],
        }
    ).to_csv(directory / "grad_mech.tsv", sep="\t", index=False)
    arrays = {
        layer: {name: np.zeros_like(array) for name, array in layer_arrays.items()}
        for layer, layer_arrays in network.parameters.items()
    }
    for sid, (layer, name, index) in network.parameter_ids().items():
        if sid in grad.index:
            arrays[layer][name][index] = grad[sid]
        else:
            arrays[layer][name] = np.zeros(0)
    with h5py.File(directory / "grad_net1.hdf5", "w") as f:
        f.create_group("metadata")["pytorch_format"] = True
        for layer, layer_arrays in arrays.items():
            for name, array in layer_arrays.items():
                f[f"parameters/net1/{layer}/{name}"] = array
    return case


def test_a_problem_import_case_passes(tmp_path: Path) -> None:
    """A case whose reference values are the values of `sbmlsim` passes."""
    case = _problem_import_case(tmp_path)
    result = case.run()
    assert result.passed, result.message
    assert result.max_difference is not None
    assert result.max_difference < 1e-3


def test_a_problem_import_case_with_a_frozen_layer(tmp_path: Path) -> None:
    """The gradient of a frozen layer is an empty array without a reference."""
    result = _problem_import_case(tmp_path, frozen_layer=True).run()
    assert result.passed, result.message


def test_a_log_likelihood_outside_of_the_tolerance(tmp_path: Path) -> None:
    """The log-likelihood is compared first and named."""
    result = _problem_import_case(tmp_path, llh=1.0).run()
    assert result.status is CaseStatus.TOLERANCE
    assert result.message.startswith("the log-likelihood:")


def test_a_gradient_outside_of_the_tolerance(tmp_path: Path) -> None:
    """A derivative which differs by more than the tolerance is named."""
    result = _problem_import_case(tmp_path, gradient_of={"alpha": 1e6}).run()
    assert result.status is CaseStatus.TOLERANCE
    assert result.message.startswith("the gradient:")


def test_a_problem_import_case_with_priors_is_unsupported(tmp_path: Path) -> None:
    """A case which states the log-posterior is not compared."""
    result = _problem_import_case(tmp_path, log_posterior=-1.0).run()
    assert result.status is CaseStatus.UNSUPPORTED
    assert "sciml-priors" in result.message


def test_a_problem_import_case_names_what_is_missing(tmp_path: Path) -> None:
    """A reference value without a parameter, and the other way round, is an error."""
    case = _problem_import_case(tmp_path)
    df = pd.read_csv(case.gradient_files["mech"], sep="\t")
    df.loc[len(df)] = ["kappa", 1.0]
    df = df[df["parameterId"] != "beta"]
    df.to_csv(case.gradient_files["mech"], sep="\t", index=False)
    result = case.run()
    assert result.status is CaseStatus.ERROR
    assert "['kappa'] of the reference values are not estimated" in result.message
    assert "['beta'] have no reference value" in result.message


def test_a_problem_import_case_with_a_simulation_of_nothing(tmp_path: Path) -> None:
    """A reference value of the simulations without a fit mapping is an error."""
    case = _problem_import_case(tmp_path)
    df = pd.read_csv(case.simulation_files[0], sep="\t")
    df.loc[len(df)] = ["other", "e1", 1.0, 1.0]
    df.to_csv(case.simulation_files[0], sep="\t", index=False)
    result = case.run()
    assert result.status is CaseStatus.ERROR
    assert "1 of the 21 reference values" in result.message


def test_the_simulations_of_measurements_without_an_experiment(
    tmp_path: Path,
) -> None:
    """A fit mapping without an experiment has the rows without an experiment."""
    case = _problem_import_case(tmp_path, experiment=False)
    assert case.run().passed
    df = pd.read_csv(case.simulation_files[0], sep="\t")
    df.loc[len(df)] = ["prey_o", "e1", 1.0, 1.0]
    df.to_csv(case.simulation_files[0], sep="\t", index=False)
    result = case.run()
    assert result.status is CaseStatus.ERROR
    assert "1 of the 21 reference values" in result.message


def test_the_most_severe_outcome_is_the_result(tmp_path: Path) -> None:
    """An error outranks a log-likelihood outside of the tolerance."""
    case = _problem_import_case(tmp_path, llh=1.0)
    df = pd.read_csv(case.gradient_files["mech"], sep="\t")
    df.loc[len(df)] = ["kappa", 1.0]
    df.to_csv(case.gradient_files["mech"], sep="\t", index=False)
    result = case.run()
    assert result.status is CaseStatus.ERROR
    assert "['kappa'] of the reference values are not estimated" in result.message


def test_a_problem_import_case_which_cannot_be_read(tmp_path: Path) -> None:
    """A problem which is not read is an error which names the reason."""
    case = _problem_import_case(tmp_path)
    (case.path / "petab" / "net1.yaml").unlink()
    result = case.run()
    assert result.status is CaseStatus.ERROR
    assert "net1.yaml" in result.message
    assert "does not exist" in result.message
