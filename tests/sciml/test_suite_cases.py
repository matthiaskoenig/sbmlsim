"""Tests of reading and running a case of the PEtab SciML test suite.

The cases are written by the tests, nothing is downloaded.
"""

import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from petab_sciml import Input, Layer, NNModel, NNModelStandard, Node

from sbmlsim.sciml.testsuite import (
    SCIML_SUITE_COMMIT,
    CaseStatus,
    InitializationCase,
    ModelImportCase,
    ProblemImportCase,
    SciMLSuite,
    compare_arrays,
    parameter_key,
)

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
