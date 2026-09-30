"""The cases of the PEtab SciML test suite on disk.

The [test suite](https://github.com/PEtab-dev/petab_sciml_testsuite) has
three groups of cases, every case is a directory named by its number with a
`solutions.yaml`:

| group | case | compared |
| --- | --- | --- |
| `ml_model_import` | `ModelImportCase` | the outputs of the forward pass |
| `initialization` | `InitializationCase` | the nominal values of the arrays |
| `sciml_problem_import` | `ProblemImportCase` | likelihood, simulations, gradient |

The suite has no releases, so it is pinned by a commit. `SciMLSuite` is the
directory of the groups with the download and the cache of the commit.

The arrays of the suite are in the PyTorch layout, which is the layout of
`sbmlsim`: the axis orders `input_order_py` and `output_order_py` of a case
name the axes of the arrays as they are stored, nothing is permuted.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from petab.v2.core import MappingTable, ParameterTable, ProblemConfig
from petab.v2.extensions.sciml import SciMLConfig
from petab_sciml.constants import ARRAY

from sbmlsim.fit.options import FitSettings, ParameterScaleType
from sbmlsim.fit.petab_v2.likelihood import (
    gradient,
    log_likelihood,
)
from sbmlsim.fit.petab_v2.likelihood import (
    nominal_parameters as nominal_parameter_set,
)
from sbmlsim.fit.petab_v2.reader import DEFAULT_EXPERIMENT, PetabReader
from sbmlsim.fit.petab_v2.sciml import SciMLProblemError
from sbmlsim.sciml.errors import NetworkImportError, UnsupportedLayerError
from sbmlsim.sciml.network import (
    Network,
    NetworkParameters,
    element_id,
    load_array_data,
)
from sbmlsim.sciml.parameters import nominal_parameters
from sbmlsim.testsuite import cache

logger = logging.getLogger(__name__)

#: commit of the test suite the tests run against
SCIML_SUITE_COMMIT = "0622bbfc5e12eb9b482659eabd1756ca0e87dfc8"

#: archive of a commit of the suite
SCIML_SUITE_URL = (
    "https://github.com/PEtab-dev/petab_sciml_testsuite/archive/{commit}.zip"
)

#: the environment variable which points at the cases when they are not in
#: the cache, i.e. at the directory `test_cases` of a checkout
SCIML_SUITE_PATH_VARIABLE = "SBMLSIM_SCIML_SUITE_PATH"

#: the groups of cases, which are the directories of the suite
MODEL_IMPORT = "ml_model_import"
INITIALIZATION = "initialization"
PROBLEM_IMPORT = "sciml_problem_import"

#: absolute tolerance of the outputs of a network. The cases of the group
#: `ml_model_import` do not state one, these are the tolerances the suite
#: checks its own reference values with (`pysrc/ml_import_helper.py`)
MODEL_IMPORT_TOLERANCE = 1e-3

#: absolute tolerance of a case with dropout, whose reference values are the
#: mean of forward passes in training mode
DROPOUT_TOLERANCE = 1e-2

#: absolute and relative tolerance of the integrator in the cases of the
#: group `sciml_problem_import`. The reference values are simulated with the
#: tolerances `1e-12`, and the error of a simulation enters the gradient
#: multiplied by the sensitivities of the log-likelihood, which are of the
#: order `1e4`
PROBLEM_IMPORT_TOLERANCE = 1e-13

#: relative step and order of the differences of the gradient. The reference
#: values are central differences of five points
GRADIENT_STEP = 1e-6
GRADIENT_ORDER = 4

#: the key of the gradient of the parameters which are not elements of a
#: network in the `grad_files` of a case
MECHANISTIC = "mech"


class CaseStatus(StrEnum):
    """The outcome of a case."""

    PASS = "pass"
    TOLERANCE = "tolerance"
    SHAPE = "shape"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


#: the outcomes of a comparison from the least to the most severe
SEVERITY: tuple[CaseStatus, ...] = (
    CaseStatus.PASS,
    CaseStatus.TOLERANCE,
    CaseStatus.SHAPE,
    CaseStatus.UNSUPPORTED,
    CaseStatus.ERROR,
)


@dataclass(frozen=True)
class CaseResult:
    """The outcome of a case.

    Attributes:
        group: the group of the case.
        cid: the number of the case, e.g. `001`.
        status: the outcome.
        message: what failed, empty for a case which passes.
        max_difference: the largest absolute difference to the reference
            values, `None` when nothing was compared.
    """

    group: str
    cid: str
    status: CaseStatus
    message: str = ""
    max_difference: float | None = None

    @property
    def passed(self) -> bool:
        """Check whether the case passes."""
        return self.status == CaseStatus.PASS

    @property
    def key(self) -> str:
        """Get the key of the case in the baseline, e.g. `ml_model_import/001`."""
        return f"{self.group}/{self.cid}"


def read_solutions(path: Path) -> dict[str, Any]:
    """Read the `solutions.yaml` of a case.

    Args:
        path: the directory of the case.

    Returns:
        The content of the file.

    Raises:
        ValueError: if the directory has no `solutions.yaml` or the file is
            not a mapping.
    """
    solutions_path = path / "solutions.yaml"
    if not solutions_path.is_file():
        raise ValueError(f"The case '{path}' has no 'solutions.yaml'")
    solutions = yaml.safe_load(solutions_path.read_text(encoding="utf-8"))
    if not isinstance(solutions, dict):
        raise ValueError(f"'{solutions_path}' is not a mapping")
    return solutions


def required(solutions: dict[str, Any], key: str, path: Path) -> Any:
    """Get a value of the `solutions.yaml` of a case which must be there.

    Args:
        solutions: the content of the file.
        key: the key of the value.
        path: the directory of the case, for the message.

    Returns:
        The value.

    Raises:
        ValueError: if the file has no value for the key.
    """
    value = solutions.get(key)
    if value is None:
        raise ValueError(f"The case '{path}' has no '{key}' in 'solutions.yaml'")
    return value


def read_array(path: Path, group: str) -> np.ndarray:
    """Read the single array of an input or output file of a case.

    Args:
        path: the HDF5 file.
        group: `inputs` or `outputs`.

    Returns:
        The array in the PyTorch layout, in double precision.

    Raises:
        ValueError: if the file does not hold exactly one array in the group
            or is not in the PyTorch layout.
    """
    arrays: list[np.ndarray] = []
    pytorch_format: list[bool] = []

    def collect(name: str, item: object) -> None:
        if not isinstance(item, h5py.Dataset):
            return
        if name == "metadata/pytorch_format":
            pytorch_format.append(bool(item[()]))
        elif name.startswith(f"{group}/"):
            arrays.append(np.asarray(item[()], dtype=float))

    with h5py.File(path, "r") as f:
        f.visititems(collect)
    if pytorch_format != [True]:
        raise ValueError(f"'{path}' is not in the PyTorch layout")
    if len(arrays) != 1:
        raise ValueError(f"'{path}' holds {len(arrays)} arrays in '{group}', not one")
    return arrays[0]


def compare_arrays(
    observed: np.ndarray, expected: np.ndarray, tolerance: float
) -> tuple[CaseStatus, str, float | None]:
    """Compare an array with its reference values.

    Args:
        observed: the values of `sbmlsim`.
        expected: the reference values.
        tolerance: the absolute tolerance.

    Returns:
        The outcome, the message and the largest absolute difference, `None`
        when the shapes differ or a value is not finite.
    """
    if observed.shape != expected.shape:
        return (
            CaseStatus.SHAPE,
            f"the shape is {observed.shape}, expected {expected.shape}",
            None,
        )
    for name, values in (("the values", observed), ("the reference values", expected)):
        if not np.all(np.isfinite(values)):
            return CaseStatus.TOLERANCE, f"{name} are not finite", None
    if observed.size == 0:
        return CaseStatus.PASS, "", 0.0
    difference = float(np.max(np.abs(observed - expected)))
    if not difference <= tolerance:
        return (
            CaseStatus.TOLERANCE,
            f"the largest difference {difference:.3g} is above the tolerance "
            f"{tolerance:.3g}",
            difference,
        )
    return CaseStatus.PASS, "", difference


def _worst(
    group: str, cid: str, outcomes: list[tuple[CaseStatus, str, float | None]]
) -> CaseResult:
    """Get the result of a case from the outcomes of its comparisons.

    Args:
        group: the group of the case.
        cid: the number of the case.
        outcomes: outcome, message and largest difference of every comparison.

    Returns:
        The first of the most severe comparisons which fail, see `SEVERITY`,
        or the pass with the largest difference of all comparisons. A case
        without a comparison is an `ERROR`, it would pass without checking
        anything.
    """
    if not outcomes:
        return CaseResult(group, cid, CaseStatus.ERROR, "nothing was compared")
    status, message, difference = max(
        outcomes, key=lambda outcome: SEVERITY.index(outcome[0])
    )
    if status != CaseStatus.PASS:
        return CaseResult(group, cid, status, message, difference)
    differences = [d for _, _, d in outcomes if d is not None]
    return CaseResult(
        group, cid, CaseStatus.PASS, "", max(differences) if differences else None
    )


@dataclass(frozen=True)
class ModelImportCase:
    """A case of the group `ml_model_import`.

    A case is a network with combinations of inputs, arrays and the outputs
    the network has for them. The combination `i` is the input `i`, the
    arrays `i` and the output `i`.

    Attributes:
        cid: the number of the case, e.g. `001`.
        path: the directory of the case.
        net_file: the NN YAML.
        inputs: the input files of every combination, one per input of the
            network.
        parameters: the array file of every combination, empty for a network
            without arrays.
        outputs: the output file of every combination.
        input_order: the axes of the inputs, e.g. `["C", "H", "W"]`.
        output_order: the axes of the outputs.
        dropout: the number of forward passes in training mode the reference
            values are the mean of, `None` for a case without dropout.
    """

    cid: str
    path: Path
    net_file: Path
    inputs: list[list[Path]]
    parameters: list[Path]
    outputs: list[Path]
    input_order: list[str] = field(default_factory=list)
    output_order: list[str] = field(default_factory=list)
    dropout: int | None = None

    @classmethod
    def from_directory(cls, path: Path) -> ModelImportCase:
        """Read a case from its directory.

        Args:
            path: the directory of the case, named by its number.

        Returns:
            The case.

        Raises:
            ValueError: if the directory has no `solutions.yaml`, if the file
                does not list inputs and outputs, or if the inputs, the array
                files and the outputs are not listed for the same non-empty
                list of combinations.
        """
        solutions = read_solutions(path)
        if "net_input" in solutions:
            columns = [solutions["net_input"]]
        else:
            keys = sorted(
                (key for key in solutions if key.startswith("net_input_arg")),
                key=lambda key: int(key.removeprefix("net_input_arg")),
            )
            columns = [solutions[key] for key in keys]
        if not columns or "net_output" not in solutions:
            raise ValueError(f"The case '{path}' lists no inputs or no outputs")
        outputs = [path / name for name in solutions["net_output"]]
        parameters = [path / name for name in solutions.get("net_ps", [])]
        lengths = {len(column) for column in columns}
        if (
            not outputs
            or lengths != {len(outputs)}
            or (parameters and len(parameters) != len(outputs))
        ):
            raise ValueError(
                f"The case '{path}' lists {sorted(lengths)} inputs per argument, "
                f"{len(parameters)} array files and {len(outputs)} outputs, the "
                f"combinations must agree and not be empty"
            )
        return cls(
            cid=path.name,
            path=path,
            net_file=path / solutions.get("net_file", "net.yaml"),
            inputs=[
                [path / name for name in row] for row in zip(*columns, strict=True)
            ],
            parameters=parameters,
            outputs=outputs,
            input_order=list(solutions.get("input_order_py", [])),
            output_order=list(solutions.get("output_order_py", [])),
            dropout=solutions.get("dropout"),
        )

    @property
    def tolerance(self) -> float:
        """Get the absolute tolerance of the outputs."""
        return MODEL_IMPORT_TOLERANCE if self.dropout is None else DROPOUT_TOLERANCE

    def network(self, i: int) -> Network:
        """Get the network with the arrays of a combination.

        Args:
            i: the index of the combination.

        Returns:
            The network.
        """
        array_path = self.parameters[i] if self.parameters else None
        return Network.from_files(self.net_file, array_path)

    def run(self) -> CaseResult:
        """Evaluate the network for every combination and compare the outputs.

        Returns:
            The outcome of the case. It does not raise: a layer without an
            implementation is `UNSUPPORTED` and any other error is `ERROR`.
        """
        outcomes: list[tuple[CaseStatus, str, float | None]] = []
        try:
            for i, output_path in enumerate(self.outputs):
                inputs = [read_array(path, "inputs") for path in self.inputs[i]]
                expected = read_array(output_path, "outputs")
                for order, array in (
                    (self.input_order, inputs[0]),
                    (self.output_order, expected),
                ):
                    if order and len(order) != array.ndim:
                        raise ValueError(
                            f"an array with {array.ndim} axes does not have "
                            f"the axes {order}"
                        )
                (observed,) = self.network(i).forward(*inputs)
                outcomes.append(compare_arrays(observed, expected, self.tolerance))
        except UnsupportedLayerError as err:
            return CaseResult(MODEL_IMPORT, self.cid, CaseStatus.UNSUPPORTED, str(err))
        except Exception as err:
            return CaseResult(MODEL_IMPORT, self.cid, CaseStatus.ERROR, str(err))
        return _worst(MODEL_IMPORT, self.cid, outcomes)


def parameter_key(model_entity_id: str) -> str | None:
    """Get the key of an entry from a `modelEntityId` of the mapping table.

    Args:
        model_entity_id: the entity, e.g. `net1.parameters[layer1].weight`.

    Returns:
        The key of `sbmlsim.sciml.parameters`, i.e. `net1` for
        `net1.parameters`, `net1.layer1` for `net1.parameters[layer1]` and
        `net1.layer1.weight` for `net1.parameters[layer1].weight`. `None` for
        an entity which is not the parameters of a network.
    """
    network, separator, rest = model_entity_id.partition(".parameters")
    if not separator or not network:
        return None
    if not rest:
        return network
    if not rest.startswith("["):
        return None
    layer, closing, array = rest[1:].partition("]")
    if not closing or not layer:
        return None
    if not array:
        return f"{network}.{layer}"
    if not array.startswith(".") or len(array) == 1:
        return None
    return f"{network}.{layer}{array}"


@dataclass(frozen=True)
class InitializationCase:
    """A case of the group `initialization`.

    A case is a PEtab SciML problem whose parameter table sets the nominal
    values of a part of a network, and the arrays the network has after the
    import.

    Attributes:
        cid: the number of the case, e.g. `001`.
        path: the directory of the case.
        problem_path: the YAML of the problem.
        tolerance: the absolute tolerance of the arrays.
        parameter_files: id of the network -> the file with its reference
            values.
    """

    cid: str
    path: Path
    problem_path: Path
    tolerance: float
    parameter_files: dict[str, Path]

    @classmethod
    def from_directory(cls, path: Path) -> InitializationCase:
        """Read a case from its directory.

        Args:
            path: the directory of the case, named by its number.

        Returns:
            The case.

        Raises:
            ValueError: if the directory has no `solutions.yaml`, or if the
                file has no tolerance or no reference files.
        """
        solutions = read_solutions(path)
        tolerance = float(required(solutions, "tol", path))
        files = required(solutions, "parameter_files", path)
        if not files:
            raise ValueError(f"The case '{path}' lists no reference files")
        return cls(
            cid=path.name,
            path=path,
            problem_path=path / "petab" / "problem.yaml",
            tolerance=tolerance,
            parameter_files={network: path / name for network, name in files.items()},
        )

    def nominal(self) -> dict[str, NetworkParameters]:
        """Import the networks of the problem with their nominal values.

        The problem is read with the classes of `petab` which need no
        `torch`: the configuration, the parameter table and the mapping
        table. The rows of the parameter table which the mapping table
        resolves to the parameters of a network are the entries of
        `nominal_parameters`; the `nominalValue` `array` keeps the values of
        the array file.

        Returns:
            id of the network -> the arrays of the network.

        Raises:
            NetworkImportError: if the problem is not a PEtab SciML problem,
                a network is not in the format `YAML` or an array has no
                values.
        """
        base = self.problem_path.parent
        raw = yaml.safe_load(self.problem_path.read_text(encoding="utf-8"))
        config = ProblemConfig(**raw, base_path=base)
        sciml = (config.extensions or {}).get("sciml")
        if not isinstance(sciml, SciMLConfig):
            raise NetworkImportError(f"'{self.problem_path}' has no 'sciml' extension")

        keys: dict[str, str] = {}
        for mapping_file in config.mapping_files:
            table = MappingTable.from_tsv(base / str(mapping_file))
            for mapping in table.elements:
                key = parameter_key(mapping.model_id or "")
                if key is not None:
                    keys[mapping.petab_id] = key
        values: dict[str, float] = {}
        for parameter_file in config.parameter_files:
            for parameter in ParameterTable.from_tsv(
                base / str(parameter_file)
            ).elements:
                value = parameter.nominal_value
                if parameter.id in keys and value is not None and value != ARRAY:
                    values[keys[parameter.id]] = float(value)

        nominal: dict[str, NetworkParameters] = {}
        for sid, network_config in (sciml.neural_networks or {}).items():
            if network_config.format.lower() != "yaml":
                raise NetworkImportError(
                    f"Network '{sid}': the format '{network_config.format}' "
                    f"is not supported, only 'YAML'"
                )
            network = Network.from_files(base / str(network_config.location), sid=sid)
            for array_file in sciml.array_files:
                array_path = base / str(array_file)
                if sid in load_array_data(array_path).parameters:
                    network = replace(
                        network, parameters=network.read_arrays(array_path)
                    )
            nominal[sid] = nominal_parameters(
                network,
                {
                    key: value
                    for key, value in values.items()
                    if key == sid or key.startswith(f"{sid}.")
                },
            )
        return nominal

    def expected(self) -> dict[str, NetworkParameters]:
        """Read the reference values of the arrays.

        Returns:
            id of the network -> the arrays of the network.

        Raises:
            ValueError: if a reference file has no arrays of its network.
        """
        expected: dict[str, NetworkParameters] = {}
        for sid, path in self.parameter_files.items():
            data = load_array_data(path)
            if sid not in data.parameters:
                raise ValueError(
                    f"The reference file '{path}' has no arrays of the network "
                    f"'{sid}', it has {sorted(data.parameters)}"
                )
            expected[sid] = {
                layer: {
                    name: np.asarray(array, dtype=float)
                    for name, array in arrays.items()
                }
                for layer, arrays in data.parameters[sid].items()
            }
        return expected

    def run(self) -> CaseResult:
        """Import the networks and compare their nominal values.

        Returns:
            The outcome of the case. It does not raise: a layer without an
            implementation is `UNSUPPORTED` and any other error is `ERROR`.
        """
        outcomes: list[tuple[CaseStatus, str, float | None]] = []
        try:
            nominal = self.nominal()
            for sid, layers in self.expected().items():
                for layer, arrays in layers.items():
                    for name, expected in arrays.items():
                        observed = nominal.get(sid, {}).get(layer, {}).get(name)
                        if observed is None:
                            outcomes.append(
                                (
                                    CaseStatus.ERROR,
                                    f"the array '{sid}.{layer}.{name}' was "
                                    f"not imported",
                                    None,
                                )
                            )
                            continue
                        status, message, difference = compare_arrays(
                            observed, expected, self.tolerance
                        )
                        if message:
                            message = f"'{sid}.{layer}.{name}': {message}"
                        outcomes.append((status, message, difference))
        except UnsupportedLayerError as err:
            return CaseResult(
                INITIALIZATION, self.cid, CaseStatus.UNSUPPORTED, str(err)
            )
        except Exception as err:
            return CaseResult(INITIALIZATION, self.cid, CaseStatus.ERROR, str(err))
        return _worst(INITIALIZATION, self.cid, outcomes)


@dataclass(frozen=True)
class ProblemImportCase:
    """A case of the group `sciml_problem_import`.

    A case is a PEtab SciML problem with the log-likelihood, the simulations
    at the measurements and the gradient of the log-likelihood at the nominal
    values of its parameters. The reference values of the gradient are the
    central differences of five points of a simulation with the tolerances
    `1e-12`.

    Attributes:
        cid: the number of the case, e.g. `001`.
        path: the directory of the case.
        problem_path: the YAML of the problem.
        llh: the log-likelihood at the nominal values, `None` for a case with
            priors, which states `log_posterior`.
        log_posterior: the log-posterior at the nominal values, `None` for a
            case without priors.
        simulation_files: the simulations at the measurement points.
        gradient_files: `mech` or the id of a network -> the gradient of the
            mechanistic parameters (TSV) or of the network (HDF5).
        tol_llh: tolerance of the log-likelihood or the log-posterior.
        tol_simulations: tolerance of the simulations.
        tol_grad: tolerance of the gradient.
    """

    cid: str
    path: Path
    problem_path: Path
    llh: float | None
    log_posterior: float | None
    simulation_files: list[Path]
    gradient_files: dict[str, Path]
    tol_llh: float
    tol_simulations: float
    tol_grad: float

    @classmethod
    def from_directory(cls, path: Path) -> ProblemImportCase:
        """Read a case from its directory.

        Args:
            path: the directory of the case, named by its number.

        Returns:
            The case.

        Raises:
            ValueError: if the directory has no `solutions.yaml`, or if the
                file has no tolerance of the likelihood, the simulations or
                the gradient.
        """
        solutions = read_solutions(path)
        tol_llh = solutions.get("tol_llh", solutions.get("tol_log_posterior"))
        if tol_llh is None:
            raise ValueError(
                f"The case '{path}' has no 'tol_llh' or 'tol_log_posterior' in "
                f"'solutions.yaml'"
            )
        llh = solutions.get("llh")
        log_posterior = solutions.get("log_posterior")
        return cls(
            cid=path.name,
            path=path,
            problem_path=path / "petab" / "problem.yaml",
            llh=None if llh is None else float(llh),
            log_posterior=None if log_posterior is None else float(log_posterior),
            simulation_files=[
                path / name for name in solutions.get("simulation_files", [])
            ],
            gradient_files={
                key: path / name
                for key, name in solutions.get("grad_files", {}).items()
            },
            tol_llh=float(tol_llh),
            tol_simulations=float(required(solutions, "tol_simulations", path)),
            tol_grad=float(required(solutions, "tol_grad", path)),
        )

    def settings(self) -> FitSettings:
        """Get the settings the problem of the case is initialized with.

        Returns:
            Settings with the linear scale, a fixed grid and the tolerances
            `PROBLEM_IMPORT_TOLERANCE`: two simulations on a variable grid
            differ by more than a difference of the gradient resolves.
        """
        return FitSettings(
            parameter_scale=ParameterScaleType.LINEAR,
            variable_step_size=False,
            absolute_tolerance=PROBLEM_IMPORT_TOLERANCE,
            relative_tolerance=PROBLEM_IMPORT_TOLERANCE,
        )

    def expected_simulations(self) -> pd.DataFrame:
        """Read the reference values of the simulations.

        Returns:
            The rows of the simulation files with the columns `observableId`,
            `experimentId`, `time` and `simulation`.

        Raises:
            ValueError: if the case has no simulation file, or if a file lacks
                a column.
        """
        if not self.simulation_files:
            raise ValueError(f"The case '{self.path}' lists no simulation files")
        frames = [
            pd.read_csv(path, sep="\t", float_precision="round_trip")
            for path in self.simulation_files
        ]
        df = pd.concat(frames, ignore_index=True)
        missing = sorted(
            {"observableId", "experimentId", "time", "simulation"} - set(df.columns)
        )
        if missing:
            raise ValueError(
                f"The simulation files of the case '{self.path}' have no columns "
                f"{missing}"
            )
        return df

    def expected_gradient(self) -> dict[str, float]:
        """Read the reference values of the gradient.

        Returns:
            id of the parameter -> derivative of the log-likelihood. The id of
            an element of a network is its id in `sbmlsim`, see
            `sbmlsim.sciml.network.element_id`. An array which the file
            stores as an empty array is frozen and has no derivative.

        Raises:
            ValueError: if the case has no gradient of the mechanistic
                parameters, or if a file cannot be read.
        """
        if MECHANISTIC not in self.gradient_files:
            raise ValueError(
                f"The case '{self.path}' has no gradient of the mechanistic "
                f"parameters, the key '{MECHANISTIC}' of 'grad_files'"
            )
        expected: dict[str, float] = {}
        for key, path in self.gradient_files.items():
            if key == MECHANISTIC:
                df = pd.read_csv(path, sep="\t", float_precision="round_trip")
                for pid, value in zip(df["parameterId"], df["value"], strict=True):
                    expected[str(pid)] = float(value)
                continue
            data = load_array_data(path)
            if key not in data.parameters:
                raise ValueError(
                    f"The gradient file '{path}' has no arrays of the network "
                    f"'{key}', it has {sorted(data.parameters)}"
                )
            for layer, arrays in data.parameters[key].items():
                for name, values in arrays.items():
                    array = np.asarray(values, dtype=float)
                    for index in np.ndindex(array.shape):
                        expected[element_id(key, layer, name, index)] = float(
                            array[index]
                        )
        return expected

    def run(self) -> CaseResult:
        """Read the problem and compare its values with the reference values.

        The log-likelihood, the simulations at the measurements and the
        gradient of the log-likelihood are compared, each with its tolerance.
        The models the problem is simulated with are written into a
        temporary directory.

        Returns:
            The outcome of the case. It does not raise: a problem with a gap
            or a layer without an implementation is `UNSUPPORTED` and any
            other error is `ERROR`.
        """
        if self.llh is None:
            return CaseResult(
                PROBLEM_IMPORT,
                self.cid,
                CaseStatus.UNSUPPORTED,
                "the case states the log-posterior of a problem with priors, "
                "which are not a part of the log-likelihood (gap 'sciml-priors')",
            )
        try:
            with tempfile.TemporaryDirectory(prefix="sbmlsim-sciml-") as directory:
                outcomes = self._compare(Path(directory))
        except (UnsupportedLayerError, SciMLProblemError) as err:
            if isinstance(err, SciMLProblemError) and err.gap is None:
                return CaseResult(PROBLEM_IMPORT, self.cid, CaseStatus.ERROR, str(err))
            return CaseResult(
                PROBLEM_IMPORT, self.cid, CaseStatus.UNSUPPORTED, str(err)
            )
        except Exception as err:
            return CaseResult(
                PROBLEM_IMPORT,
                self.cid,
                CaseStatus.ERROR,
                f"{type(err).__name__}: {err}",
            )
        return _worst(PROBLEM_IMPORT, self.cid, outcomes)

    def _compare(self, directory: Path) -> list[tuple[CaseStatus, str, float | None]]:
        """Compare the values of the problem with the reference values.

        Args:
            directory: the directory the models of the problem are written to.

        Returns:
            Outcome, message and largest difference of every comparison.
        """
        reader = PetabReader.from_yaml(self.problem_path)
        reader.derived_dir = directory
        problem = reader.to_optimization_problem(opid=f"case_{self.cid}")
        problem.initialize(self.settings())
        parameters = nominal_parameter_set(problem)

        def outcome(
            what: str, observed: Any, expected: Any, tolerance: float
        ) -> tuple[CaseStatus, str, float | None]:
            status, message, difference = compare_arrays(
                np.asarray(observed, dtype=float),
                np.asarray(expected, dtype=float),
                tolerance,
            )
            return status, f"{what}: {message}" if message else "", difference

        outcomes = [
            outcome(
                "the log-likelihood",
                log_likelihood(problem, parameters),
                self.llh,
                self.tol_llh,
            )
        ]

        expected = self.expected_simulations()
        predictions = problem.predictions(parameters.x(problem.pids))
        # a measurement without an experiment has a row without one
        experiment_ids = expected["experimentId"].fillna("").astype(str)
        matched: set[int] = set()
        for k, prediction in predictions.items():
            key = problem.mapping_keys[k]
            observable_id = reader.observable_id(key)
            experiment_id = problem.simulation_keys[k]
            selected = (expected["observableId"] == observable_id) & (
                experiment_ids == experiment_id
            )
            if experiment_id == DEFAULT_EXPERIMENT:
                selected |= (expected["observableId"] == observable_id) & (
                    experiment_ids == ""
                )
            rows = expected[selected].sort_values("time")
            what = f"the simulations of '{observable_id}' in '{experiment_id}'"
            twice = sorted(matched & set(rows.index))
            if twice:
                outcomes.append(
                    (
                        CaseStatus.ERROR,
                        f"{what}: the reference values {twice} belong to "
                        f"another fit mapping as well",
                        None,
                    )
                )
            matched |= set(rows.index)
            outcomes.append(
                outcome(
                    f"{what}, times",
                    problem.x_references[k],
                    rows["time"].to_numpy(),
                    0.0,
                )
            )
            outcomes.append(
                outcome(
                    what,
                    prediction,
                    rows["simulation"].to_numpy(),
                    self.tol_simulations,
                )
            )
        if len(matched) != len(expected):
            outcomes.append(
                (
                    CaseStatus.ERROR,
                    f"{len(expected) - len(matched)} of the {len(expected)} "
                    f"reference values of the simulations belong to no fit mapping",
                    None,
                )
            )

        observed = gradient(
            problem, parameters, step=GRADIENT_STEP, order=GRADIENT_ORDER
        )
        reference = self.expected_gradient()
        missing = sorted(set(reference) - set(observed.index))
        extra = sorted(set(observed.index) - set(reference))
        if missing or extra:
            outcomes.append(
                (
                    CaseStatus.ERROR,
                    f"the gradient: the parameters {missing} of the reference "
                    f"values are not estimated, and the estimated parameters "
                    f"{extra} have no reference value",
                    None,
                )
            )
        shared = [pid for pid in observed.index if pid in reference]
        outcomes.append(
            outcome(
                "the gradient",
                [observed[pid] for pid in shared],
                [reference[pid] for pid in shared],
                self.tol_grad,
            )
        )
        return outcomes


@dataclass(frozen=True)
class SciMLSuite:
    """The cases of a commit of the PEtab SciML test suite.

    Attributes:
        path: the directory which holds the directories of the groups, i.e.
            `test_cases` of the suite.
        commit: the commit of the suite.
    """

    path: Path
    commit: str

    @staticmethod
    def cache_path(commit: str = SCIML_SUITE_COMMIT) -> Path:
        """Get the directory a commit of the suite is unpacked into.

        `SBMLSIM_SCIML_SUITE_PATH` overrides it. Otherwise it is
        `sbmlsim/petab-sciml-testsuite/<commit>` in the user cache.

        Args:
            commit: the commit of the suite.

        Returns:
            The directory the groups of the commit live in.
        """
        return cache.cache_path(
            SCIML_SUITE_PATH_VARIABLE, "petab-sciml-testsuite", commit
        )

    @classmethod
    def cached(cls, commit: str = SCIML_SUITE_COMMIT) -> SciMLSuite | None:
        """Get a commit of the suite if it is already on this machine.

        Args:
            commit: the commit of the suite.

        Returns:
            The suite, or `None` if it was not downloaded yet.
        """
        path = cls.cache_path(commit)
        return cls(path=path, commit=commit) if path.is_dir() else None

    @classmethod
    def load(cls, commit: str = SCIML_SUITE_COMMIT) -> SciMLSuite:
        """Get a commit of the suite, downloading it if it is not cached.

        Args:
            commit: the commit of the suite.

        Returns:
            The suite with its cases unpacked in the cache.

        Raises:
            OSError: if the commit cannot be downloaded.
        """
        suite = cls.cached(commit)
        if suite is not None:
            # what a fetch which was killed left next to the cache
            cache.remove_stale(suite.path)
            return suite
        path = cls.cache_path(commit)
        url = SCIML_SUITE_URL.format(commit=commit)
        logger.info("Downloading the PEtab SciML test suite '%s'", commit)
        cache.fetch(url, path, select=cls._cases_dir)
        return cls(path=path, commit=commit)

    @staticmethod
    def _cases_dir(staging: Path) -> Path:
        """Get the directory of the groups of an unpacked archive.

        Args:
            staging: directory the archive was unpacked into.

        Returns:
            The directory which holds the directory `ml_model_import`.

        Raises:
            OSError: if the archive holds no such directory.
        """
        for candidate in sorted(staging.glob(f"**/{MODEL_IMPORT}")):
            if candidate.is_dir():
                return candidate.parent
        raise OSError(
            f"No directory '{MODEL_IMPORT}' in the unpacked suite '{staging}'"
        )

    def case_ids(self, group: str) -> list[str]:
        """Get the numbers of the cases of a group.

        Args:
            group: the group, i.e. the name of its directory.

        Returns:
            The names of the case directories, sorted, empty for a group the
            suite does not have.
        """
        directory = self.path / group
        if not directory.is_dir():
            return []
        return sorted(
            p.name for p in directory.iterdir() if p.is_dir() and p.name.isdigit()
        )

    def model_import_cases(self) -> Iterator[ModelImportCase]:
        """Iterate the cases of the group `ml_model_import`."""
        for cid in self.case_ids(MODEL_IMPORT):
            yield ModelImportCase.from_directory(self.path / MODEL_IMPORT / cid)

    def initialization_cases(self) -> Iterator[InitializationCase]:
        """Iterate the cases of the group `initialization`."""
        for cid in self.case_ids(INITIALIZATION):
            yield InitializationCase.from_directory(self.path / INITIALIZATION / cid)

    def problem_import_cases(self) -> Iterator[ProblemImportCase]:
        """Iterate the cases of the group `sciml_problem_import`."""
        for cid in self.case_ids(PROBLEM_IMPORT):
            yield ProblemImportCase.from_directory(self.path / PROBLEM_IMPORT / cid)

    def run(self) -> list[CaseResult]:
        """Run the cases of the suite.

        Returns:
            The results of the groups `ml_model_import`, `initialization` and
            `sciml_problem_import`, in this order.
        """
        results = [case.run() for case in self.model_import_cases()]
        results.extend(case.run() for case in self.initialization_cases())
        results.extend(case.run() for case in self.problem_import_cases())
        return results
