"""A writer of small PEtab SciML problems for the tests of the reader."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from petab_sciml import NNModelStandard

from sbmlsim.sciml import Network
from tests.sciml.experiment import DATA
from tests.sciml.hybrid import MODEL_PATH

#: the parameters of the model of the parameter table, `lin` is the scale
#: the problems of the test suite carry
MECHANISTIC = [("alpha", 1.3), ("beta", 0.9), ("delta", 1.8)]


def write_table(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] = ()
) -> Path:
    """Write the rows as a TSV table, with the columns for a table without rows."""
    df = pd.DataFrame(list(rows)) if rows else pd.DataFrame(columns=list(columns))
    df.to_csv(path, sep="\t", index=False)
    return path


def write_arrays(
    path: Path,
    networks: Sequence[Network] = (),
    inputs: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> Path:
    """Write an array file with the arrays of networks and inputs.

    Args:
        path: the HDF5 file.
        networks: the networks whose arrays are written.
        inputs: id of the input -> id of the condition -> array.

    Returns:
        The path.
    """
    with h5py.File(path, "w") as f:
        f.create_group("metadata")["pytorch_format"] = True
        for network in networks:
            for layer, arrays in network.parameters.items():
                for name, array in arrays.items():
                    f[f"parameters/{network.sid}/{layer}/{name}"] = np.asarray(array)
        for input_id, conditions in (inputs or {}).items():
            for condition, array in conditions.items():
                f[f"inputs/{input_id}/{condition}"] = np.asarray(array, dtype=float)
    return path


def write_problem(
    directory: Path,
    networks: Sequence[Network],
    pre_initialization: Mapping[str, bool],
    mapping: Sequence[tuple[str, str]],
    hybridization: Sequence[tuple[str, str]],
    parameters: Sequence[Mapping[str, Any]] = (),
    observables: Mapping[str, str] | None = None,
    conditions: Sequence[tuple[str, str, str]] = (),
    experiments: Mapping[str, str | None] | None = None,
    inputs: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    formats: Mapping[str, str] | None = None,
    extra_yaml: Mapping[str, Any] | None = None,
) -> Path:
    """Write a PEtab SciML problem with the model of Lotka and Volterra.

    Args:
        directory: the directory the files are written into.
        networks: the networks of the problem, with their nominal values.
        pre_initialization: id of the network -> whether it runs before the
            simulation.
        mapping: the rows of the mapping table, `petabEntityId` and
            `modelEntityId`.
        hybridization: the rows of the hybridization table, `targetId` and
            `targetValue`.
        parameters: the rows of the parameter table in addition to the ones of
            `alpha`, `beta` and `delta` and of the arrays of the networks.
        observables: id of the observable -> its formula, `prey` and
            `predator` by default.
        conditions: the rows of the condition table, `conditionId`,
            `targetId` and `targetValue`.
        experiments: id of the experiment -> id of its condition, one
            experiment `e1` without a condition by default.
        inputs: the arrays of the inputs, see `write_arrays`.
        formats: id of the network -> its format, `YAML` by default.
        extra_yaml: entries added to the YAML of the problem.

    Returns:
        The path of the YAML of the problem.
    """
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "lv.xml"
    model_path.write_bytes(MODEL_PATH.read_bytes())
    for network in networks:
        NNModelStandard.save_data(
            data=network.model, filename=str(directory / f"{network.sid}.yaml")
        )
    array_files = [write_arrays(directory / "arrays.hdf5", networks, inputs).name]

    observables = observables or {"prey_o": "prey", "predator_o": "predator"}
    experiments = experiments or {"e1": None}
    write_table(
        directory / "observables.tsv",
        [
            {
                "observableId": sid,
                "observableFormula": formula,
                "noiseFormula": 0.05,
                "noiseDistribution": "normal",
            }
            for sid, formula in observables.items()
        ],
    )
    measurements = []
    for experiment_id in experiments:
        for sid, formula in observables.items():
            species = (
                "prey" if "prey" in formula or sid.startswith("prey") else "predator"
            )
            for k, value in enumerate(DATA[species]):
                measurements.append(
                    {
                        "observableId": sid,
                        "experimentId": experiment_id,
                        "measurement": value,
                        "time": float(k + 1),
                    }
                )
    write_table(directory / "measurements.tsv", measurements)
    write_table(
        directory / "experiments.tsv",
        [
            {"experimentId": sid, "time": 0.0, "conditionId": condition or ""}
            for sid, condition in experiments.items()
        ],
    )
    rows: list[dict[str, Any]] = [
        {
            "parameterId": sid,
            "parameterScale": "lin",
            "lowerBound": 0.0,
            "upperBound": 15.0,
            "nominalValue": value,
            "estimate": True,
        }
        for sid, value in MECHANISTIC
    ]
    rows.extend(
        {
            "parameterId": f"{network.sid}_ps",
            "parameterScale": "lin",
            "lowerBound": "-inf",
            "upperBound": "inf",
            "nominalValue": "array",
            "estimate": True,
        }
        for network in networks
    )
    rows.extend(dict(row) for row in parameters)
    write_table(directory / "parameters.tsv", rows)
    write_table(
        directory / "mapping.tsv",
        [
            {"petabEntityId": petab_id, "modelEntityId": model_id}
            for petab_id, model_id in [
                *mapping,
                *((f"{n.sid}_ps", f"{n.sid}.parameters") for n in networks),
            ]
        ],
    )
    write_table(
        directory / "hybridization.tsv",
        [{"targetId": target, "targetValue": value} for target, value in hybridization],
        columns=["targetId", "targetValue"],
    )
    config: dict[str, Any] = {
        "format_version": "2.0.0",
        "model_files": {"lv": {"location": "lv.xml", "language": "sbml"}},
        "measurement_files": ["measurements.tsv"],
        "observable_files": ["observables.tsv"],
        "experiment_files": ["experiments.tsv"],
        "parameter_files": ["parameters.tsv"],
        "mapping_files": ["mapping.tsv"],
        "extensions": {
            "sciml": {
                "version": "0.1.0",
                "required": True,
                "array_files": array_files,
                "hybridization_files": ["hybridization.tsv"],
                "neural_networks": {
                    network.sid: {
                        "location": f"{network.sid}.yaml",
                        "pre_initialization": bool(pre_initialization[network.sid]),
                        "format": (formats or {}).get(network.sid, "YAML"),
                    }
                    for network in networks
                },
            }
        },
    }
    if conditions:
        write_table(
            directory / "conditions.tsv",
            [
                {"conditionId": condition, "targetId": target, "targetValue": value}
                for condition, target, value in conditions
            ],
        )
        config["condition_files"] = ["conditions.tsv"]
    config.update(extra_yaml or {})
    path = directory / "problem.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
