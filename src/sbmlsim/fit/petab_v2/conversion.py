"""The conversion of a problem of PEtab v1 to PEtab v2, with its simulations.

`sbmlsim` reads PEtab v2 only. A problem of PEtab v1, e.g. a case of the v1
test suite or a problem of the benchmark collection, is converted once with
`petab.v2.petab1to2` and read as the problem of PEtab v2 it is then.

The reference simulations of such a problem are tables of PEtab v1, with the
conditions of a simulation (`simulationConditionId`,
`preequilibrationConditionId`) where PEtab v2 has its experiment
(`experimentId`). They are converted with the problem: the problem with the
simulations in place of its measurements is converted as well, so the
simulations get the experiments `petab1to2` gives the measurements, and are
compared with a simulation of the problem of PEtab v2 row by row. Nothing
after the conversion reads PEtab v1.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

#: the file the converted simulations are written to
SIMULATIONS_FILE = "simulations.tsv"

#: the column of the value of a simulation and of a measurement
SIMULATION = "simulation"
MEASUREMENT = "measurement"


def convert_v1(
    yaml_file: Path,
    output_dir: Path,
    simulation_files: Sequence[Path] = (),
) -> Path | None:
    """Convert a problem of PEtab v1 and its simulations to PEtab v2.

    Args:
        yaml_file: the YAML file of the problem of PEtab v1.
        output_dir: the directory the problem of PEtab v2 is written to, it
            must not exist.
        simulation_files: the tables of the reference simulations of the
            problem, in PEtab v1; they are written to `SIMULATIONS_FILE` of
            the output directory as one table of PEtab v2, i.e. a measurement
            table with the column `simulation`.

    Returns:
        The table of the simulations, `None` without simulation files.

    Raises:
        Exception: what `petab1to2` raises for a problem it cannot convert.
    """
    _petab1to2(yaml_file, output_dir, validate=True)
    if not simulation_files:
        return None
    path = output_dir / SIMULATIONS_FILE
    convert_simulations(yaml_file, simulation_files).to_csv(path, sep="\t", index=False)
    return path


def convert_simulations(
    yaml_file: Path, simulation_files: Sequence[Path]
) -> pd.DataFrame:
    """Convert the simulations of a problem of PEtab v1 to PEtab v2.

    Args:
        yaml_file: the YAML file of the problem of PEtab v1.
        simulation_files: the tables of its simulations, in PEtab v1.

    Returns:
        The simulations as one table of PEtab v2, i.e. a measurement table
        with the experiments `petab1to2` gives the measurements and the
        column `simulation`.

    Raises:
        Exception: what `petab1to2` raises for a problem it cannot convert.
    """
    simulations = pd.concat(
        [pd.read_csv(path, sep="\t") for path in simulation_files], ignore_index=True
    )
    converted = _converted_measurements(yaml_file, simulations)
    return converted.rename(columns={MEASUREMENT: SIMULATION})


def _petab1to2(yaml_file: Path, output_dir: Path, validate: bool) -> None:
    """Convert with `petab1to2`, logging its warnings at the level DEBUG."""
    from petab.v2.petab1to2 import petab1to2

    with warnings.catch_warnings(record=True) as caught:
        # e.g. that PEtab v2 has no scales of the parameters, for every problem
        warnings.simplefilter("always")
        petab1to2(yaml_file, output_dir=output_dir, validate=validate)
    for warning in caught:
        logger.debug("'%s': %s", yaml_file.name, warning.message)


def _converted_measurements(yaml_file: Path, simulations: pd.DataFrame) -> pd.DataFrame:
    """Convert the simulations of a problem as the measurements of the problem.

    The problem is copied with the simulations as its one measurement table
    and converted without validation, a simulation need not be a valid
    measurement, e.g. `NaN`.

    Args:
        yaml_file: the YAML file of the problem of PEtab v1.
        simulations: the simulations, a table of PEtab v1.

    Returns:
        The measurement table of the converted copy, with the experiments of
        PEtab v2 and the simulations in the column `measurement`.
    """
    with tempfile.TemporaryDirectory(prefix="petab1to2_") as tmp:
        source = Path(tmp) / "v1"
        shutil.copytree(yaml_file.parent, source)
        table = "sbmlsim_simulations.tsv"
        simulations.rename(columns={SIMULATION: MEASUREMENT}).to_csv(
            source / table, sep="\t", index=False
        )
        config: dict[str, Any] = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        for problem in config.get("problems", []):
            problem["measurement_files"] = [table]
        copy = source / yaml_file.name
        copy.write_text(yaml.safe_dump(config), encoding="utf-8")
        target = Path(tmp) / "v2"
        _petab1to2(copy, target, validate=False)
        return measurement_table(target)


def measurement_table(directory: Path) -> pd.DataFrame:
    """Read the measurement tables of a problem of PEtab v2 as one table.

    Args:
        directory: the directory of the problem, with its one YAML file.

    Returns:
        The measurements in the order of the files of the problem.

    Raises:
        ValueError: if the directory has no YAML file or the problem no
            measurement table.
    """
    yaml_files = sorted(directory.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"The directory '{directory}' has no YAML file.")
    config: dict[str, Any] = yaml.safe_load(yaml_files[0].read_text(encoding="utf-8"))
    files = list(config.get("measurement_files") or [])
    if not files:
        raise ValueError(f"The problem '{yaml_files[0]}' has no measurement table.")
    return pd.concat(
        [pd.read_csv(directory / name, sep="\t") for name in files], ignore_index=True
    )
