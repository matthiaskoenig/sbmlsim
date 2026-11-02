"""A problem of PEtab v1, which `sbmlsim` reads after its conversion to PEtab v2."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tests.simulator.models import sbml

#: the times of the measurements and their values, `A` decays with `k = 0.1`
TIMES = (0.0, 10.0)
VALUES = tuple(float(np.exp(-0.1 * t)) for t in TIMES)


def write_v1_problem(directory: Path, prefix: str = "") -> Path:
    """Write a problem of PEtab v1 with the simulations of its measurements.

    `A` decays with `k`, the observable `obs_a` is `A` with the noise `0.5`
    in the condition `c0`, the measurements and the simulations are the exact
    solution at the nominal `k`.

    Args:
        directory: the directory of the problem, which is created.
        prefix: the prefix of the files, e.g. `_0001` of a case of the test
            suite, whose YAML file is `_0001.yaml`.

    Returns:
        The YAML file of the problem.
    """
    directory.mkdir(parents=True)
    (directory / "_model.xml").write_text(
        sbml("model decay\n  A = 1; k = 0.1\n  J: A -> ; k*A\nend")
    )
    rows = {
        "observableId": ["obs_a"] * len(TIMES),
        "simulationConditionId": ["c0"] * len(TIMES),
        "time": list(TIMES),
    }
    pd.DataFrame({**rows, "measurement": VALUES}).to_csv(
        directory / "_measurements.tsv", sep="\t", index=False
    )
    pd.DataFrame({**rows, "simulation": VALUES}).to_csv(
        directory / "_simulations.tsv", sep="\t", index=False
    )
    pd.DataFrame({"conditionId": ["c0"]}).to_csv(
        directory / "_conditions.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {"observableId": ["obs_a"], "observableFormula": ["A"], "noiseFormula": ["0.5"]}
    ).to_csv(directory / "_observables.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "parameterId": ["k"],
            "parameterScale": ["lin"],
            "lowerBound": [0.01],
            "upperBound": [1.0],
            "nominalValue": [0.1],
            "estimate": [1],
        }
    ).to_csv(directory / "_parameters.tsv", sep="\t", index=False)
    yaml_file = directory / f"{prefix or 'problem'}.yaml"
    yaml_file.write_text(
        yaml.safe_dump(
            {
                "format_version": 1,
                "parameter_file": "_parameters.tsv",
                "problems": [
                    {
                        "sbml_files": ["_model.xml"],
                        "condition_files": ["_conditions.tsv"],
                        "measurement_files": ["_measurements.tsv"],
                        "observable_files": ["_observables.tsv"],
                    }
                ],
            }
        )
    )
    return yaml_file
