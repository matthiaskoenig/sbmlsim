"""The submission of the results to the SBML Test Suite Database.

A submission is an archive with the results of every case which could be
simulated, one CSV per case named after it, and a manifest with the versions
the results were produced with. A case which the simulator cannot read or
integrate has no results and is left out, which is how a submission says that
it does not support the case. Uploading the archive is a deliberate step and is
not done here.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import pandas as pd

from sbmlsim import __version__
from sbmlsim.model import AbstractModel
from sbmlsim.simulator import Simulator
from sbmlsim.testsuite.cases import SemanticCase, SemanticSuite
from sbmlsim.testsuite.report import versions
from sbmlsim.testsuite.runner import (
    INTEGRATOR_ABSOLUTE_TOLERANCE,
    INTEGRATOR_RELATIVE_TOLERANCE,
    map_cases,
    simulate_case,
)

logger = logging.getLogger(__name__)


def case_csv(case: SemanticCase) -> tuple[str | None, str]:
    """Simulate a case and write its results as the submission names them.

    The function runs in the processes of `map_cases`, which do not share the
    logging of the process which started them, so it answers with the reason
    rather than logging it.

    Args:
        case: the case to simulate.

    Returns:
        The CSV of the results and an empty message, or `None` and why the case
        could not be read or integrated.
    """
    simulator = Simulator(
        n_workers=1,
        absolute_tolerance=INTEGRATOR_ABSOLUTE_TOLERANCE,
        relative_tolerance=INTEGRATOR_RELATIVE_TOLERANCE,
        variable_step_size=False,
    )
    try:
        model = simulator.load(AbstractModel(source=case.model_path))
        observed = simulate_case(case, simulator, model)
    except Exception as err:
        lines = str(err).strip().splitlines()
        return None, lines[0][:300] if lines else type(err).__name__
    # the submission names the columns as the case names its variables, i.e.
    # `S1` and not the selection `[S1]`; the columns are the selections of the
    # case in their order
    df = pd.DataFrame(observed.values, columns=["time", *case.variables])
    return df.to_csv(index=False), ""


def write_submission(
    suite: SemanticSuite, output_dir: Path, workers: int | None = None
) -> Path:
    """Write the archive which is submitted to the SBML Test Suite Database.

    Args:
        suite: the suite to run.
        output_dir: directory the archive is written into.
        workers: number of processes the cases are simulated in, the cores
            available to this process by default.

    Returns:
        Path of the archive.
    """
    cases = list(suite.cases())
    results = map_cases(case_csv, cases, workers=workers)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sbmlsim-{__version__}-sbml-test-suite-{suite.version}.zip"
    submitted = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for case, (csv, message) in zip(cases, results, strict=True):
            if csv is None:
                logger.info("'%s': no results (%s)", case.cid, message)
                continue
            zf.writestr(f"{case.cid}.csv", csv)
            submitted += 1
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "simulator": "sbmlsim",
                    "versions": versions(),
                    "suite_version": suite.version,
                    "n_cases": len(cases),
                    "n_submitted": submitted,
                },
                indent=2,
            ),
        )
    logger.info(
        "Submission archive with %s of %s cases: %s", submitted, len(cases), path
    )
    return path
