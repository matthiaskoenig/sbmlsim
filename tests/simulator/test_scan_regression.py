"""The scan examples give the values they gave before the scan core.

The values were recorded with the API of 0.8.5 in the layout `(*dims, time)`,
see `tests/data/scan_regression.json`.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples import scan as example_scan
from examples.glucose.experiments.dose_response import DoseResponseExperiment
from examples.repressilator.repressilator_scans import RepressilatorScanExperiment
from sbmlsim.experiment import ExperimentRunner
from sbmlsim.result import ScanResult
from sbmlsim.simulator import Simulator

DATA = Path(__file__).parents[1] / "data" / "scan_regression.json"
RECORD: dict[str, dict[str, Any]] = json.loads(DATA.read_text(encoding="utf-8"))
EXAMPLES = Path(__file__).parents[2] / "examples"

# The values were recorded on one machine. The math library and the code which
# roadrunner compiles for the processor of another machine round in the last
# bits, which the oscillations of the repressilator grow to 3e-8 relative (the
# runners of GitHub); a change of one ulp in the scanned values already gives
# 4e-9. A scan which differs in what it simulates differs far above 1e-6.
RTOL = 1e-6


def _compare(res: ScanResult, recorded: dict[str, Any], step: int = 1) -> None:
    for key, values in recorded.items():
        expected = np.asarray(values, dtype=float)
        if key == "time" and not res.ragged:
            actual = np.broadcast_to(res["time"].values[::step], expected.shape)
        else:
            actual = res[key].values[..., ::step]
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=1e-12, err_msg=key)


def _results(
    experiment_class: Any, base: Path, data: Path, reduced: bool, tmp_path: Path
) -> dict[str, ScanResult]:
    runner = ExperimentRunner(
        [experiment_class], simulator=Simulator(), base_path=base, data_path=data
    )
    # the values are compared, the figures are not drawn
    runner.run_experiments(
        output_path=tmp_path,
        show_figures=False,
        figure_formats=[],
        reduced_selections=reduced,
    )
    return next(iter(runner.experiments.values())).results


@pytest.mark.parametrize("name", ["run_scan0d", "run_scan1d", "run_scan2d"])
def test_the_scans_of_the_scan_example(name: str) -> None:
    _compare(getattr(example_scan, name)(), RECORD[f"scan.{name}"])


def test_the_scans_of_the_repressilator(tmp_path: Path) -> None:
    base = EXAMPLES / "repressilator"
    results = _results(RepressilatorScanExperiment, base, base, False, tmp_path)
    keys = [key for key in RECORD if key.startswith("repressilator_scans.")]
    assert len(keys) == 4
    for key in keys:
        res = results[key.split(".", 1)[1]]
        # a target the scan changes is the coordinate of its dimension and no
        # timecourse of the scan (it was one before a result kept the
        # coordinates apart), so only the other variables are compared
        recorded = {
            k: v
            for k, v in RECORD[key].items()
            if k == "time" or k not in res.ds.coords
        }
        assert len(recorded) < len(RECORD[key])
        _compare(res, recorded, step=200)


def test_the_dose_response_of_the_glucose(tmp_path: Path) -> None:
    glucose = EXAMPLES / "glucose"
    res = _results(DoseResponseExperiment, glucose, glucose / "data", True, tmp_path)[
        "task_glc_scan"
    ]
    recorded = dict(RECORD["dose_response.task_glc_scan"])
    # the glucose of the scan is the coordinate of its dimension
    glucose_values = np.asarray(recorded.pop("[glc_ext]"), dtype=float)
    assert "[glc_ext]" in res.ds.coords
    np.testing.assert_allclose(res["[glc_ext]"].values, glucose_values[:, 0], rtol=1e-9)
    _compare(res, recorded)
