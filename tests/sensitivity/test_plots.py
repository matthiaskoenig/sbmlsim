"""The plots of a sensitivity result."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure

from sbmlsim.sensitivity import plot_heatmap, plot_indices, plot_morris
from sbmlsim.sensitivity.result import PARAMETER, SensitivityResult


def _sobol() -> SensitivityResult:
    rng = np.random.default_rng(1)
    data = {}
    for o in ("auc", "cmax"):
        for key in ("S1", "ST", "S1_conf", "ST_conf"):
            data[f"{o}.{key}"] = ((PARAMETER, "dose"), rng.random((3, 2)))
    units = dict.fromkeys(data, "dimensionless")
    return SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: ["a", "b", "c"], "dose": [0, 1]},
            attrs={"units": units, "method": "sobol"},
        )
    )


def _morris() -> SensitivityResult:
    data = {
        f"y.{key}": ((PARAMETER,), np.array([1.0, 0.5, 0.1]))
        for key in ("mu", "mu_star", "sigma", "mu_star_conf")
    }
    return SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: ["a", "b", "c"]},
            attrs={"units": dict.fromkeys(data, ""), "method": "morris"},
        )
    )


def test_plot_heatmap(tmp_path: Path) -> None:
    figure = plot_heatmap(_sobol(), "ST", dose=0, path=tmp_path / "h.png", cutoff=None)
    assert isinstance(figure, Figure) and (tmp_path / "h.png").exists()
    with pytest.raises(ValueError, match="dose"):
        plot_heatmap(_sobol(), "ST")  # two doses: choose one


def test_plot_indices_and_morris() -> None:
    figure = plot_indices(_sobol(), "auc", dose=1)
    assert isinstance(figure, Figure) and len(figure.axes[0].patches) == 6
    figure = plot_morris(_morris(), "y")
    assert isinstance(figure, Figure) and len(figure.axes[0].texts) == 3
