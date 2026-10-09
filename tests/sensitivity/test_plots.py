"""The plots of a sensitivity result."""

from pathlib import Path

import matplotlib
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

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


def _mesh(figure: Figure) -> QuadMesh:
    """Get the color mesh of the heatmap, the one on the largest axes."""
    meshes = [
        (ax.get_position().width * ax.get_position().height, ax.collections[0])
        for ax in figure.axes
        if ax.collections and isinstance(ax.collections[0], QuadMesh)
    ]
    return max(meshes, key=lambda m: m[0])[1]


def _same_colors(mesh: QuadMesh, name: str) -> bool:
    """Check that the color map of a mesh has the colors of a named one."""
    points = np.linspace(0, 1, 11)
    return bool(
        np.allclose(mesh.get_cmap()(points), matplotlib.colormaps[name](points))
    )


def test_plot_indices_data() -> None:
    result = _sobol()
    figure = plot_indices(result, "auc", dose=1)
    ax = figure.axes[0]
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    s1 = result["auc.S1"].sel(dose=1).values
    st = result["auc.ST"].sel(dose=1).values
    np.testing.assert_allclose([b.get_height() for b in bars[:3]], s1)
    np.testing.assert_allclose([b.get_height() for b in bars[3:]], st)
    segments = [
        seg
        for c in ax.collections
        if isinstance(c, LineCollection)
        for seg in c.get_segments()
    ]
    lengths = sorted(float(seg[1][1] - seg[0][1]) for seg in segments)
    conf = np.concatenate(
        [
            2 * result["auc.S1_conf"].sel(dose=1).values,
            2 * result["auc.ST_conf"].sel(dose=1).values,
        ]
    )
    np.testing.assert_allclose(lengths, np.sort(conf))
    assert ax.get_ylabel() == "Sobol index"


def test_plot_morris_positions() -> None:
    figure = plot_morris(_morris(), "y")
    offsets = np.asarray(figure.axes[0].collections[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 0], [1.0, 0.5, 0.1])
    np.testing.assert_allclose(offsets[:, 1], [1.0, 0.5, 0.1])


def _single() -> SensitivityResult:
    data = {f"o{k}.ST": ((PARAMETER,), np.array([0.5])) for k in range(20)}
    return SensitivityResult(
        xr.Dataset(data, coords={PARAMETER: ["a"]}, attrs={"units": {}})
    )


def test_heatmap_one_parameter_and_size() -> None:
    figure = plot_heatmap(_single(), "ST")
    width, height = figure.get_size_inches()
    assert 0 < height <= 40 and width <= 24
    figure = plot_heatmap(_sobol(), "ST", dose=0)
    assert figure.get_size_inches()[1] < 10


def _normalized() -> SensitivityResult:
    values = np.array([[0.5, 1.0], [np.nan, np.nan], [0.01, 0.02]])
    data = {f"o{k}.normalized": ((PARAMETER,), values[:, k]) for k in range(2)}
    return SensitivityResult(
        xr.Dataset(data, coords={PARAMETER: ["a", "b", "c"]}, attrs={"units": {}})
    )


def test_heatmap_keeps_undefined_rows() -> None:
    figure = plot_heatmap(_normalized(), "normalized")
    mesh = _mesh(figure)
    labels = [t.get_text() for ax in figure.axes for t in ax.get_yticklabels()]
    assert "b" in labels and "c" not in labels
    ax = mesh.axes
    assert ax is not None
    grey = [
        p for p in ax.patches if np.allclose(p.get_facecolor(), to_rgba("lightgrey"))
    ]
    assert len(grey) == 2


def test_heatmap_defaults() -> None:
    mesh = _mesh(plot_heatmap(_sobol(), "ST", dose=0))
    assert _same_colors(mesh, "Reds") and mesh.get_clim() == (0.0, 1.0)
    mesh = _mesh(plot_heatmap(_normalized(), "normalized"))
    assert _same_colors(mesh, "seismic") and mesh.get_clim() == (-2.0, 2.0)
    mesh = _mesh(plot_heatmap(_sobol(), "ST", dose=0, cmap="viridis", vmax=3))
    assert _same_colors(mesh, "viridis") and mesh.get_clim() == (0.0, 3.0)


def test_unknown_selections() -> None:
    with pytest.raises(ValueError, match=r"Unknown selection 'foo'.*dose"):
        plot_indices(_sobol(), "auc", foo=1)
    with pytest.raises(ValueError, match=r"Unknown label 5.*\[0, 1\]"):
        plot_heatmap(_sobol(), "ST", dose=5)
    with pytest.raises(ValueError, match=r"Unknown observable 'x'.*auc"):
        plot_indices(_sobol(), "x", dose=0)
    with pytest.raises(ValueError, match=r"Unknown observable 'x'"):
        plot_morris(_morris(), "x")
    with pytest.raises(ValueError, match="index 'nope'"):
        plot_heatmap(_sobol(), "nope", dose=0)
