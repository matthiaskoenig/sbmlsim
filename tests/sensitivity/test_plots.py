"""The plots of a sensitivity result."""

import re
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text

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


def test_heatmap_title_fits_the_figure() -> None:
    figure = plot_heatmap(_sobol(), "ST", dose=0, title="normalized, [S1] = 1")
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    (title,) = [
        t for t in figure.findobj(Text) if t.get_text() == "normalized, [S1] = 1"
    ]
    box = title.get_window_extent(renderer)
    assert box.x0 >= 0 and box.x1 <= figure.bbox.x1 and box.y1 <= figure.bbox.y1
    mesh = _mesh(figure)
    assert mesh.axes is not None
    assert box.y0 > mesh.axes.get_window_extent(renderer).y1


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


def test_plot_morris_labels_inside_frame() -> None:
    figure = plot_morris(_morris(), "y")
    figure.canvas.draw()
    ax = figure.axes[0]
    frame = ax.get_window_extent()
    for note in ax.texts:
        box = note.get_window_extent()
        assert box.x1 < frame.x1 and box.y1 < frame.y1


def _heatmap_axes(figure: Figure) -> Axes:
    (ax,) = [ax for ax in figure.axes if ax.get_label() == "heatmap"]
    return ax


def _colorbar_axes(figure: Figure) -> Axes:
    (ax,) = [ax for ax in figure.axes if ax.get_label() == "<colorbar>"]
    return ax


def _texts(labels: list[Text]) -> list[Text]:
    return [t for t in labels if t.get_text() and t.get_visible()]


def _ticks(axis: Axis) -> list[Text]:
    """Get the tick labels an axis draws, the ones inside its limits."""
    low, high = sorted(axis.get_view_interval())
    return [
        label
        for label, loc in zip(axis.get_ticklabels(), axis.get_ticklocs(), strict=True)
        if label.get_text() and low <= loc <= high
    ]


def _overlap(texts: list[Text]) -> bool:
    """Check whether two neighbouring texts overlap.

    Two labels rotated by 45 degrees are parallel: they are clear of each other
    when the distance of their lines, the distance of their anchors times
    sin(45), is at least the height of a line.
    """
    if not texts:
        return False
    if texts[0].get_rotation() == 45:
        figure = texts[0].get_figure(root=True)
        assert figure is not None
        anchors = sorted(
            float(t.get_transform().transform(t.get_position())[0]) for t in texts
        )
        height = max(float(t.get_fontsize()) for t in texts) * figure.dpi / 72.0
        return any((b - a) * np.sin(np.pi / 4) < height for a, b in pairwise(anchors))
    boxes = [t.get_window_extent() for t in texts]
    return any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])


def _inside(figure: Figure, texts: list[Text]) -> bool:
    """Check that the texts are inside the figure."""
    frame = figure.bbox
    return all(
        t.get_window_extent().x0 >= frame.x0 - 0.5
        and t.get_window_extent().x1 <= frame.x1 + 0.5
        and t.get_window_extent().y0 >= frame.y0 - 0.5
        and t.get_window_extent().y1 <= frame.y1 + 0.5
        for t in texts
    )


def _result(
    index: str,
    values: np.ndarray,
    parameters: list[str],
    observables: list[str],
    unit: str = "dimensionless",
    method: str = "sobol",
) -> SensitivityResult:
    data = {
        f"{o}.{index}": ((PARAMETER,), values[:, k]) for k, o in enumerate(observables)
    }
    return SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: parameters},
            attrs={"units": dict.fromkeys(data, unit), "method": method},
        )
    )


def test_heatmap_cutoff_depends_on_the_index() -> None:
    values = np.array([[0.05, 0.01], [0.002, 0.003], [0.04, 0.0]])
    mu_star = _result("mu_star", values, ["a", "b", "c"], ["o1", "o2"], unit="mM")
    figure = plot_heatmap(mu_star, "mu_star")
    ax = _heatmap_axes(figure)
    # no cutoff for an index with a unit
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"] or sorted(
        t.get_text() for t in ax.get_yticklabels()
    ) == ["a", "b", "c"]
    assert _colorbar_axes(figure).get_ylabel() == "mu_star [mM]"
    # an explicit cutoff is absolute in the unit of the index
    figure = plot_heatmap(mu_star, "mu_star", cutoff=0.03)
    labels = sorted(t.get_text() for t in _heatmap_axes(figure).get_yticklabels())
    assert labels == ["a", "c"]
    # 0.1 for a dimensionless index, which no parameter reaches here
    st = _result("ST", values, ["a", "b", "c"], ["o1", "o2"])
    with pytest.raises(ValueError, match=r"cutoff 0\.1.*0\.05"):
        plot_heatmap(st, "ST")
    with pytest.raises(ValueError, match=r"cutoff 1.*0\.05"):
        plot_heatmap(mu_star, "mu_star", cutoff=1.0)
    figure = plot_heatmap(st, "ST", cutoff=0)
    assert len(_heatmap_axes(figure).get_yticklabels()) == 3


def test_heatmap_degenerate_rows() -> None:
    """Identical or undefined rows draw without a warning (filterwarnings = error)."""
    identical = _result("normalized", np.ones((3, 2)), ["a", "b", "c"], ["o1", "o2"])
    figure = plot_heatmap(identical, "normalized")
    assert len(_heatmap_axes(figure).get_yticklabels()) == 3
    undefined = _result(
        "normalized", np.full((3, 2), np.nan), ["a", "b", "c"], ["o1", "o2"]
    )
    figure = plot_heatmap(undefined, "normalized")
    assert len(_heatmap_axes(figure).get_yticklabels()) == 3
    figure = plot_heatmap(undefined, "normalized", cutoff=None)
    assert len(_heatmap_axes(figure).get_yticklabels()) == 3


def _check_heatmap_layout(figure: Figure) -> None:
    figure.draw_without_rendering()
    ax = _heatmap_axes(figure)
    cbar = _colorbar_axes(figure)
    rows = _texts(ax.get_yticklabels())
    cols = _texts(ax.get_xticklabels())
    ticks = _ticks(cbar.yaxis)
    assert all(t.get_rotation() == 0 for t in rows)
    assert not _overlap(rows) and not _overlap(cols) and not _overlap(ticks)
    cells = ax.get_window_extent()
    for t in [*ticks, cbar.yaxis.label]:
        assert not t.get_window_extent().overlaps(cells)
    for t in [*rows, *cols]:
        assert not t.get_window_extent().overlaps(cbar.get_window_extent())
    titles = [t for t in figure.texts if t.get_text()]
    assert _inside(figure, [*rows, *cols, *ticks, cbar.yaxis.label, *titles])


def test_heatmap_layout_of_short_names() -> None:
    values = np.array([[0.5, -1.2], [2500.0, 0.3]])
    figure = plot_heatmap(
        _result("mu", values, ["k1", "k2"], ["o1", "o2"], unit="mM"), "mu"
    )
    _check_heatmap_layout(figure)
    ax = _heatmap_axes(figure)
    assert all(t.get_rotation() == 0 for t in ax.get_xticklabels())
    assert _colorbar_axes(figure).get_ylabel() == "mu [mM]"


def test_heatmap_layout_of_long_and_many_names() -> None:
    rng = np.random.default_rng(1)
    parameters = [f"Vmax_transport_liver_{k}" for k in range(40)]
    observables = [f"concentration_in_plasma_{k}" for k in range(6)]
    result = _result("ST", rng.random((40, 6)), parameters, observables)
    figure = plot_heatmap(result, "ST", title="ST of the plasma concentrations")
    _check_heatmap_layout(figure)
    ax = _heatmap_axes(figure)
    assert all(t.get_rotation() == 45 for t in _texts(ax.get_xticklabels()))
    # the title sits right above the cells
    (title,) = [
        t
        for t in figure.findobj(Text)
        if t.get_text() == "ST of the plasma concentrations"
    ]
    gap = title.get_window_extent().y0 - ax.get_window_extent().y1
    assert 0 < gap < 0.6 * figure.dpi
    one = _result("ST", np.array([[0.5]]), ["Vmax_transport_liver"], ["o1"])
    _check_heatmap_layout(plot_heatmap(one, "ST"))


def test_heatmap_names_its_index_and_unit() -> None:
    figure = plot_heatmap(_normalized(), "normalized")
    assert _colorbar_axes(figure).get_ylabel() == "normalized"
    mixed = _result("raw", np.ones((2, 2)), ["a", "b"], ["o1", "o2"], unit="mM")
    mixed.ds.attrs["units"]["o2.raw"] = "mM/s"
    assert _colorbar_axes(plot_heatmap(mixed, "raw")).get_ylabel() == "raw"


def test_plot_indices_contrast_and_labels() -> None:
    figure = plot_indices(_sobol(), "auc", dose=1)
    ax = figure.axes[0]
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    errors = [c for c in ax.collections if isinstance(c, LineCollection)]
    for bar in bars:
        face = np.asarray(bar.get_facecolor()[:3])
        for lines in errors:
            line = np.asarray(lines.get_edgecolor())[0, :3]
            assert np.abs(face - line).sum() > 1.0
    figure.draw_without_rendering()
    labels = _texts(ax.get_xticklabels())
    assert all(t.get_rotation() == 0 for t in labels)
    assert ax.get_xlabel() == "Parameter"


def test_plot_indices_long_names() -> None:
    parameters = [f"Vmax_transport_liver_{k}" for k in range(12)]
    data = {
        f"y.{key}": ((PARAMETER,), np.linspace(0.1, 0.9, 12))
        for key in ("S1", "ST", "S1_conf", "ST_conf")
    }
    result = SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: parameters},
            attrs={"units": dict.fromkeys(data, "dimensionless"), "method": "fast"},
        )
    )
    figure = plot_indices(result, "y")
    figure.draw_without_rendering()
    labels = _texts(figure.axes[0].get_xticklabels())
    assert all(t.get_rotation() == 45 for t in labels)
    assert not _overlap(labels) and _inside(figure, labels)
    assert figure.axes[0].get_ylabel() == "FAST index"


def test_the_plots_check_the_analysis() -> None:
    with pytest.raises(ValueError, match=r"S1 and ST.*Sobol or FAST.*morris"):
        plot_indices(_morris(), "y")
    with pytest.raises(ValueError, match=r"mu_star and sigma.*Morris.*sobol"):
        plot_morris(_sobol(), "auc", dose=0)


def test_plot_morris_axes_and_undefined_parameters() -> None:
    data = {
        f"y.{key}": ((PARAMETER,), np.array([5e-5, 2e-5, np.nan]))
        for key in ("mu", "mu_star", "sigma", "mu_star_conf")
    }
    result = SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: ["a", "b", "c"]},
            attrs={"units": dict.fromkeys(data, "mM"), "method": "morris"},
        )
    )
    figure = plot_morris(result, "y")
    figure.draw_without_rendering()
    ax = figure.axes[0]
    assert ax.get_xlim()[0] == 0 and ax.get_ylim()[0] == 0
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        # a common power of ten, not 0.00005
        text = re.sub(r"^\$\\mathdefault\{(.*)\}\$$", r"\1", label.get_text())
        assert len(text) <= 4
    notes = [t.get_text() for t in figure.findobj(Text)]
    assert any("undefined" in n and "c" in n for n in notes)
    assert ax.get_xlabel() == "mu_star [mM]"


def _table(values: np.ndarray, columns: list[str] | None = None) -> SensitivityResult:
    n_rows, n_cols = values.shape
    names = columns or [f"o{i}" for i in range(n_cols)]
    data = {f"{n}.mu_star": ((PARAMETER,), values[:, i]) for i, n in enumerate(names)}
    return SensitivityResult(
        xr.Dataset(
            data,
            coords={PARAMETER: [f"p{i}" for i in range(n_rows)]},
            attrs={"units": dict.fromkeys(data, ""), "method": "morris"},
        )
    )


def _cells(figure: Figure) -> list[str]:
    (ax,) = [a for a in figure.axes if a.get_label() == "heatmap"]
    return [t.get_text() for t in ax.texts]


def test_forty_observables_do_not_overlap() -> None:
    rng = np.random.default_rng(0)
    names = [f"cmax_{i:02d}" for i in range(40)]
    figure = plot_heatmap(_table(rng.random((3, 40)) + 0.5, names), "mu_star")
    figure.draw_without_rendering()
    (ax,) = [a for a in figure.axes if a.get_label() == "heatmap"]
    labels = ax.get_xticklabels()
    width = ax.get_window_extent().width / 40
    if labels[0].get_rotation() == 0:
        boxes = [t.get_window_extent() for t in labels]
        assert not any(a.overlaps(b) for a, b in pairwise(boxes))
    else:
        # rotated by 45 degrees, parallel texts clear each other if the height of
        # the text fits in the distance between them across the slant
        text_height = float(labels[0].get_fontsize()) * float(figure.dpi) / 72
        assert labels[0].get_rotation() == 45
        assert width * np.sin(np.pi / 4) > text_height


def test_an_infinite_value_is_clustered_like_nan() -> None:
    values = np.array([[1.0, np.inf], [1.2, 0.0], [np.nan, 0.5]])
    figure = plot_heatmap(_table(values), "mu_star", cutoff=0)
    assert isinstance(figure, Figure)


def test_a_zero_is_printed_as_zero() -> None:
    figure = plot_heatmap(_table(np.zeros((2, 2))), "mu_star", cutoff=0)
    assert set(_cells(figure)) == {"0"}
    values = np.array([[1.0, -1e-9], [-0.0, 0.5]])
    texts = _cells(plot_heatmap(_table(values), "mu_star", cutoff=0))
    assert sorted(texts) == ["0", "0", "0.50", "1.00"]
    assert not any("-" in t or t[:2] == "\N{MINUS SIGN}0" for t in texts)


def test_a_parameter_without_indices_is_marked() -> None:
    result = _sobol()
    result.ds["auc.S1"].values[1] = np.nan
    result.ds["auc.ST"].values[1] = np.nan
    figure = plot_indices(result, "auc", dose=0)
    (ax,) = figure.axes
    assert [t.get_text() for t in ax.texts] == ["n/a"]
    assert ax.get_xlim()[1] > len(result.parameters) - 1 + 0.2
