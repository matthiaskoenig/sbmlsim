"""Tests of the settings of figures, which cascade from the run to a figure (#260)."""

import copy
import importlib.util
import logging
import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, override

import pytest
from matplotlib.image import imread

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentResult, ExperimentRunner, SimulationExperiment
from sbmlsim.experiment.runner import run_experiments
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Figure, FigureSettings
from sbmlsim.plot.serialization_matplotlib import MatplotlibFigureSerializer
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task

#: the settings of today and their defaults, which were class attributes of
#: `Figure`, and `legend_scope` (0.9.6), which was none
DEFAULTS: dict[str, Any] = {
    "fig_dpi": 72,
    "fig_facecolor": "white",
    "fig_subplots_wspace": 0.3,
    "fig_subplots_hspace": 0.3,
    "panel_width": 7.0,
    "panel_height": 5.0,
    "fig_titlesize": 25,
    "fig_titleweight": "bold",
    "axes_titlesize": 20,
    "axes_titleweight": "bold",
    "axes_labelsize": 18,
    "axes_labelweight": "bold",
    "xtick_labelsize": 15,
    "ytick_labelsize": 15,
    "legend_fontsize": 13,
    "legend_position": "inside",
    "legend_loc": "best",
    "legend_scope": "plot",
}


def _figure(
    experiment: SimulationExperiment | None,
    settings: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Figure:
    """A figure of one panel with a curve of X and a legend."""
    figure = Figure(experiment=experiment, sid="fig", settings=settings, **kwargs)
    plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("X"))[0]
    plot.curve(x=Data("time", task="task"), y=Data("[X]", task="task"), label="X")
    return figure


class _Base(SimulationExperiment):
    """An experiment of the repressilator with one figure."""

    #: the settings of its figure
    fig_settings: Mapping[str, Any] | None = None

    @override
    def models(self) -> dict:
        return {"m": AbstractModel(source=REPRESSILATOR_SBML)}

    @override
    def simulations(self) -> dict:
        return {"sim": Simulation(end=20, steps=20)}

    @override
    def tasks(self) -> dict:
        return {"task": Task(model="m", simulation="sim")}

    @override
    def figures(self) -> dict:
        return {"fig": _figure(self, settings=self.fig_settings)}


class Styled(_Base):
    """An experiment which sets the settings of its figures."""

    figure_settings = MappingProxyType({"legend_fontsize": 10, "fig_dpi": 300})


class Plain(_Base):
    """An experiment which sets nothing."""


def _runner(*classes: type[SimulationExperiment]) -> ExperimentRunner:
    return ExperimentRunner(
        experiment_classes=list(classes),
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )


def _run(
    path: Path, *classes: type[SimulationExperiment], **kwargs: Any
) -> list[ExperimentResult]:
    results = _runner(*classes).run_experiments(
        output_path=path, figure_formats=["png"], keep_results=True, **kwargs
    )
    assert [r.error for r in results] == [None] * len(classes)
    return results


def _legend_fontsize(result: ExperimentResult, settings: Any = None) -> float:
    """Get the font size of the legend of the figure of a run experiment."""
    mpl_figures = result.experiment.create_mpl_figures(figure_settings=settings)
    legend = mpl_figures["fig"].axes[0].get_legend()
    return legend.get_texts()[0].get_fontsize()


def _pixels(path: Path) -> tuple[int, int]:
    """Get the height and the width of a PNG in pixels."""
    height, width = imread(path).shape[:2]
    return height, width


def test_a_figure_renders_the_same_alone_and_after_another_experiment(
    tmp_path: Path,
) -> None:
    """The settings of one experiment do not leak into the figures of another."""
    alone = _run(tmp_path / "alone", Plain)
    both = _run(tmp_path / "both", Styled, Plain)

    plain = _pixels(tmp_path / "alone" / "Plain" / "Plain_fig.png")
    assert _pixels(tmp_path / "both" / "Plain" / "Plain_fig.png") == plain
    assert _legend_fontsize(both[1]) == _legend_fontsize(alone[0]) == 13
    # the experiment which sets them has them
    styled = _pixels(tmp_path / "both" / "Styled" / "Styled_fig.png")
    assert styled[0] > 3 * plain[0] and styled[1] > 3 * plain[1]
    assert _legend_fontsize(both[0]) == 10


def test_the_settings_are_the_attributes_of_figure_of_before() -> None:
    """`Figure.fig_dpi = 300` of before is `fig_dpi=300`, with the same defaults."""
    assert FigureSettings.names() == tuple(DEFAULTS)
    assert vars(FigureSettings()) == DEFAULTS
    assert FigureSettings(fig_dpi=300).fig_dpi == 300
    # the numbers are floats, which ty checks
    settings = FigureSettings(fig_dpi=150.5, legend_fontsize=9.5, axes_labelsize=12.5)
    assert (settings.fig_dpi, settings.legend_fontsize) == (150.5, 9.5)
    with pytest.raises(AttributeError):
        FigureSettings().fig_dpi = 300  # ty: ignore[invalid-assignment]


def test_the_more_specific_level_wins() -> None:
    """The defaults, the run, the experiment and the figure, in this order."""
    run: dict[str, Any] = {"fig_dpi": 100, "legend_fontsize": 11, "axes_labelsize": 12}
    experiment = {"legend_fontsize": 9, "axes_labelsize": 14}
    figure = {"axes_labelsize": 16}
    assert FigureSettings.resolve() == FigureSettings()
    assert FigureSettings.resolve(run=run) == FigureSettings(**run)
    resolved = FigureSettings.resolve(run=run, experiment=experiment, figure=figure)
    assert resolved == FigureSettings(fig_dpi=100, legend_fontsize=9, axes_labelsize=16)
    # a `FigureSettings` of the run is a complete set
    full = FigureSettings(fig_dpi=150, legend_position="outside")
    assert FigureSettings.resolve(run=full, figure={"fig_dpi": 50}) == FigureSettings(
        fig_dpi=50, legend_position="outside"
    )


def test_a_figure_resolves_the_settings_of_its_experiment() -> None:
    """`resolved_settings` reads `figure_settings` of the experiment it is given."""

    class Experiment(SimulationExperiment):
        figure_settings = MappingProxyType({"legend_fontsize": 9, "axes_labelsize": 14})

    figure = Figure(experiment=None, sid="fig", settings={"axes_labelsize": 16})
    run = {"fig_dpi": 100, "legend_fontsize": 11}
    assert figure.resolved_settings(Experiment(), run=run) == FigureSettings(
        fig_dpi=100, legend_fontsize=9, axes_labelsize=16
    )
    assert figure.resolved_settings(None, run=run) == FigureSettings(
        fig_dpi=100, legend_fontsize=11, axes_labelsize=16
    )
    assert figure.resolved_settings() == FigureSettings(axes_labelsize=16)


def test_a_run_renders_with_the_cascade(tmp_path: Path) -> None:
    """The settings of a run reach every figure, the experiment's override them."""
    results = _run(tmp_path / "run", Styled, Plain, figure_settings={"fig_dpi": 144})
    alone = _run(tmp_path / "alone", Styled, Plain)
    plain = _pixels(tmp_path / "run" / "Plain" / "Plain_fig.png")
    expected = _pixels(tmp_path / "alone" / "Plain" / "Plain_fig.png")
    assert plain[0] == pytest.approx(2 * expected[0], abs=4)
    assert plain[1] == pytest.approx(2 * expected[1], abs=4)
    # the experiment's dpi wins over the run's
    styled = _pixels(tmp_path / "run" / "Styled" / "Styled_fig.png")
    assert styled == _pixels(tmp_path / "alone" / "Styled" / "Styled_fig.png")
    assert _legend_fontsize(alone[1]) == 13
    assert _legend_fontsize(results[0], {"legend_fontsize": 20}) == 10
    assert _legend_fontsize(results[1], {"legend_fontsize": 20}) == 20


def test_the_figure_overrides_its_experiment(tmp_path: Path) -> None:
    """`Figure(settings=...)` wins over `figure_settings` of the experiment."""

    class FigureStyled(Styled):
        fig_settings = MappingProxyType({"legend_fontsize": 8})

    results = _run(tmp_path, FigureStyled)
    assert _legend_fontsize(results[0]) == 8
    mpl = results[0].experiment.create_mpl_figures()["fig"]
    assert mpl.dpi == 300


def test_simulation_experiment_run_takes_figure_settings(tmp_path: Path) -> None:
    """`SimulationExperiment.run(figure_settings=...)` renders with them."""
    experiment = _runner(Plain).experiments["Plain"]
    experiment.run(
        Simulator(),
        output_path=tmp_path,
        figure_formats=["png"],
        figure_settings=FigureSettings(fig_dpi=144),
    )
    experiment.run(
        Simulator(), output_path=tmp_path / "default", figure_formats=["png"]
    )
    wide = _pixels(tmp_path / "Plain_fig.png")
    default = _pixels(tmp_path / "default" / "Plain_fig.png")
    assert wide[1] == pytest.approx(2 * default[1], abs=4)


def test_run_experiments_takes_figure_settings(tmp_path: Path) -> None:
    """The function `run_experiments` passes the settings to every experiment."""
    run_experiments(
        Plain,
        output_path=tmp_path,
        base_path=Path("."),
        data_path=Path("."),
        figure_settings={"panel_width": 14.0},
    )
    run_experiments(
        Plain,
        output_path=tmp_path / "default",
        base_path=Path("."),
        data_path=Path("."),
    )
    wide = _svg_width(tmp_path / "Plain" / "Plain_fig.svg")
    default = _svg_width(tmp_path / "default" / "Plain" / "Plain_fig.svg")
    # the axes take most of the 7 inches of 72 points more
    assert wide - default > 5 * 72


def _svg_width(path: Path) -> float:
    """Get the width of an SVG in points."""
    match = re.search(r'<svg[^>]*width="([0-9.]+)pt"', path.read_text(encoding="utf-8"))
    assert match is not None
    return float(match.group(1))


@pytest.mark.parametrize(
    "where",
    ["run", "runner", "run_experiments", "serializer", "experiment", "figure"],
)
def test_an_unknown_key_raises_naming_the_settings(tmp_path: Path, where: str) -> None:
    """A key which is no setting raises at every level and names the settings."""
    unknown = {"legend_size": 10}
    match = r"legend_size.*fig_dpi, fig_facecolor, .*legend_position, legend_loc"
    with pytest.raises(ValueError, match=match):
        if where == "run":
            experiment = _runner(Plain).experiments["Plain"]
            experiment.run(Simulator(), figure_settings=unknown, on_error="log")
        elif where == "runner":
            _runner(Plain).run_experiments(tmp_path, figure_settings=unknown)
        elif where == "run_experiments":
            run_experiments(
                Plain, tmp_path, Path("."), Path("."), figure_settings=unknown
            )
        elif where == "serializer":
            figure = Figure(experiment=None, sid="fig")
            MatplotlibFigureSerializer.to_figure(None, figure, settings=unknown)
        elif where == "experiment":

            class Unknown(Plain):
                figure_settings = MappingProxyType(unknown)

            ExperimentRunner(Unknown, Path("."), Path("."), on_error="raise")
        else:
            Figure(experiment=None, sid="fig", settings=unknown)
    # nothing ran: the error is raised before any experiment
    assert not (tmp_path / "Plain").exists()


def test_an_unknown_key_of_an_experiment_fails_its_initialization(
    tmp_path: Path,
) -> None:
    """The experiment's settings are checked at `initialize`, the runner records it."""

    class Unknown(Plain):
        figure_settings = MappingProxyType({"legend_size": 10})

    runner = _runner(Unknown, Plain)
    assert "legend_size" in str(runner.failed["Unknown"])
    results = runner.run_experiments(tmp_path)
    assert results[0].error is not None and "legend_size" in results[0].error
    assert results[1].error is None


def test_the_settings_of_an_experiment_are_a_mapping() -> None:
    """A `FigureSettings` is a complete set, which belongs to the run."""

    class Complete(Plain):
        figure_settings = FigureSettings(fig_dpi=300)  # ty: ignore[invalid-assignment]

    with pytest.raises(TypeError, match="experiment 'Complete' are a mapping"):
        ExperimentRunner(Complete, Path("."), Path("."), on_error="raise")


def test_the_settings_of_the_experiment_class_are_read_only() -> None:
    """The default cannot be changed in place, a subclass assigns its own."""
    assert dict(SimulationExperiment.figure_settings) == {}
    with pytest.raises(TypeError):
        SimulationExperiment.figure_settings["fig_dpi"] = 300  # ty: ignore[invalid-assignment]
    assert dict(Plain.figure_settings) == {}
    assert dict(Styled.figure_settings) == {"legend_fontsize": 10, "fig_dpi": 300}


def test_the_settings_of_a_figure_are_read_only_and_checked() -> None:
    """`Figure.settings` is a read-only copy, an assignment is checked."""
    overrides = {"fig_dpi": 300}
    figure = Figure(experiment=None, sid="fig", settings=overrides)
    overrides["fig_dpi"] = 100
    assert dict(figure.settings) == {"fig_dpi": 300}
    with pytest.raises(TypeError):
        figure.settings["fig_dpi"] = 100  # ty: ignore[invalid-assignment]
    figure.settings = {"legend_fontsize": 10}
    assert dict(figure.settings) == {"legend_fontsize": 10}
    with pytest.raises(ValueError, match="the figure 'fig'"):
        figure.settings = {"legend_size": 10}
    figure.settings = None
    assert dict(figure.settings) == {}


@pytest.mark.parametrize("settings", [[], "", ()])
def test_settings_of_a_figure_which_are_no_mapping_raise(settings: Any) -> None:
    """A falsy value which is no mapping is not taken for no settings."""
    with pytest.raises(TypeError, match="the figure 'fig' are a mapping"):
        Figure(experiment=None, sid="fig", settings=settings)


@pytest.mark.parametrize("name", list(DEFAULTS))
def test_a_setting_on_the_class_or_a_figure_fails_loudly(name: str) -> None:
    """Setting or reading a setting on `Figure` or a figure raises with the way now."""
    figure = Figure(experiment=None, sid="fig")
    parts = [
        "FigureSettings",
        "figure_settings=",
        "SimulationExperiment.figure_settings",
        "figure_settings = MappingProxyType({",
        "Figure(settings=...)",
        f"FigureSettings().{name}",
        # a figure which exists, and a figure of `figures_mpl()`
        f"fig.settings = {{**fig.settings, '{name}': <value>}}",
        f"FigureSettings.resolve(experiment=self.figure_settings).{name}",
    ]
    checks = [
        lambda: setattr(Figure, name, 1),
        lambda: getattr(Figure, name),
        lambda: setattr(figure, name, 1),
        lambda: getattr(figure, name),
    ]
    for check in checks:
        with pytest.raises(AttributeError) as info:
            check()
        for part in parts:
            assert part in str(info.value)
    # nothing was set
    assert name not in vars(Figure)
    assert name not in vars(figure)
    assert not hasattr(Figure, name)


def test_a_setting_at_the_module_level_raises_at_import(tmp_path: Path) -> None:
    """`Figure.legend_fontsize = 10` of an experiment module of before raises."""
    path = tmp_path / "old_experiment.py"
    path.write_text(
        "from sbmlsim.plot import Figure\n\nFigure.legend_fontsize = 10\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("old_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(
        AttributeError,
        match=re.escape("figure_settings = MappingProxyType({'legend_fontsize'"),
    ):
        spec.loader.exec_module(module)


def test_a_setting_in_the_body_of_a_subclass_fails_loudly() -> None:
    """A setting defined in a subclass of `Figure` would be ignored, it raises."""
    match = r"'legend_fontsize' is no attribute.*SimulationExperiment\.figure_settings"
    with pytest.raises(AttributeError, match=match):

        class LegendFigure(Figure):
            legend_fontsize = 10

    for name in DEFAULTS:
        with pytest.raises(AttributeError, match=f"'{name}' is no attribute"):
            type("MyFigure", (Figure,), {name: 10})

    class MyFigure(Figure):
        """A subclass with attributes of its own is fine."""

        color = "red"

    assert MyFigure(experiment=None, sid="fig").color == "red"


def test_other_attributes_of_a_figure_are_as_before() -> None:
    """Other attributes are set and read, an unknown one raises as usual."""
    figure = Figure(experiment=None, sid="fig")
    figure.width = 10
    assert figure.width == 10
    assert Figure.__name__ == "Figure"
    with pytest.raises(AttributeError, match="'Figure' object has no attribute 'nope'"):
        _ = figure.nope  # ty: ignore[unresolved-attribute]
    with pytest.raises(AttributeError, match="type object 'Figure' has no attribute"):
        _ = Figure.nope  # ty: ignore[unresolved-attribute]


def test_a_figure_is_copied_and_pickled_with_its_settings() -> None:
    """The attribute guards keep `copy` and `pickle` working."""
    figure = Figure(experiment=None, sid="fig", width=3.0, settings={"fig_dpi": 300})
    for other in [
        copy.copy(figure),
        copy.deepcopy(figure),
        pickle.loads(pickle.dumps(figure)),
    ]:
        assert dict(other.settings) == {"fig_dpi": 300}
        assert other.width == 3.0
        assert other.resolved_settings().fig_dpi == 300


def test_an_explicit_size_is_kept_and_the_default_is_computed_at_rendering() -> None:
    """The size of the panels is read when the figure is rendered."""
    explicit = Figure(experiment=None, sid="fig", num_cols=2, width=3.0, height=2.0)
    default = Figure(experiment=None, sid="fig", num_cols=2, num_rows=3)
    partial = Figure(experiment=None, sid="fig", num_cols=2, width=3.0)
    assert default.width is None and default.height is None
    small = FigureSettings(panel_width=2.0, panel_height=1.0)

    assert explicit.size(FigureSettings()) == explicit.size(small) == (3.0, 2.0)
    assert default.size(FigureSettings()) == (14.0, 15.0)
    assert default.size(small) == (4.0, 3.0)
    assert partial.size(small) == (3.0, 1.0)

    for figure, expected in [(explicit, (3.0, 2.0)), (default, (4.0, 3.0))]:
        fig = MatplotlibFigureSerializer.to_figure(None, figure, settings=small)
        assert tuple(fig.get_size_inches()) == pytest.approx(expected)
    fig = MatplotlibFigureSerializer.to_figure(None, default)
    assert tuple(fig.get_size_inches()) == pytest.approx((14.0, 15.0))

    # a figure of the experiment gets the panels of the experiment
    class Panels(SimulationExperiment):
        figure_settings = MappingProxyType({"panel_width": 3.0, "panel_height": 2.0})

    fig = MatplotlibFigureSerializer.to_figure(Panels(), default)
    assert tuple(fig.get_size_inches()) == pytest.approx((6.0, 6.0))


def test_the_pages_of_plotly_have_the_size_of_the_settings(tmp_path: Path) -> None:
    """The pixels of a page are the inches of the figure times 72, whatever `fig_dpi`."""
    pytest.importorskip("plotly")
    from sbmlsim.plot.serialization_plotly import PlotlyFigureSerializer

    explicit = Figure(experiment=None, sid="fig", width=3.0, height=2.0)
    default = Figure(experiment=None, sid="fig", num_cols=2)
    run = {"fig_dpi": 100, "panel_width": 4.0}
    for figure, expected in [
        (explicit, (216, 144)),
        (default, (576, 360)),
    ]:
        layout = PlotlyFigureSerializer.to_figure(None, figure, settings=run).layout
        assert (layout.width, layout.height) == expected
    layout = PlotlyFigureSerializer.to_figure(None, default).layout
    assert (layout.width, layout.height) == (1008, 360)


def test_a_run_at_300_dpi_has_pages_of_the_default_size(tmp_path: Path) -> None:
    """`fig_dpi` is the resolution of the static images, not the pixels of a page."""
    pytest.importorskip("plotly")
    _runner(Plain).run_experiments(tmp_path / "default", figure_formats=["html"])
    page = _page_size(tmp_path / "default" / "Plain" / "Plain_fig.html")
    assert page == (504, 360)

    # the dpi of the run and the dpi of the experiment
    _runner(Styled, Plain).run_experiments(
        tmp_path / "dpi",
        figure_formats=["html", "png"],
        figure_settings={"fig_dpi": 300},
    )
    for sid in ["Plain", "Styled"]:
        assert _page_size(tmp_path / "dpi" / sid / f"{sid}_fig.html") == page
        # the image has the resolution, about 300 / 72 times the default pixels
        height, width = _pixels(tmp_path / "dpi" / sid / f"{sid}_fig.png")
        assert width > 3 * page[0] and height > 3 * page[1]


def _page_size(path: Path) -> tuple[int, int]:
    """Get the width and the height of the layout of a plotly page."""
    layout = path.read_text(encoding="utf-8").split('"layout":')[1]
    width = re.search(r'"width":(\d+)', layout)
    height = re.search(r'"height":(\d+)', layout)
    assert width is not None and height is not None
    return int(width.group(1)), int(height.group(1))


class OldSignature(Plain):
    """An experiment whose override of `run` has no `figure_settings`."""

    @override
    def run(
        self,
        simulator: Simulator | None = None,
        output_path: Path | None = None,
        figure_formats: list[str] | None = None,
    ) -> ExperimentResult:
        return super().run(
            simulator=simulator, output_path=output_path, figure_formats=figure_formats
        )


def test_an_override_of_run_without_figure_settings_is_run_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The settings of the run are not passed to a `run` which does not take them."""
    with caplog.at_level(logging.WARNING):
        results = _run(tmp_path, OldSignature, Plain, figure_settings={"fig_dpi": 144})
    assert "'OldSignature' does not take `figure_settings`" in caplog.text
    old = _pixels(tmp_path / "OldSignature" / "OldSignature_fig.png")
    plain = _pixels(tmp_path / "Plain" / "Plain_fig.png")
    assert plain[1] > 1.5 * old[1]
    assert results[0].error is None
    for settings in [None, {}, FigureSettings(), {"fig_dpi": 72}]:
        # a run which sets nothing, the override misses nothing
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _run(tmp_path / "default", OldSignature, figure_settings=settings)
        assert "figure_settings" not in caplog.text


def test_the_migration_of_a_figure_which_exists_works() -> None:
    """`fig.settings = {**fig.settings, ...}` of the message sets one setting."""
    figure = Figure(experiment=None, sid="fig", settings={"legend_fontsize": 9.5})
    figure.settings = {**figure.settings, "legend_position": "outside"}
    resolved = figure.resolved_settings()
    assert (resolved.legend_fontsize, resolved.legend_position) == (9.5, "outside")


def test_the_scope_of_the_legends_is_a_plot_or_a_row() -> None:
    """A legend per plot or per row; another value raises (#327)."""
    for scope in ("plot", "row"):
        assert (
            FigureSettings.resolve(figure={"legend_scope": scope}).legend_scope == scope
        )
    for scope in ("rows", "figure"):
        with pytest.raises(ValueError, match=r"legend_scope.*\['plot', 'row'\]"):
            FigureSettings.resolve(figure={"legend_scope": scope})
