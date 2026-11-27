"""The figures of an experiment are drawn by the renderer of sbmlsim-core.

A figure of the figure model is resolved into a `Chart` and written as SVG,
PNG on request, and its spec as JSON beside every figure; `figures_mpl()` is a
hook of charts, and a figure of it which is no chart fails under its key.
"""

import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, override

import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.figure_formats import FIGURE_FORMATS, figure_formats_of
from sbmlsim.model import AbstractModel
from sbmlsim.plot import Axis, Chart, Figure
from sbmlsim.project import FigureOutputs
from sbmlsim.resources import REPRESSILATOR_SBML
from sbmlsim.simulation import Simulation
from sbmlsim.simulator import Simulator
from sbmlsim.task import Task

#: the message of a run which asks for the pages of plotly
HTML_GONE = (
    "the format html is gone with plotly; the report shows the svg and becomes "
    "interactive with the renderer"
)


class _Repressilator(SimulationExperiment):
    """An experiment of the repressilator with one figure."""

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
        return {"fig": _figure(self, "fig")}


def _figure(
    experiment: SimulationExperiment, sid: str, settings: Any = None, **style: Any
) -> Figure:
    """A figure of the selection `[X]` over the time, which a bad style breaks."""
    figure = Figure(experiment=experiment, sid=sid, settings=settings)
    plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("X"))[0]
    plot.curve(
        x=Data("time", task="task"), y=Data("[X]", task="task"), label="X", **style
    )
    return figure


def _experiment(experiment_class: type[SimulationExperiment]) -> Any:
    """An experiment initialized by a runner, which loads its models."""
    runner = ExperimentRunner(
        [experiment_class],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )
    return runner.experiments[experiment_class.__name__], runner.simulator


def _run(experiment_class: type[SimulationExperiment], out: Path, **kwargs: Any) -> Any:
    """Initialize and run an experiment into `out`."""
    experiment, simulator = _experiment(experiment_class)
    return experiment, experiment.run(simulator, output_path=out, **kwargs)


def _png_width(path: Path) -> int:
    """The width in pixels of a PNG, from its header."""
    return struct.unpack(">I", path.read_bytes()[16:20])[0]


def test_figures_written_as_svg_png_json(tmp_path: Path) -> None:
    """A figure is an SVG, a PNG on request and its spec as JSON beside it."""
    experiment, result = _run(_Repressilator, tmp_path, figure_formats=["svg", "PNG"])

    stem = tmp_path / f"{experiment.sid}_fig"
    assert (
        stem.with_suffix(".svg")
        .read_text(encoding="utf-8")
        .lstrip()
        .startswith(("<svg", "<?xml"))
    )
    assert stem.with_suffix(".png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    spec = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert spec["plots"]
    assert result.figures == {"fig": ["svg", "png", "json"]}
    assert not result.failed_figures


def test_figures_are_svg_and_json_by_default(tmp_path: Path) -> None:
    """Without formats a figure is an SVG with its spec and no PNG."""
    experiment, result = _run(_Repressilator, tmp_path)

    assert result.figures == {"fig": ["svg", "json"]}
    assert not (tmp_path / f"{experiment.sid}_fig.png").exists()


def test_the_formats_of_a_run() -> None:
    """The formats are lower case, without duplicates and end with the spec."""
    assert FIGURE_FORMATS == ("svg", "png", "json")
    assert figure_formats_of(None) == ["svg", "json"]
    assert figure_formats_of(["PNG", "svg", "png"]) == ["png", "svg", "json"]
    assert figure_formats_of(["json"]) == ["json"]
    with pytest.raises(ValueError, match="pdf"):
        figure_formats_of(["pdf"])
    with pytest.raises(TypeError):
        figure_formats_of("svg")


def test_html_format_refused(tmp_path: Path) -> None:
    """A run which asks for the pages of plotly is refused before it runs."""
    experiment, simulator = _experiment(_Repressilator)
    with pytest.raises(ValueError) as error:
        experiment.run(
            simulator,
            output_path=tmp_path,
            figure_formats=["svg", "html"],
            on_error="log",
        )
    assert str(error.value) == HTML_GONE
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="html is gone"):
        FigureOutputs(formats=("svg", "html"))


def test_png_dpi(tmp_path: Path) -> None:
    """A PNG has the resolution of its figure, which a given dpi replaces."""

    class Dpi(_Repressilator):
        @override
        def figures(self) -> dict:
            return {"fig": _figure(self, "fig", settings={"fig_dpi": 144})}

    experiment, _ = _run(Dpi, tmp_path / "figure", figure_formats=["png"])
    charts = experiment.create_figures()
    width = charts["fig"].width
    path = tmp_path / "figure" / f"{experiment.sid}_fig.png"
    assert _png_width(path) == pytest.approx(width * 144 / 72, abs=2)

    paths = experiment.save_figures(tmp_path / "given", charts, ["png"], dpi=36)
    assert {p.suffix for p in paths["fig"]} == {".png", ".json"}
    given = tmp_path / "given" / f"{experiment.sid}_fig.png"
    assert _png_width(given) == pytest.approx(width * 36 / 72, abs=2)


def _chart() -> Chart:
    """A chart of one line."""
    chart = Chart(title="drawn")
    chart.plot().curve([0.0, 1.0], [0.0, 1.0], label="line")
    return chart


class _Saved:
    """A figure of another library, which saves itself, e.g. one of matplotlib."""

    def savefig(self, path: Path, **kwargs: Any) -> None:
        Path(path).write_text("figure", encoding="utf-8")


class _Hooked(_Repressilator):
    """An experiment with a chart of its own in `figures_mpl`."""

    @override
    def figures_mpl(self) -> dict[str, Any]:
        return {"drawn": _chart()}


class _Foreign(_Repressilator):
    """An experiment whose `figures_mpl` returns a figure which is no chart."""

    @override
    def figures_mpl(self) -> dict[str, Any]:
        return {"custom": _Saved(), "drawn": _chart()}


def test_a_chart_of_the_hook_is_drawn_by_the_renderer(tmp_path: Path) -> None:
    """A chart of the hook is written with its spec, as the other figures."""
    experiment, result = _run(_Hooked, tmp_path, figure_formats=["svg", "png"])

    stem = tmp_path / f"{experiment.sid}_drawn"
    assert stem.with_suffix(".svg").read_text(encoding="utf-8").lstrip()[:5] in (
        "<svg ",
        "<?xml",
    )
    assert stem.with_suffix(".png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    assert result.figures == {
        "fig": ["svg", "png", "json"],
        "drawn": ["svg", "png", "json"],
    }
    assert not result.failed_figures


def test_a_figure_of_the_hook_which_is_no_chart_fails(tmp_path: Path) -> None:
    """A figure of matplotlib or any other library is refused, naming `Chart`.

    It is not saved, a file of it from an earlier run is removed, and the
    charts of the hook are written.
    """
    sid = _experiment(_Foreign)[0].sid
    stale = tmp_path / f"{sid}_custom.svg"
    stale.write_text("an earlier run", encoding="utf-8")

    experiment, result = _run(_Foreign, tmp_path, on_error="log")

    assert set(result.failed_figures) == {"custom"}
    message = result.failed_figures["custom"]
    assert message.startswith("TypeError: The figure 'custom' of ")
    assert "_Saved, not a sbmlsim.plot.Chart" in message
    assert result.figures == {"fig": ["svg", "json"], "drawn": ["svg", "json"]}
    assert not list(tmp_path.glob(f"{experiment.sid}_custom.*"))


def test_a_figure_of_the_hook_which_is_no_chart_raises(tmp_path: Path) -> None:
    """With `on_error="raise"` the error of the figure is raised."""
    with pytest.raises(TypeError, match=r"not a sbmlsim\.plot\.Chart"):
        _run(_Foreign, tmp_path)


def test_figure_failure_reported(tmp_path: Path) -> None:
    """A figure whose data raises is listed as failed, the others are written."""

    class Broken(_Repressilator):
        @override
        def figures(self) -> dict:
            return {
                "fig": _figure(self, "fig"),
                "bad": _figure(self, "bad", style="not a style"),
            }

    experiment, result = _run(Broken, tmp_path, on_error="log")

    assert set(result.failed_figures) == {"bad"}
    assert result.figures == {"fig": ["svg", "json"]}
    assert (tmp_path / f"{experiment.sid}_fig.svg").exists()
    assert not list(tmp_path.glob(f"{experiment.sid}_bad.*"))


def test_rc_left_figure_outputs() -> None:
    """The parameters of matplotlib are no setting of the figures any more."""
    with pytest.raises(ValueError, match="rc"):
        FigureOutputs(rc={"font.family": "serif"})


#: run the HCTZ experiments and report the plotting libraries imported
_NO_MATPLOTLIB = """
import json, sys
from pathlib import Path
from examples.hctz_fitting.project import PROJECT

results = PROJECT.run_experiments(
    None, Path(sys.argv[1]), n_workers=1, figure_formats=["svg", "png"],
    progress=False, collect_figures=True,
)
print(json.dumps({
    "figures": sum(len(r.figures) for r in results),
    "modules": [m for m in ("matplotlib", "seaborn", "plotly") if m in sys.modules],
}))
"""


def test_no_matplotlib_in_a_run(tmp_path: Path) -> None:
    """A run of experiments draws its figures without any plotting library."""
    root = Path(__file__).parents[2]
    process = subprocess.run(
        [sys.executable, "-c", _NO_MATPLOTLIB, str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert process.returncode == 0, process.stderr[-3000:]
    outcome = json.loads(process.stdout.strip().splitlines()[-1])
    assert outcome["figures"] > 0
    assert outcome["modules"] == []
    assert list((tmp_path / "_figures").glob("*.svg"))
    assert list((tmp_path / "_figures").glob("*.png"))


def test_a_chart_of_the_hook_has_the_resolution_of_the_run(tmp_path: Path) -> None:
    """`png_dpi` reaches the charts of the hook."""
    widths = {}
    for dpi in (36, 72):
        experiment, result = _run(
            _Hooked, tmp_path / str(dpi), figure_formats=["png"], png_dpi=dpi
        )
        assert result.figures["drawn"] == ["png", "json"]
        widths[dpi] = _png_width(tmp_path / str(dpi) / f"{experiment.sid}_drawn.png")
    assert widths[72] == pytest.approx(2 * widths[36], abs=2)


def test_a_chart_of_the_hook_is_written_as_its_spec_alone(tmp_path: Path) -> None:
    """A run of the spec alone writes the spec of a chart of the hook."""
    experiment, result = _run(_Hooked, tmp_path, figure_formats=["json"])
    assert result.figures == {"fig": ["json"], "drawn": ["json"]}
    assert (tmp_path / f"{experiment.sid}_drawn.json").exists()
    assert not list(tmp_path.glob(f"{experiment.sid}_drawn.svg"))


def test_show_figures_shows_the_hook_figures_without_an_output_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a notebook the charts of the figures and of the hook are displayed."""
    import types

    shown: list[Any] = []
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: object()  # ty: ignore[missing-attribute]
    display = types.ModuleType("IPython.display")
    display.display = shown.append  # ty: ignore[missing-attribute]
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)

    experiment, simulator = _experiment(_Hooked)
    experiment.run(simulator, show_figures=True)
    assert [type(figure).__name__ for figure in shown] == ["Chart", "Chart"]
