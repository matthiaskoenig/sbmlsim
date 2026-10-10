"""Curves over scan points and bands in the figure model, checked at initialize."""

from pathlib import Path

import pytest

from sbmlsim.data import Data
from sbmlsim.experiment import ExperimentRunner, SimulationExperiment
from sbmlsim.plot import Axis, Band, Curve, Figure, Plot
from sbmlsim.plot.plotting import CurveType
from sbmlsim.simulator import Simulator
from tests.plot.scan_experiment import ScanFigures


def _runner(experiment_class: type[SimulationExperiment]) -> ExperimentRunner:
    return ExperimentRunner(
        experiment_classes=[experiment_class],
        simulator=Simulator(),
        base_path=Path("."),
        data_path=Path("."),
    )


def _with_figure(draw) -> type[SimulationExperiment]:
    """An experiment of the scans with one plot which `draw(plot)` fills."""

    class WithFigure(ScanFigures):
        def figures(self) -> dict:
            figure = Figure(experiment=self, sid="fig", num_rows=1, num_cols=1)
            plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("C"))[0]
            draw(plot)
            return {"fig": figure}

    return WithFigure


def test_a_curve_and_a_band_know_their_dimensions() -> None:
    curve = Curve(x=Data("time", task="t"), y=Data("[C]", task="t"), over="dose")
    assert curve.over == ("dose",) and curve.to_dict()["over"] == ["dose"]
    assert Curve(x=Data("time", task="t"), y=Data("[C]", task="t")).over == ()
    band = Band(
        Data("time", task="t"), Data("[C]", task="t"), across="draw", over=["dose"]
    )
    d = band.to_dict()
    assert (
        d["across"] == "draw"
        and d["quantiles"] == [0.05, 0.95]
        and d["over"] == ["dose"]
    )
    with pytest.raises(ValueError, match="quantiles"):
        Band(
            Data("time", task="t"),
            Data("[C]", task="t"),
            across="draw",
            quantiles=(0.9, 0.1),
        )


def test_plot_band_and_curve_over() -> None:
    def draw(plot) -> None:
        plot.curve(
            x=Data("time", task="task_doses"),
            y=Data("[C]", task="task_doses"),
            over="dose",
        )
        band = plot.band(
            Data("time", task="task_draws"),
            Data("[C]", task="task_draws"),
            across="draw",
        )
        assert band.sid == f"{plot.sid}_band0" and plot.bands == [band]

    experiment = _runner(_with_figure(draw)).experiments["WithFigure"]
    plot = experiment._figures["fig"].get_plots()[0]
    assert plot.curves[0].over == ("dose",) and len(plot.bands) == 1


def test_a_scan_dimension_which_is_not_named_raises_at_initialize() -> None:
    def draw(plot) -> None:
        plot.curve(x=Data("time", task="task_doses"), y=Data("[C]", task="task_doses"))

    with pytest.raises(
        ValueError,
        match=r"y of curve '.*' has the dimension 'dose'; name it with over='dose'",
    ):
        _runner(_with_figure(draw))


def test_over_a_dimension_the_data_has_not_raises_at_initialize() -> None:
    def draw(plot) -> None:
        plot.curve(
            x=Data("time", task="task_doses"),
            y=Data("[C]", task="task_doses"),
            over="nope",
        )

    with pytest.raises(ValueError, match=r"draws a line per point of \['nope'\]"):
        _runner(_with_figure(draw))


def test_a_value_per_simulation_over_its_dimension_passes() -> None:
    def draw(plot) -> None:
        plot.curve(
            x=Data("dose.PODOSE", task="task_doses"),
            y=Data("pk.cmax", task="task_doses"),
        )

    _runner(_with_figure(draw))


def test_more_than_four_points_of_a_second_dimension_raise() -> None:
    def draw(plot) -> None:
        plot.curve(
            x=Data("time", task="task_grid2"),
            y=Data("[C]", task="task_grid2"),
            over=("dose", "rate"),
        )

    _runner(_with_figure(draw))  # two points of ke: fine

    def draw_five(plot) -> None:
        plot.curve(
            x=Data("time", task="task_grid5"),
            y=Data("[C]", task="task_grid5"),
            over=("dose", "rate"),
        )

    with pytest.raises(ValueError, match="at most 4"):
        _runner(_with_figure(draw_five))


def test_a_dimension_named_twice_and_a_band_over_two_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="twice"):
        Curve(x=Data("time", task="t"), y=Data("[C]", task="t"), over=("dose", "dose"))
    with pytest.raises(ValueError, match="one dimension"):
        Band(
            Data("time", task="t"),
            Data("[C]", task="t"),
            across="draw",
            over=("dose", "rate"),
        )


def test_a_band_needs_its_dimension_and_names_the_others() -> None:
    def no_across(plot) -> None:
        plot.band(
            Data("time", task="task_doses"),
            Data("[C]", task="task_doses"),
            across="draw",
        )

    with pytest.raises(ValueError, match="reduces the dimension 'draw'"):
        _runner(_with_figure(no_across))

    def unnamed(plot) -> None:
        plot.band(
            Data("time", task="task_dose_draws"),
            Data("[C]", task="task_dose_draws"),
            across="draw",
        )

    with pytest.raises(ValueError, match="name it with over='dose'"):
        _runner(_with_figure(unnamed))

    def named(plot) -> None:
        plot.band(
            Data("time", task="task_dose_draws"),
            Data("[C]", task="task_dose_draws"),
            across="draw",
            over="dose",
        )

    _runner(_with_figure(named))


def test_a_band_of_a_ragged_scan_raises_at_initialize() -> None:
    def draw(plot) -> None:
        plot.band(
            Data("time", task="task_ragged"),
            Data("[C]", task="task_ragged"),
            across="dose",
        )

    with pytest.raises(ValueError, match="common grid"):
        _runner(_with_figure(draw))


def test_the_data_of_a_band_counts_for_the_selections() -> None:
    class BandOnly(ScanFigures):
        def data(self) -> dict:
            return {}

        def figures(self) -> dict:
            figure = Figure(experiment=self, sid="fig", num_rows=1, num_cols=1)
            plot = figure.create_plots(xaxis=Axis("time"), yaxis=Axis("C"))[0]
            plot.band(
                Data("time", task="task_draws"),
                Data("[C]", task="task_draws"),
                across="draw",
            )
            return {"fig": figure}

    runner = _runner(BandOnly)
    experiment = runner.experiments["BandOnly"]
    experiment.run(runner.simulator)
    assert "[C]" in experiment.results["task_draws"].ds.data_vars


def test_the_limit_of_a_second_dimension_counts_the_selected_points() -> None:
    def two_of_five(plot) -> None:
        plot.curve(
            x=Data("time", task="task_grid5"),
            y=Data("[C]", task="task_grid5", sel={"rate": [0, 1]}),
            over=("dose", "rate"),
        )

    _runner(_with_figure(two_of_five))

    def all_five(plot) -> None:
        plot.curve(
            x=Data("time", task="task_grid5"),
            y=Data("[C]", task="task_grid5"),
            over=("dose", "rate"),
        )

    with pytest.raises(ValueError, match="at most 4"):
        _runner(_with_figure(all_five))


def test_a_band_which_names_a_dimension_twice_is_called_a_band() -> None:
    with pytest.raises(ValueError, match="The band 'b' names a dimension twice"):
        Band(
            Data("time", task="t"),
            Data("[C]", task="t"),
            across="draw",
            over=("dose", "dose"),
            sid="b",
        )
    with pytest.raises(ValueError, match="The curve 'C' names a dimension twice"):
        Curve(
            x=Data("time", task="t"),
            y=Data("[C]", task="t"),
            over=("dose", "dose"),
            label="C",
        )


def test_the_representation_of_a_curve() -> None:
    curve = Curve(x=Data("time", task="t"), y=Data("[C]", task="t"), over="dose")
    assert "over=('dose',) xerr=False yerr=False" in repr(curve)


def test_a_bar_curve_over_a_dimension_raises_when_it_is_defined() -> None:
    with pytest.raises(ValueError, match="bar curve, which draws no line per point"):
        Curve(
            x=Data("time", task="t"),
            y=Data("[C]", task="t"),
            over="dose",
            type=CurveType.BAR,
        )


def test_plot_band_takes_an_id_and_an_order() -> None:
    plot = Plot(sid="p")
    band = plot.band(
        Data("time", task="t"),
        Data("[C]", task="t"),
        across="draw",
        sid="mine",
        order=3,
    )
    assert band.sid == "mine" and band.order == 3
