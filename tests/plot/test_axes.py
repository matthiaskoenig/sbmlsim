"""The limits of the log axes of a figure of matplotlib."""

from matplotlib.figure import Figure


def test_an_automatic_log_axis_of_a_custom_figure_is_limited() -> None:
    """The tiny first steps of the integrator do not stretch a log axis."""
    from sbmlsim.plot.axes import limit_log_axes

    figure = Figure()
    free, fixed = figure.subplots(1, 2)
    for ax in (free, fixed):
        ax.plot([0.0, 1e-9, 1.0, 2.0], [0.0, 1e-40, 10.0, 1.0])
        ax.set_yscale("log")
    fixed.set_ylim(1e-50, 100.0)
    limit_log_axes(figure)
    # six decades below the largest value, with the margin of the decades
    assert 1e-6 < free.get_ylim()[0] < 1e-5
    assert fixed.get_ylim() == (1e-50, 100.0)
