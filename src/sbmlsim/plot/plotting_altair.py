"""
Interactive plots with altair
"""
import altair as alt

from sbmlsim.result import XResult


def lineplot(result):
    source = result.mean

    line = alt.Chart(source).mark_line().encode(x="time", y="mean(Miles_per_Gallon)")

    # band = alt.Chart(source).mark_errorband(extent='ci').encode(
    #    x='Year',
    #    y=alt.Y('Miles_per_Gallon', title='Miles/Gallon'),
    # )

    # band + line
    line.display()


lineplot(None)


def timecourse_plot():
    """Plot lines with confidence or other intervals.
    - add experimental data

    interactive plots
    """

    pass


def scan_plot():
    # plot the results of a parameter scan

    pass


def sensitivity_plot():
    # plot results of sensitivity analysis
    pass


def dist_plot():
    # plot the results of simulated parameter distributions
    # - plot the distributions & correlations between parameters (histograms and scatters)
    pass


def add_line(ax, data, yid, xid="time", color="black", label="", xf=1.0, **kwargs):
    """
    :param ax:
    :param xid:
    :param yid:

    :param color:
    :return:
    """
    kwargs_plot = dict(kwargs_sim)
    kwargs_plot.update(kwargs)

    if isinstance(data, XResult):
        x = data.mean[xid] * xf

        # FIXME: std areas should be within min/max areas!
        ax.fill_between(
            x,
            data.min[yid],
            data.mean[yid] - data.std[yid],
            color=color,
            alpha=0.3,
            label="__nolabel__",
        )
        ax.fill_between(
            x,
            data.mean[yid] + data.std[yid],
            data.max[yid],
            color=color,
            alpha=0.3,
            label="__nolabel__",
        )
        ax.fill_between(
            x,
            data.mean[yid] - data.std[yid],
            data.mean[yid] + data.std[yid],
            color=color,
            alpha=0.5,
            label="__nolabel__",
        )

        ax.plot(x, data.mean[yid], "-", color=color, label="sim {}".format(label))
    else:
        x = data[xid] * xf
        ax.plot(x, data[yid], "-", color=color, label="sim {}".format(label))


if __name__ == "__main__":
    from sbmlsim.models.model import load_model
    from sbmlsim.parametrization import ChangeSet
    from sbmlsim.simulator.simulation_serial import (
        Timecourse,
        TimecourseSimulation,
        timecourses,
    )
    from sbmlsim.test import MODEL_REPRESSILATOR

    r = load_model(MODEL_REPRESSILATOR)

    # parameter sensitivity
    changeset = ChangeSet.parameter_sensitivity_changeset(r, sensitivity=0.05)
    tc_sims = TimecourseSimulation(
        [
            Timecourse(start=0, end=100, steps=100),
            Timecourse(
                start=0,
                end=200,
                steps=100,
                model_changes={"boundary_condition": {"X": True}},
            ),
            Timecourse(
                start=0,
                end=100,
                steps=100,
                model_changes={"boundary_condition": {"X": False}},
            ),
        ]
    ).ensemble(changeset=changeset)

    result = timecourses(r, tc_sims)
