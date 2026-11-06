"""Experiments of the toy project which fail, for the tests of the failures of a run.

They are not in the module of the experiments, so the experiments of the toy
project and the checks of its registry do not see them; `BROKEN` has them.
"""

from typing import override

from matplotlib.figure import Figure as FigureMPL

from sbmlsim.experiment import SimulationExperiment
from sbmlsim.simulation import Simulation
from tests.project_toy.experiments import ToyDecay


class ToyFigureFails(ToyDecay):
    """The decay, whose own figures fail when they are drawn."""

    @override
    def figures_mpl(self) -> dict[str, FigureMPL]:
        """Fail when the figures are drawn."""
        raise RuntimeError("the figures of ToyFigureFails fail")


class ToyCannotInitialize(ToyDecay):
    """The decay, whose simulations cannot be created."""

    @override
    def simulations(self) -> dict[str, Simulation]:
        """Fail when the experiment is initialized."""
        raise RuntimeError("the simulations of ToyCannotInitialize fail")


#: the experiments with the ones which fail
BROKEN: list[type[SimulationExperiment]] = [
    ToyDecay,
    ToyFigureFails,
    ToyCannotInitialize,
]

#: the experiments without `ToyDecaySlow` and `ToyDecayHalf`
PARTIAL: list[type[SimulationExperiment]] = [ToyDecay]

#: no experiments and no fits
NOT_EXPERIMENTS = 42
NO_FITS: dict[str, object] = {}
