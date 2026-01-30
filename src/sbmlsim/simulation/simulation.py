"""Abstract base simulation."""

import abc
from abc import ABC

from pymetadata import log

from sbmlsim.simulation.algorithm import Algorithm
from sbmlsim.simulation.base import BaseObject
from sbmlsim.simulation.range import Dimension
from sbmlsim.units import UnitsInformation


logger = log.get_logger(__name__)


class Simulation(BaseObject):
    """Simulation class.

    A simulation is the execution of some defined algorithm(s). Simulations are
    described differently depending on the type of simulation experiment to be
    performed.

    Simulation is an abstract class and serves as parent class for the different
    types of simulations.
    """

    def __init__(self, sid: str, algorithm: Algorithm, name: str = None):
        """Construct Simulation.

        The mandatory attribute algorithm defines the simulation algorithms used
        for the execution of the simulation. The algorithms are defined
        via the Algorithm class.
        """
        super(Simulation, self).__init__(sid=sid, name=name)
        self.algorithm: Algorithm = algorithm

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Simulation({self.sid}, {self.name}, {self.algorithm}"


class Analysis(Simulation):
    """Analysis class.

    The Analysis represents any sort of analysis or simulation of a Model, entirely defined by its child
    Algorithm.
    """

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Analysis({self.sid}, {self.name}, {self.algorithm}"


class SteadyState(Simulation):
    """SteadyState class.

    The SteadyState represents a steady state computation (as for example
    implemented by NLEQ or Kinsolve).
    """

    def __repr__(self) -> str:
        """Get string representation."""
        return f"SteadyState({self.sid}, {self.name}, {self.algorithm}"


class OneStep(Simulation):
    """OneStep class.

    The OneStep class calculates one further output step for the model from its
    current state.
    """

    def __repr__(self) -> str:
        """Get string representation."""
        return f"OneStep({self.sid}, {self.name}, {self.algorithm}"

    def __init__(self, sid: str, step: float, algorithm: Algorithm, name: str = None):
        """Construct OneStep."""
        super(OneStep, self).__init__(sid=sid, name=name, algorithm=algorithm)
        self.step: float = step


class UniformTimeCourse(Simulation):
    """UniformTimeCourse class.

    The UniformTimeCourse class calculates a time course output with equidistant
    time points.
    """

    def __repr__(self) -> str:
        """Get string representation."""
        return f"UniformTimeCourse({self.sid}, {self.name}, {self.algorithm}"

    def __init__(
        self,
        sid: str,
        algorithm: Algorithm,
        start: float,
        end: float,
        steps: int,
        initial_time: float,
        name: str = None,
    ):
        """Construct UniformTimeCourse."""
        super(UniformTimeCourse, self).__init__(sid=sid, name=name, algorithm=algorithm)
        self.start: float = start
        self.end: float = end
        self.steps: int = steps
        self.initial_time: float = initial_time


class AbstractSim(ABC):
    """AbstractSim.

    Base class of simulations.
    """

    @abc.abstractmethod
    def dimensions(self) -> list[Dimension]:
        """Get dimension of the simulation."""
        raise NotImplementedError

    @abc.abstractmethod
    def normalize(self, uinfo: UnitsInformation) -> None:
        """Normalize simulation."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_model_changes(self, changes: dict) -> None:
        """Add model changes to model."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        d = {
            "type": self.__class__.__name__,
        }
        return d
