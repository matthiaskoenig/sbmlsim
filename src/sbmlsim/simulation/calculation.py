"""Module for performing all the SED-ML calculations."""


from abc import abstractmethod
from typing import List

from sbmlsim.simulation.base import BaseObject


class Calculation(BaseObject):
    """Calculation class.

    Used by ComputeChange, DataGenerator and FunctionalRange.
    """

    def __init__(
        self, sid: str, variables: List, parameters: List, math: str, name: str = None
    ):
        """Construct Calculation."""
        super(Calculation, self).__init__(sid=sid, name=name)
        self.variables: List = variables
        self.parameters: List = parameters
        self.math: str = math

    @abstractmethod
    def values(self):
        pass


class DataGenerator(Calculation):
    pass


class FunctionalRange(Calculation):
    """"""

    pass
