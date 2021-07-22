"""Module for performing all the Calculations."""


from abc import abstractmethod
from typing import List, Optional

from sbmlsim.simulation.base import BaseObject, BaseObjectSIdRequired


class Parameter(BaseObjectSIdRequired):
    """Parameter class.
    
    The Parameter class (Figure 2.4) is used to create named parameters with a constant value. 
    A Parameter can be used wherever a mathematical expression to compute a value is defined, e.g.,
    in ComputeChange, FunctionalRange or DataGenerator. The Parameter definitions are local to the
    particular class defining them.
    """

    def __init__(
        self, sid: str, value: float, unit: str, name: Optional[str] = None
    ):
        """Construct Parameter."""
        super(Parameter, self).__init__(sid=sid, name=name)
        self.value: float = value
        self.unit: str = unit

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Parameter(sid={self.sid}, name={self.name}, value={self.value}, unit={self.unit})"


class Variable(BaseObjectSIdRequired):

    def __init__(
        self, sid: str, value: float, unit: str, name: Optional[str] = None
    ):
        """Construct Variable."""
        super(Parameter, self).__init__(sid=sid, name=name)
        self.value: float = value
        self.unit: str = unit


class Calculation(BaseObjectSIdRequired):
    """Calculation class.

    Used by ComputeChange, DataGenerator and FunctionalRange.
    """

    def __init__(
        self, sid: str, variables: List[Variable], parameters: List[Parameter], math: str, name: Optional[str] = None
    ):
        """Construct Calculation."""
        super(Calculation, self).__init__(sid=sid, name=name)
        self.variables: List[Variable] = variables
        self.parameters: List[Parameter] = parameters
        self.math: str = math

    @abstractmethod
    def values(self):
        # FIXME
        # evaluate with actual data
            astnode = mathml.formula_to_astnode(self.function)
            variables = {}
            for var_key, variable in self.variables.items():
                # lookup via key
                if isinstance(variable, str):
                    variables[var_key] = experiment._data[variable].data
                elif isinstance(variable, Data):
                    variables[var_key] = variable.get_data(experiment=experiment)
            for par_key, par_value in self.parameters.items():
                variables[par_key] = par_value

            x = mathml.evaluate(astnode=astnode, variables=variables)


class ComputeChange(Calculation):
    """ComputeChange class."""
    
    pass

class DataGenerator(Calculation):
    """DataGenerator class."""

    pass


class FunctionalRange(Calculation):
    """FunctionalRange class."""

    pass


if __name__ == "__main__":
    parameters: List[Parameter] = [
        Parameter(sid="p1", value=10.0, unit="mM"),
    ]

    print(parameters)
