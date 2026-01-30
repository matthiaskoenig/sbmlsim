"""Module for performing all the Calculations."""

from typing import Optional

from sbmlsim.simulation.base import BaseObject, BaseObjectSIdRequired, Symbol, Target


class Parameter(BaseObjectSIdRequired):
    """Parameter class.

    The Parameter class (Figure 2.4) is used to create named pars with a constant value.
    A Parameter can be used wherever a mathematical expression to compute a value is defined, e.g.,
    in ComputeChange, FunctionalRange or DataGenerator. The Parameter definitions are local to the
    particular class defining them.
    """

    def __init__(
        self,
        sid: str,
        value: float,
        unit: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Construct Parameter."""
        super(Parameter, self).__init__(sid=sid, name=name)
        self.value: float = value
        self.unit: Optional[str] = unit

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Parameter(sid={self.sid}, name={self.name}, value={self.value}, unit={self.unit})"


class AppliedDimension(BaseObject):
    """AppliedDimension class.

    A AppliedDimension object is used when the term of the Variable is a function that reduces the dimen-
    sionality of the data.

    Dimension reducing functions can be applied in two contexts:
    First to reduce data from RepeatedTasks and nested RepeatedTasks which requires the taskReference
    of the variable to be set and to be a reference to a RepeatedTask.
    All AppliedDimensions must have the target set and reference either one of the
    possibly nested RepeatedTask Sids or the Task within the RepeatedTask.
    Second to reduce data from a multi-dimensional DataSource in a DataGenerator which
    requires the target of the variable to be set to reference the respective DataSource.
    The AppliedDimensions must have the dimensionTarget set to a NuMLIdRef referencing a dimension of the data."
    "If the listOfAppliedDimensions contains 2 or more AppliedDimensions the reducing function is applied on an element-by-element basis."


    """

    def __init__(
        self,
        target: Optional[str] = None,
        dimension_target: Optional[str] = None,
        sid: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Construct Parameter."""
        super(AppliedDimension, self).__init__(sid=sid, name=name)
        self.target: Optional[str] = target
        self.dimension_target: Optional[str] = dimension_target

    def __repr__(self) -> str:
        """Get string representation."""
        return f"AppliedDimension(sid={self.sid}, name={self.name}, target={self.target}, dimension_target={self.dimension_target})"


class Variable(BaseObjectSIdRequired):
    """Variable class.

    A Variable is a reference to an already existing entity, either explicitly created in the
    SED-ML Document, or to an implicitly defined symbol.
    """

    def __init__(
        self,
        sid: str,
        model_reference: Optional[str],
        task_reference: Optional[str],
        target: Optional[Target] = None,
        symbol: Optional[Symbol] = None,
        unit: Optional[str] = None,
        name: Optional[str] = None,
        term: Optional[str] = None,
        applied_dimensions: Optional[list[AppliedDimension]] = None,
    ):
        """Construct Variable."""
        super(Variable, self).__init__(sid=sid, name=name)
        self.model_reference: Optional[str] = model_reference
        self.task_reference: Optional[str] = task_reference
        self.target: Optional[Target] = target
        self.symbol: Optional[Symbol] = symbol
        self.unit: Optional[str] = unit
        self.term: Optional[str] = term
        self.applied_dimensions: Optional[list[AppliedDimension]] = applied_dimensions

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Variable(sid={self.sid}, name={self.name}, target={self.target}, symbol={self.symbol}, term={self.term})"


class DependentVariable(Variable):
    """DependentVariable class.

    A dependent variable
    is necessary when the desired variable is a composite of two other variables, such as ‘the rate of change
    of S1 with respect to time’.
    """

    def __init__(
        self,
        sid: str,
        model_reference: Optional[str],
        task_reference: Optional[str],
        target: Optional[Target] = None,
        symbol: Optional[Symbol] = None,
        target2: Optional[Target] = None,
        symbol2: Optional[Symbol] = None,
        unit: Optional[str] = None,
        name: Optional[str] = None,
        term: Optional[str] = None,
        applied_dimensions: Optional[list[AppliedDimension]] = None,
    ):
        """Construct DependentVariable."""
        super(DependentVariable, self).__init__(
            sid=sid,
            name=name,
            model_reference=model_reference,
            task_reference=task_reference,
            target=target,
            symbol=symbol,
            unit=unit,
            term=term,
            applied_dimensions=applied_dimensions,
        )
        self.target2: Optional[Target] = target2
        self.symbol2: Optional[Symbol] = symbol2


class Calculation(BaseObjectSIdRequired):
    """Calculation class.

    Used by ComputeChange, DataGenerator and FunctionalRange.
    """

    def __init__(
        self,
        sid: str,
        variables: list[Variable],
        parameters: list[Parameter],
        math: str,
        name: Optional[str] = None,
    ):
        """Construct Calculation."""
        super(Calculation, self).__init__(sid=sid, name=name)
        self.variables: list[Variable] = variables
        self.parameters: list[Parameter] = pars
        self.math: str = math

    # @abstractmethod
    def values(self):
        """Access to values."""
        pass
        # FIXME
        # evaluate with actual data
        # astnode = mathml.formula_to_astnode(self.function)
        # variables = {}
        # for var_key, variable in self.variables.items():
        #     # lookup via key
        #     if isinstance(variable, str):
        #         variables[var_key] = experiment._data[variable].data
        #     elif isinstance(variable, Data):
        #         variables[var_key] = variable.get_data(experiment=experiment)
        # for par_key, par_value in self.pars.items():
        #     variables[par_key] = par_value

        # x = mathml.evaluate(astnode=astnode, variables=variables)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"Calculation(sid={self.sid}, name={self.name}, variables={self.variables}, parameters={self.parameters}, math={self.math})"


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
    from pymetadata.console import console

    pars: list[Parameter] = [
        Parameter(sid="p1", value=10.0, unit="mM"),
        Parameter(sid="p2", value=0),
    ]
    console.log(pars)

    dims: list[AppliedDimension] = [
        AppliedDimension(sid="dim1", target="repeated_task1")
    ]
    console.log(dims)

    vars: list[Variable] = [
        Variable(
            sid="S1_model1",
            target="S1",
            model_reference="model1",
            task_reference="repeated_task1",
        ),
        Variable(
            sid="S2_model1",
            target="S2",
            model_reference="model1",
            task_reference="repeated_task1",
            applied_dimensions=dims,
        ),
    ]
    console.log(vars)

    calculation = Calculation(
        sid="calculation1",
        parameters=pars,
        variables=vars,
        math="p1 + p2 + S1_model1 + S2_model1",
    )
    console.log(calculation)
