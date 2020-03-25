from pint import UnitRegistry

from sbmlsim.data import Data
import numpy as np

from sbmlsim.processing import mathml


class Function(object):
    """ Functional data calculation.

    The idea ist to provide an object which can calculate a generic math function
    based on given input symbols.

    Important challenge is to handle the correct functional evaluation.
    """
    def __init__(self, index, formula, variables):
        self.index = index
        self.formula = formula
        self.variables = variables

    def data(self):
        # evalutate with actual data
        astnode = mathml.formula_to_astnode(self.formula)
        res = mathml.evaluate(astnode=astnode, variables=self.variables)
        return res


if __name__ == "__main__":
    f1 = Function(
        index="test", formula="(x + y + z)/x",
        variables={
         'x': 0.1 * np.ones(shape=[1, 10]),
         'y': 3.0 * np.ones(shape=[1, 10]),
         'z': 2.0 * np.ones(shape=[1, 10]),
        })
    res = f1.data()
    print(res)