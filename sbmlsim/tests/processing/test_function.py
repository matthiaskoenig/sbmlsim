import numpy as np
from sbmlsim.processing.function import Function


def test_function1():
    f1 = Function(
        index="test", formula="(x + y + z)/x",
        variables={
         'x': 0.1 * np.ones(shape=[1, 10]),
         'y': 3.0 * np.ones(shape=[1, 10]),
         'z': 2.0 * np.ones(shape=[1, 10]),
        })
    res = f1.data()
    print(res)
