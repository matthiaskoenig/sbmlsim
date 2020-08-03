from typing import Dict, Callable

from sbmlsim.result import XResult


class DataGenerator:
    """
    DataGenerators allow to postprocess existing data. This can be a variety of operations.

    - Slicing: reduce the dimension of a given XResult, by slicing a subset
    -

    """


    def __init__(self, xresults: Dict[str, XResult], f: Callable):
        self.xresults = xresults
        self.f = f

    def process(self) -> XResult:
        return self.f(self.xresults)