"""DataGenerator."""

from sbmlsim.data import DataSet
from sbmlsim.result import XResult


class DataGeneratorFunction:
    """DataGeneratorFunction."""

    def __call__(
        self, xresults: dict[str, XResult], dsets: dict[str, DataSet] = None
    ) -> dict[str, XResult]:
        """Call the function."""
        raise NotImplementedError


class DataGeneratorIndexingFunction(DataGeneratorFunction):
    """DataGeneratorIndexingFunction."""

    def __init__(self, index: int, dimension: str = "_time"):
        self.index = index
        self.dimension = dimension

    def __call__(self, xresults: dict[str, XResult], dsets=None) -> dict[str, XResult]:
        """Reduce based on '_time' dimension with given index."""
        results = {}
        for key, xres in xresults.items():
            xds_new = xres.xds.isel({self.dimension: self.index})
            xres_new = XResult(xdataset=xds_new, uinfo=xres.uinfo)
            results[key] = xres_new

        return results


class DataGenerator:
    """DataGenerator.

    DataGenerators allow to postprocess existing data. This can be a variety of
    operations.

    - Slicing: reduce the dimension of a given XResult, by slicing a subset on a
      given dimension
    - Cumulative processing: mean, sd, ...
    - Complex processing, such as pharmacokinetics calculation.
    """

    def __init__(
        self,
        f: DataGeneratorFunction,
        xresults: dict[str, XResult],
        dsets: dict[str, DataSet] = None,
    ):
        self.xresults = xresults
        self.dsets = dsets
        self.f = f

    def process(self) -> XResult:
        """Process the data generator."""
        return self.f(xresults=self.xresults, dsets=self.dsets)
