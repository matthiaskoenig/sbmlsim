"""DataGenerator."""

from sbmlsim.data import DataSet
from sbmlsim.result import XResult


class DataGeneratorFunction:
    """DataGeneratorFunction."""

    def __call__(
        self, xresults: dict[str, XResult], dsets: dict[str, DataSet] | None = None
    ) -> dict[str, XResult]:
        """Call the function.

        Args:
            xresults: Results to process.
            dsets: Datasets to process.

        Returns:
            Processed results.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class DataGeneratorIndexingFunction(DataGeneratorFunction):
    """DataGeneratorIndexingFunction."""

    def __init__(self, index: int, dimension: str = "_time"):
        """Initialize DataGeneratorIndexingFunction.

        A ragged result, whose simulations have different time points, is
        interpolated onto the union of its time points first, so that an
        index is the same time in every simulation.

        Args:
            index: Index to select on the dimension.
            dimension: Dimension to reduce, the time by default.
        """
        self.index = index
        self.dimension = dimension

    def __call__(
        self, xresults: dict[str, XResult], dsets: dict[str, DataSet] | None = None
    ) -> dict[str, XResult]:
        """Reduce a dimension, by default the time, at the index.

        Args:
            xresults: Results to process.
            dsets: Datasets to process (unused).

        Returns:
            Reduced results.
        """
        results = {}
        for key, xres in xresults.items():
            if self.dimension == "_time" and "_point" in xres.xds.dims:
                xres = xres.interpolate(xres.time_points())
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
        dsets: dict[str, DataSet] | None = None,
    ):
        """Initialize DataGenerator.

        Args:
            f: Function applied to the data.
            xresults: Results to process.
            dsets: Datasets to process.
        """
        self.xresults = xresults
        self.dsets = dsets
        self.f = f

    def process(self) -> dict[str, XResult]:
        """Process the data generator.

        Returns:
            Processed results.
        """
        return self.f(xresults=self.xresults, dsets=self.dsets)
