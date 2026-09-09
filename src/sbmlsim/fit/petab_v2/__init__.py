"""PEtab v2 layer of the parameter fitting.

An `sbmlsim` optimization problem is written as a PEtab v2 problem and a PEtab
v2 problem is read into an optimization problem, i.e., a fit of `sbmlsim` is
exchanged with the tools of the PEtab ecosystem and a problem of the PEtab
benchmark collection is fitted with `sbmlsim`.

PEtab does not express everything an `sbmlsim` fit is, e.g. the units of the
data, the settings of the fit or what a fit does with a subset of the data.
What the tables do not hold goes into the `sbmlsim` extension of the problem,
so a problem which is written and read again is the fit it started from, and
`sbmlsim.fit.petab_v2.gaps` reports what an export loses for other tools.
"""

from sbmlsim.fit.petab_v2.export import PetabExporter, to_petab
from sbmlsim.fit.petab_v2.extension import (
    EXTENSION_ID,
    EXTENSION_VERSION,
    SbmlsimExtension,
)
from sbmlsim.fit.petab_v2.gaps import (
    GAPS,
    Gap,
    GapKind,
    gaps_of_problem,
    gaps_table,
)
from sbmlsim.fit.petab_v2.reader import PetabReader, from_petab

__all__ = [
    "EXTENSION_ID",
    "EXTENSION_VERSION",
    "GAPS",
    "Gap",
    "GapKind",
    "PetabExporter",
    "PetabReader",
    "SbmlsimExtension",
    "from_petab",
    "gaps_of_problem",
    "gaps_table",
    "to_petab",
]
