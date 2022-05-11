"""BaseObjects for SED-ML and simulation."""
from abc import ABC
from typing import Optional


class BaseObject(ABC):
    """Base class for SED-ML bases.

    FIXME: support annotations and notes
    """

    def __init__(self, sid: Optional[str], name: Optional[str]):
        """Initialize BaseObject."""
        self.sid: Optional[str] = sid
        self.name: Optional[str] = name


class BaseObjectSIdRequired(BaseObject):
    """Base class for SED-ML bases with required sid."""

    def __init__(self, sid: str, name: Optional[str]):
        """Initialize BaseObjectSIdRequired."""
        super(BaseObjectSIdRequired, self).__init__(sid=sid, name=name)


class Target:
    """Target class.

    An instance of Variable can refer to a model constituent inside a particular model through the address
    stored in the target attribute, such as an XPath expression.
    Note that while it is possible to write XPath expressions that select multiple nodes within a referenced
    model, when used within a target attribute, a single element or attribute must be selected by the
    expression.
    The target attribute may also be used in three situations to reference another SED-ML element with
    mathematical meaning, by containing a fragment identifier consisting of a hash character (#) followed
    by the SId of the element (i.e. “#id001”).
    """

    def __init__(self, target: str):
        """Initialize Target."""
        self.target = target


class Symbol:
    """Symbol class.

    The symbol attribute of type string is used to refer either to a predefined, implicit variable or to a
    predefined implicit function to be performed on the target. In both cases, the symbol should be a kisaoID (and
    follow the format of that attribute) that represents that variable’s concept. The notion of implicit
    variables is explained in Section 3.2.5. For backwards compatibility, the old string “urn:sedml:symbol:time”
    is also allowed, though interpreters should interpret “KISAO:0000832” as meaning the same thing.
    """

    values = [
        # "urn:sedml:symbol:time",
        "KISAO:0000832",  # time
        "KISAO:0000836",  # amount
        "KISAO:0000837",  # particle number
        "KISAO:0000838",  # concentration
        "KISAO:0000654",  # amount rate
        "KISAO:0000652",  # concentration rate
        "KISAO:0000653",  # particle number rate
        # Concentration control coefficient matrix (scaled)
        # FIXME: add additional terms in
        # https://bioportal.bioontology.org/ontologies/KISAO/?p=classes&conceptid=http%3A%2F%2Fwww.biomodels.net%2Fkisao%2FKISAO%23_KISAO_0000800
    ]

    def __init__(self, symbol: str):
        """Initialize Symbol."""

        if symbol == "urn:sedml:symbol:time":
            symbol = "KISAO:0000832"

        if symbol not in Symbol.values:
            raise ValueError(f"Unknown symbol encountered: {symbol}")

        self.symbol = symbol
