"""The SIds of the keys of an experiment, made from texts such as the labels of the data.

The keys of the models, datasets, simulations, tasks, data, fit mappings and
figures of a `SimulationExperiment` are SIds, which the experiment checks when
it initializes. `to_sid` gives the SId of a text, e.g. of a label of the data
or of a dose, so a key can be built from it: `f"task_po_{to_sid(dose)}"`.
"""

import re

__all__ = ["to_sid"]

#: a decimal point between two digits, which `to_sid` writes as `p`
_DECIMAL_POINT = re.compile(r"(?<=[0-9])\.(?=[0-9])")

#: a character which an SId does not allow
_NOT_IN_SID = re.compile(r"[^A-Za-z0-9_]")


def to_sid(text: str) -> str:
    """Get the SId of a text, e.g. of a label of the data, as a key of an experiment.

    A decimal point between two digits becomes `p` (`0.5mg` is `0p5mg`) and
    every other character which an SId does not allow becomes `_`
    (`po 2.5 mg` is `po_2p5_mg`, `µg/ml` is `_g_ml`). One character becomes
    one character, and an SId stays as it is. The function maps characters, so
    a part of a key gives the same part of the SId of the key
    (`f"task_po_{to_sid(dose)}"`). Whether a whole key is a valid SId, i.e. not
    empty and not starting with a digit, is not checked here (the text of a
    part of a key may start with a digit): the experiment checks every key when
    it initializes and names all invalid ones.

    Args:
        text: the text.

    Returns:
        The SId.
    """
    return _NOT_IN_SID.sub("_", _DECIMAL_POINT.sub("p", text))
