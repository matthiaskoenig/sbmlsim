"""The form of a reference `"module:attribute"`, standard library only."""

from __future__ import annotations

__all__ = ["check_reference"]


def check_reference(reference: str) -> None:
    """Check that a reference has the form `"module:attribute"`.

    The module may start with dots (relative to a package), the attribute may
    be dotted (an attribute of an attribute).

    Args:
        reference: the reference to check.

    Raises:
        ValueError: if the reference does not have the form.
    """
    module, separator, attribute = reference.partition(":")
    names = [*module.lstrip(".").split("."), *attribute.split(".")]
    if not separator or not all(name.isidentifier() for name in names):
        raise ValueError(f"A reference is 'module:attribute', not '{reference}'.")
