"""The comparisons of the maths of an SBML model, which an engine shifts at their threshold.

The integrators of AMICI (CVODES) and of COPASI (LSODA) find the root of a
comparison where its root function changes its sign, and not where it leaves
0: a comparison which is exactly at its threshold when the integrator starts
to watch it is not found when it turns later. The models of both engines
therefore compile a comparison shifted by a parameter, `a > b + shift`, which
is 0 but at the start of an integration at which the comparison is exactly at
its threshold, see `shift_comparison`.
"""

from __future__ import annotations

from collections.abc import Sequence

import libsbml

#: the strict comparisons of a trigger, see `trigger_comparisons`
STRICT_COMPARISONS = (libsbml.AST_RELATIONAL_GT, libsbml.AST_RELATIONAL_LT)

#: every comparison of a trigger, see `trigger_comparisons`
COMPARISONS = (
    *STRICT_COMPARISONS,
    libsbml.AST_RELATIONAL_GEQ,
    libsbml.AST_RELATIONAL_LEQ,
)

#: a comparison found in a math: its parent (`None` for the math itself), its
#: position in the parent and the node
_Found = tuple["libsbml.ASTNode | None", int, "libsbml.ASTNode"]


def trigger_comparisons(
    trigger: libsbml.ASTNode, kinds: Sequence[int] = STRICT_COMPARISONS
) -> list[_Found]:
    """Get the comparisons of a trigger of given kinds, through its logical operators.

    Args:
        trigger: the math of the trigger.
        kinds: the types of the nodes of the comparisons, the strict ones
            `>` and `<` by default (`STRICT_COMPARISONS`), or every
            comparison (`COMPARISONS`).

    Returns:
        The comparisons in the order of the math, each with its parent
        (`None` for the trigger itself) and its position in it.
    """
    return _comparisons(trigger, kinds, logical=True)


def math_comparisons(math: libsbml.ASTNode, kinds: Sequence[int]) -> list[_Found]:
    """Get the comparisons of given kinds anywhere in a math, e.g. in the conditions of its piecewises.

    Args:
        math: the math.
        kinds: the types of the nodes of the comparisons.

    Returns:
        The comparisons in the order of the math, a comparison before the
        ones in its operands, each with its parent (`None` for the math
        itself) and its position in it.
    """
    return _comparisons(math, kinds, logical=False)


def _comparisons(
    math: libsbml.ASTNode, kinds: Sequence[int], logical: bool
) -> list[_Found]:
    """Get the comparisons of given kinds of a math, see `trigger_comparisons` and `math_comparisons`.

    Args:
        math: the math.
        kinds: the types of the nodes of the comparisons.
        logical: whether to look through the logical operators only, else
            through every node, the operands of a comparison included.
    """
    found: list[_Found] = []
    stack: list[_Found] = [(None, 0, math)]
    while stack:
        parent, index, node = stack.pop()
        comparison = node.getType() in kinds and node.getNumChildren() == 2
        if comparison:
            found.append((parent, index, node))
        if not logical or (not comparison and node.isLogical()):
            stack.extend(
                (node, k, node.getChild(k))
                for k in reversed(range(node.getNumChildren()))
            )
    return found


def shift_comparison(comparison: libsbml.ASTNode, pid: str) -> None:
    """Shift a comparison by a parameter so that it keeps its value at its threshold.

    `a > b` becomes `a > b + shift`, `a < b` `a + shift < b`, `a >= b`
    `a >= b - shift` and `a <= b` `a <= b + shift`: a positive shift keeps
    the value of the comparison at its threshold until the operands move by
    the shift in the direction which changes it, a shift of 0 is the
    comparison as written.

    Args:
        comparison: the node of the comparison, changed in place.
        pid: the id of the parameter of the shift.
    """
    kind = comparison.getType()
    index = 0 if kind == libsbml.AST_RELATIONAL_LT else 1
    operation = (
        libsbml.AST_MINUS if kind == libsbml.AST_RELATIONAL_GEQ else libsbml.AST_PLUS
    )
    shifted = libsbml.ASTNode(operation)
    shifted.addChild(comparison.getChild(index).deepCopy())
    name = libsbml.ASTNode(libsbml.AST_NAME)
    name.setName(pid)
    shifted.addChild(name)
    comparison.replaceChild(index, shifted, True)
