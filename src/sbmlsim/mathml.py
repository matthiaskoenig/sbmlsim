"""Evaluation of formulas and MathML expressions.

A `Data` of type `FUNCTION` is a formula of other data, see `sbmlsim.data`.
The formula is parsed into the abstract syntax tree of libsedml, which
implements the L3 formula syntax of SBML, and evaluated on the arrays of the
variables with sympy.

`formula_expression`, `formula_symbols` and `evaluate_formula` read a formula
with libsbml and `sbmlmath`, which make a symbol of every identifier.
`expr_from_formula` hands the text of a formula to `sympify`, which reads
`beta`, `gamma`, `lambda`, `S` or `I` as the functions and constants of sympy
and not as identifiers of a model. `expression_to_astnode` and
`expression_to_formula` are the way back, from a sympy expression to the math
of SBML.
"""

import functools
import logging
from collections.abc import Mapping
from typing import Any

import libsbml
import libsedml
import numpy as np
import sympy
from sbmlmath import SBMLMathMLParser, SBMLMathMLPrinter, TimeSymbol
from sbmlmath.csymbol import SBML_L3V2_AVOGADRO_VALUE, CSymbol
from sympy import Function, lambdify, sympify

logger = logging.getLogger(__name__)

#: the name of the time of the model in the variables of a formula
TIME = "time"


def formula_to_astnode(formula: str) -> libsedml.ASTNode:
    """Parse ASTNode from formula."""
    astnode = libsedml.parseL3Formula(formula)
    if not astnode:
        logger.error("Formula could not be parsed: '%s'", formula)
        logger.error(libsedml.getLastParseL3Error())
    return astnode


def astnode_to_formula(astnode: libsedml.ASTNode) -> str:
    """Write ASTNode as formula."""
    return libsedml.formulaToL3String(astnode)


def parse_mathml_str(mathml_str: str):
    """Parse MathML string."""
    astnode: libsedml.ASTNode = libsedml.readMathMLFromString(mathml_str)
    return parse_astnode(astnode)


def parse_formula(formula: str) -> libsedml.ASTNode:
    """Parse formula to ASTNode."""
    astnode = formula_to_astnode(formula)
    return parse_astnode(astnode)


def parse_astnode(astnode: libsedml.ASTNode) -> Any:
    """Parse ASTNode.

    An AST node in libSBML is a recursive tree structure; each node has a type,
    a pointer to a value, and a list of children nodes. Each ASTNode node may
    have none, one, two, or more children depending on its type. There are
    node types to represent numbers (with subtypes to distinguish integer,
    real, and rational numbers), names (e.g., constants or variables),
    simple mathematical operators, logical or relational operators and
    functions.

    see also: http://sbml.org/Software/libSBML/docs/python-api/libsedml-math.html

    :param mathml:
    :return:
    """
    formula = libsedml.formulaToL3String(astnode)

    # iterate over ASTNode and figure out variables
    # variables = _get_variables(astnode)

    # create sympy expression
    return expr_from_formula(formula)

    # print(formula, expr)


def expr_from_formula(formula: str):
    """Parse sympy expression from given formula string."""
    # [2] create sympy expressions with variables and formula
    # necessary to map the expression trees
    # create symbols
    formula = replace_piecewise(formula)
    formula = formula.replace("&&", "&")
    formula = formula.replace("||", "|")

    # additional methods
    # ns = {}
    # symbols = []
    # exec_('from sbmlsim.processing.mathml_functions import piecewise', ns)
    # from sympy import Symbol
    # for variable in sorted(variables):
    #    symbol = Symbol(variable)
    #    ns[variable] = symbol
    #    symbols.append(symbol)
    # expr = sympify(formula, locals=ns)
    return sympify(formula)


class _ArrayMax(Function):
    """`max` of a function of data: of one array its largest value."""


class _ArrayMin(Function):
    """`min` of a function of data: of one array its smallest value."""


def _array_max(*args: Any) -> Any:
    """Reduce one array to its maximum, take the maximum of several values."""
    if len(args) == 1:
        return np.nanmax(args[0])
    return np.maximum.reduce(np.broadcast_arrays(*args))


def _array_min(*args: Any) -> Any:
    """Reduce one array to its minimum, take the minimum of several values."""
    if len(args) == 1:
        return np.nanmin(args[0])
    return np.minimum.reduce(np.broadcast_arrays(*args))


def evaluate(astnode: libsedml.ASTNode, variables: dict):
    """Evaluate the astnode with values.

    The values are arrays of data, e.g. a timecourse, so `max` and `min` of a
    single argument reduce it, `Y/max(Y)` is `Y` normalized to its maximum;
    sympy would simplify `Max(Y)` to `Y`.
    """
    formula = replace_piecewise(libsedml.formulaToL3String(astnode))
    formula = formula.replace("&&", "&").replace("||", "|")
    functions: dict[str, Any] = {"max": _ArrayMax, "min": _ArrayMin}
    # the overloads of sympy do not declare `locals`
    expr = sympify(formula, locals=functions)  # ty: ignore[no-matching-overload]
    symbols = sorted(expr.free_symbols, key=str)
    f = lambdify(
        args=symbols,
        expr=expr,
        modules=[{"_ArrayMax": _array_max, "_ArrayMin": _array_min}, "numpy"],
    )
    # only the variables of the expression are passed
    return f(*[variables[str(symbol)] for symbol in symbols])


def _get_variables(
    astnode: libsedml.ASTNode, variables: set[str] | None = None
) -> set[str]:
    """Add variable names to the variables."""
    if variables is None:
        variables = set()

    num_children = astnode.getNumChildren()
    if num_children == 0:
        if astnode.isName():
            name = astnode.getName()
            variables.add(name)
    else:
        for k in range(num_children):
            child = astnode.getChild(k)  # type: libsedml.ASTNode
            _get_variables(child, variables=variables)

    return variables


def replace_piecewise(formula):
    """Replace libsedml piecewise with sympy piecewise."""
    while True:
        index = formula.find("piecewise(")
        if index == -1:
            break

        # process piecewise
        search_idx = index + 9

        # init counters
        bracket_open = 0
        pieces = []
        piece_chars = []

        while search_idx < len(formula):
            c = formula[search_idx]
            if c == ",":
                if bracket_open == 1:
                    pieces.append("".join(piece_chars).strip())
                    piece_chars = []
            else:
                if c == "(":
                    if bracket_open != 0:
                        piece_chars.append(c)
                    bracket_open += 1
                elif c == ")":
                    if bracket_open != 1:
                        piece_chars.append(c)
                    bracket_open -= 1
                else:
                    piece_chars.append(c)

            if bracket_open == 0:
                pieces.append("".join(piece_chars).strip())
                break

            # next character
            search_idx += 1

        # find end index
        if (len(pieces) % 2) == 1:
            pieces.append("True")  # last condition is True
        sympy_pieces = []
        for k in range(int(len(pieces) / 2)):
            sympy_pieces.append(f"({pieces[2 * k]}, {pieces[2 * k + 1]})")
        new_str = f"Piecewise({','.join(sympy_pieces)})"
        formula = formula.replace(formula[index : search_idx + 1], new_str)

    return formula


_MATHML_NAMESPACE = "http://www.w3.org/1998/Math/MathML"


def _truncate(quotient: sympy.Expr) -> sympy.Expr:
    """Round a quotient towards zero, as `quotient` and `rem` of SBML do.

    Args:
        quotient: the expression to round.

    Returns:
        The expression, a piecewise of `floor` and `ceiling`, which the MathML
        of SBML has.
    """
    return sympy.Piecewise(
        (sympy.floor(quotient), quotient >= 0), (sympy.ceiling(quotient), True)
    )


class _FormulaParser(SBMLMathMLParser):
    """The MathML parser of `sbmlmath` with the `rem` and `quotient` of SBML.

    `sbmlmath` reads both with the modulo of sympy, which rounds towards
    negative infinity. The `rem` of SBML is the remainder of the truncated
    division, `rem(-7, 3)` is -1, and the `quotient` is the truncated
    quotient, `quotient(-7, 3)` is -2.
    """

    def handle_apply(self, element: Any) -> sympy.Expr:
        """Parse an `apply`, `rem` and `quotient` truncate.

        Args:
            element: the `apply` element.

        Returns:
            The expression.
        """
        operator, *operands = element
        tag = str(operator.tag).removeprefix(f"{{{_MATHML_NAMESPACE}}}")
        if tag in ("rem", "quotient") and len(operands) == 2:
            x, y = (self._parse_element(operand) for operand in operands)
            if y.is_zero:
                # not defined; sympy would simplify `x - 0 * zoo` to `x`
                return sympy.nan
            truncated = _truncate(x / y)
            return truncated if tag == "quotient" else x - y * truncated
        return super().handle_apply(element)


def _log_quotient(expression: sympy.Basic) -> sympy.Basic:
    """Write the logarithms with a base as quotients of natural logarithms.

    `sbmlmath` reads `log(x)`, `log10(x)` and `log(b, x)` as `log(x, b)`,
    which neither the printer of MathML nor `lambdify` at full precision
    handle.

    Args:
        expression: the expression.

    Returns:
        The expression, in which `log(x, b)` is `log(x) / log(b)`.
    """
    return expression.replace(
        lambda node: isinstance(node, sympy.log) and len(node.args) == 2,
        lambda node: sympy.log(node.args[0]) / sympy.log(node.args[1]),
    )


def formula_expression(formula: str) -> sympy.Basic:
    """Parse an L3 formula of SBML into a sympy expression.

    Every identifier of the formula is a symbol of its name, also the
    identifiers which are functions or constants of sympy (`beta`, `gamma`,
    `lambda`, `S`, `I`). The time of the model is the symbol `time`. The
    constant `avogadro` stays the `csymbol` of SBML, which is not an
    identifier: `formula_symbols` does not list it and `evaluate_formula`
    gives it its value. A logarithm with a base is a quotient of natural
    logarithms, `rem` and `quotient` truncate as they do in SBML.

    Args:
        formula: the formula, e.g. `prey + (alpha - 1.3)`.

    Returns:
        The expression.

    Raises:
        ValueError: if the formula is empty, is not valid math (units of a
            number are not) or uses a function which is not a function of
            the MathML of SBML.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(f"The formula '{formula}' is empty")
    settings = libsbml.L3ParserSettings()
    settings.setParseUnits(False)
    astnode: libsbml.ASTNode | None = libsbml.parseL3FormulaWithSettings(
        formula, settings
    )
    if astnode is None:
        raise ValueError(
            f"The formula '{formula}' is not valid math: "
            f"{libsbml.getLastParseL3Error()}"
        )
    try:
        # through the text of the MathML: libsbml and libsedml both wrap the
        # syntax tree, and the class of a node is the one of the library which
        # was imported last
        mathml = libsbml.writeMathMLToString(astnode)
        expression = sympy.sympify(_FormulaParser(ignore_units=True).parse_str(mathml))
    except Exception as err:
        raise ValueError(
            f"The formula '{formula}' cannot be evaluated: {type(err).__name__}: {err}"
        ) from err
    functions = sorted(
        {
            str(function.func)
            for function in expression.atoms(sympy.Function)
            if isinstance(function, sympy.core.function.AppliedUndef)
        }
    )
    if functions:
        raise ValueError(
            f"The formula '{formula}' uses the functions {functions}, which "
            f"have no value: they are not functions of the MathML of SBML, or "
            f"functions of a simulation"
        )
    expression = _log_quotient(expression)
    return expression.subs(
        {
            symbol: sympy.Symbol(TIME)
            for symbol in expression.free_symbols
            if isinstance(symbol, TimeSymbol)
        }
    )


def _identifiers(expression: sympy.Basic) -> list[sympy.Basic]:
    """Get the identifiers of an expression, sorted by their names.

    Args:
        expression: the expression.

    Returns:
        The symbols of the expression which are not constants of SBML.
    """
    return sorted(
        (
            symbol
            for symbol in expression.free_symbols
            if not isinstance(symbol, CSymbol)
        ),
        key=str,
    )


def formula_symbols(formula: str) -> set[str]:
    """Get the identifiers a formula uses.

    Args:
        formula: the formula, an L3 formula of SBML.

    Returns:
        The names of the symbols of the formula, `time` for the time of the
        model. Constants like `pi` or `avogadro` are no identifiers.

    Raises:
        ValueError: if the formula is not valid math, see
            `formula_expression`.
    """
    return {str(symbol) for symbol in _identifiers(formula_expression(formula))}


def _log10(expression: sympy.Basic) -> sympy.Basic:
    """Write a quotient with the logarithm of 10 as `log10`.

    `log(1000) / log(10)` is 2.9999999999999996, `log10(1000)` is 3.

    Args:
        expression: the expression, in which a base is a quotient of
            logarithms, see `formula_expression`.

    Returns:
        The expression, in which `log(x) / log(10)` is `log10(x)`.
    """
    base = 1 / sympy.log(10)
    log10: Any = sympy.Function("log10")

    def replace(product: sympy.Mul) -> sympy.Basic:
        logarithm = next(arg for arg in product.args if isinstance(arg, sympy.log))
        rest = [arg for arg in product.args if arg not in (logarithm, base)]
        return sympy.Mul(*rest, log10(logarithm.args[0]))

    return expression.replace(
        lambda node: (
            isinstance(node, sympy.Mul)
            and base in node.args
            and any(isinstance(arg, sympy.log) for arg in node.args)
        ),
        replace,
    )


@functools.lru_cache(maxsize=1024)
def _formula_function(formula: str) -> tuple[tuple[str, ...], Any]:
    """Get the function of a formula and the names of its arguments.

    The function is built once per formula: a fit evaluates the formulas of
    its derived changes in every simulation.

    Args:
        formula: the formula.

    Returns:
        The names of the symbols of the formula and the function of their
        values, in this order.

    Raises:
        ValueError: if the formula is not valid math, see
            `formula_expression`.
    """
    expression = formula_expression(formula)
    avogadro = {
        symbol: sympy.Float(SBML_L3V2_AVOGADRO_VALUE)
        for symbol in expression.free_symbols
        if isinstance(symbol, CSymbol)
    }
    symbols = _identifiers(expression)
    # the symbols are arguments by position: an identifier of a model is not
    # always a name of python, e.g. `lambda`
    arguments = [sympy.Dummy() for _ in symbols]
    function = lambdify(
        args=arguments,
        expr=_log10(
            expression.xreplace(avogadro).xreplace(
                dict(zip(symbols, arguments, strict=True))
            )
        ),
        modules=[{"log10": np.log10}, "numpy"],
    )
    return tuple(str(symbol) for symbol in symbols), function


def evaluate_formula(formula: str, variables: Mapping[str, Any]) -> Any:
    """Evaluate an L3 formula of SBML on the values of its identifiers.

    Args:
        formula: the formula.
        variables: the value of every identifier of the formula, a number or
            an array. The values of identifiers the formula does not use are
            not used, but give the shape of the result.

    Returns:
        The value of the formula: a float if all values are numbers, else an
        array of the shape the values broadcast to, with `nan` where the
        formula is not defined.

    Raises:
        ValueError: if the formula is not valid math, see
            `formula_expression`, if an identifier has no value, if the
            shapes of the values do not broadcast, or if the value of numbers
            is complex, overflows or divides by zero.
    """
    if not isinstance(formula, str):
        raise ValueError(f"The formula '{formula}' is empty")
    symbols, function = _formula_function(formula)
    missing = [symbol for symbol in symbols if symbol not in variables]
    if missing:
        raise ValueError(
            f"The formula '{formula}' uses {missing}, which have no value. The "
            f"values are given for {sorted(variables)}"
        )
    try:
        shape = np.broadcast_shapes(*(np.shape(value) for value in variables.values()))
    except ValueError as err:
        shapes = {name: np.shape(value) for name, value in variables.items()}
        raise ValueError(
            f"The values of the formula '{formula}' cannot be broadcast to "
            f"one shape: {shapes}"
        ) from err
    try:
        value = np.asarray(function(*[variables[symbol] for symbol in symbols]), float)
    except (ArithmeticError, TypeError) as err:
        # a complex value, an overflow or a division by zero of numbers
        raise ValueError(f"The formula '{formula}' cannot be evaluated: {err}") from err
    if shape == () and value.shape == ():
        return float(value)
    return np.array(np.broadcast_to(value, np.broadcast_shapes(shape, value.shape)))


class _ExpressionPrinter(SBMLMathMLPrinter):
    """The MathML printer of `sbmlmath` for the math of SBML."""

    def _element(self, expr: sympy.Basic) -> Any:
        """Print an expression as a DOM element.

        Args:
            expr: the expression.

        Returns:
            The element.
        """
        return self._print(expr)

    def _print_Piecewise(self, expr: sympy.Piecewise) -> Any:
        """Print a piecewise, with an `otherwise` only if it has a default.

        The printer of sympy insists on a default, while the MathML of SBML
        leaves the value undefined outside of the pieces without one.

        Args:
            expr: the piecewise.

        Returns:
            The `piecewise` element.
        """
        root = self.dom.createElement("piecewise")
        pieces: Any = expr.args
        for value, condition in pieces:
            if condition == True:  # noqa: E712 - sympy's `true`
                child = self.dom.createElement("otherwise")
                child.appendChild(self._element(value))
            else:
                child = self.dom.createElement("piece")
                child.appendChild(self._element(value))
                child.appendChild(self._element(condition))
            root.appendChild(child)
        return root


def expression_to_astnode(expression: sympy.Basic) -> libsbml.ASTNode:
    """Convert a sympy expression into the math of SBML.

    The numbers of the expression carry no units, so the math is valid in
    every level of SBML. The symbol `time` is the time of the model.

    Args:
        expression: the expression.

    Returns:
        The syntax tree of the MathML of the expression.

    Raises:
        ValueError: if the expression uses a function the MathML of SBML does
            not have, e.g. the error function.
    """
    functions = sorted({str(f.func) for f in expression.atoms(sympy.Function)})
    try:
        time = TimeSymbol(TIME)
        mathml = _ExpressionPrinter(literals_dimensionless=False).doprint(
            expression.xreplace(
                {
                    symbol: time
                    for symbol in expression.free_symbols
                    if isinstance(symbol, sympy.Symbol)
                    and not isinstance(symbol, CSymbol)
                    and symbol.name == TIME
                }
            )
        )
        astnode: libsbml.ASTNode | None = libsbml.readMathMLFromString(mathml)
    except Exception as err:
        raise ValueError(
            f"The expression '{expression}' has no MathML of SBML, its "
            f"functions are {functions}: {type(err).__name__}: {err}"
        ) from err
    if astnode is None or not astnode.isWellFormedASTNode():
        raise ValueError(
            f"The expression '{expression}' has no MathML of SBML, its "
            f"functions are {functions}"
        )
    return astnode


def expression_to_formula(expression: sympy.Basic) -> str:
    """Write a sympy expression as an L3 formula of SBML.

    Args:
        expression: the expression.

    Returns:
        The formula, which `formula_expression` and libsbml read. The
        natural logarithm is `ln`: `log` is the logarithm to the base 10 in a
        formula of SBML.

    Raises:
        ValueError: if the expression has no MathML of SBML, see
            `expression_to_astnode`.
    """
    settings = libsbml.L3ParserSettings()
    settings.setParseUnits(False)
    return str(
        libsbml.formulaToL3StringWithSettings(
            expression_to_astnode(expression), settings
        )
    )


if __name__ == "__main__":
    # Piecewise in sympy
    # https://docs.sympy.org/latest/modules/functions/elementary.html#piecewise
    # Piecewise((expr, cond), (expr, cond), … )
    # necessary to do a rewrite of the piecewise function
    expr = expr_from_formula("piecewise(8, x < 4, 0.1, (4 <= x) && (x < 6), 8)")
    expr = expr_from_formula(
        "Piecewise((8, x < 4), (0.1, (x >= 5) & (x < 6)), (8, True))"
    )

    print(expr)

    # evaluate expression
    expr = parse_formula("x + y")
    print(expr, type(expr))

    """
    # evaluate the function with the values
    astnode = libsedml.readMathMLFromString(mathmlStr)

    y = 5
    res = evaluateMathML(astnode,
                         variables={'x': y})
    print('Result:', res)
    """

    """
    * The Boolean function symbols '&&' (and), '||' (or), '!' (not),
    and '!=' (not equals) may be used.
    """
