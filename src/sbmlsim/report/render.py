"""Renderers of a report document as HTML, Typst and LaTeX.

The HTML has MathML for the formulas (`latex2mathml`) and needs no network, the
Typst source is compiled to PDF with the `typst` package (`compile_pdf`), the
LaTeX is a fragment to paste into a document or a standalone document. Typst
and LaTeX read the references from the files of `write_bibliography` next to
them, `references.bib` and, for Typst, the style `references.csl`.
Tables, figures and equations are numbered in document order per kind and
referenced by their label.
"""

from __future__ import annotations

import html
import unicodedata
from pathlib import Path
from typing import override

import latex2mathml.converter
import typst

from sbmlsim.report.document import (
    Block,
    Cite,
    Code,
    Document,
    Equation,
    Figure,
    Inline,
    Math,
    Paragraph,
    Ref,
    Section,
    Table,
    Text,
)
from sbmlsim.report.references import BIBLIOGRAPHY_FILE, STYLE_FILE

__all__ = [
    "HTMLRenderer",
    "LaTeXRenderer",
    "TypstRenderer",
    "compile_pdf",
    "escape_html",
    "escape_latex",
    "escape_typst",
]

_LATEX = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}
#: symbols which pdflatex does not read from utf8 text, as LaTeX math; the
#: latin letters with accents and umlauts are read with T1 and utf8 input
_LATEX_SYMBOLS = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "Γ": r"\Gamma",
    "δ": r"\delta",
    "Δ": r"\Delta",
    "ε": r"\varepsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "Θ": r"\Theta",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "Λ": r"\Lambda",
    "μ": r"\mu",
    "µ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "Π": r"\Pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "Σ": r"\Sigma",
    "τ": r"\tau",
    "φ": r"\varphi",
    "Φ": r"\Phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "Ψ": r"\Psi",
    "ω": r"\omega",
    "Ω": r"\Omega",
    "∞": r"\infty",
    "±": r"\pm",
    "∓": r"\mp",
    "×": r"\times",
    "·": r"\cdot",
    "÷": r"\div",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "∼": r"\sim",
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "−": "-",
    "√": r"\surd",
    "∑": r"\sum",
    "∫": r"\int",
    "∂": r"\partial",
    "∈": r"\in",
    "°": r"^{\circ}",
    "⁰": "^{0}",
    "¹": "^{1}",
    "²": "^{2}",
    "³": "^{3}",
    "⁴": "^{4}",
    "⁵": "^{5}",
    "⁶": "^{6}",
    "⁷": "^{7}",
    "⁸": "^{8}",
    "⁹": "^{9}",
    "⁻": "^{-}",
    "₀": "_{0}",
    "₁": "_{1}",
    "₂": "_{2}",
    "₃": "_{3}",
}

#: text symbols which pdflatex does not read from utf8 text
_LATEX_TEXT = {
    "\u00a0": "~",
    "\u2009": r"\,",
    "\u2013": "--",
    "\u2014": "---",
    "\u2018": "`",
    "\u2019": "'",
    "\u201c": "``",
    "\u201d": "''",
    "\u2026": r"\ldots{}",
    "\u2030": r"\textperthousand{}",
}
_TYPST = set("\\#$*_@<>[]`~=/")
_KIND = {Table: "Table", Figure: "Figure", Equation: "Equation", Section: "Section"}


def escape_html(text: str) -> str:
    """Escape text for HTML.

    Args:
        text: literal text.

    Returns:
        Text for HTML content and attributes.
    """
    return html.escape(text, quote=True)


def escape_latex(text: str) -> str:
    """Escape text for LaTeX.

    Args:
        text: literal text.

    Returns:
        Text which LaTeX typesets literally. pdflatex reads it with T1 and
        utf8 input: symbols (greek letters, `±`, `∞`, superscripts) are math
        (`ensuremath`), latin letters with accents stay as they are.
    """
    return "".join(_escape_latex_char(c) for c in text)


def _escape_latex_char(c: str) -> str:
    """Escape a character for LaTeX, see `escape_latex`."""
    if c in _LATEX:
        return _LATEX[c]
    if c.isascii():
        return c
    if c in _LATEX_SYMBOLS:
        return f"\\ensuremath{{{_LATEX_SYMBOLS[c]}}}"
    if c in _LATEX_TEXT:
        return _LATEX_TEXT[c]
    if c.isalpha() and ord(c) < 0x180:
        return c
    # a compatibility form, e.g. a ligature, else the character as it is
    folded = unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode()
    return "".join(_LATEX.get(f, f) for f in folded) if folded else c


def escape_typst(text: str) -> str:
    """Escape text for Typst markup.

    Args:
        text: literal text.

    Returns:
        Text which Typst typesets literally.
    """
    return "".join("\\" + c if c in _TYPST else c for c in text)


def _typst_string(text: str) -> str:
    """Typst string literal."""
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _after_call(text: str) -> str:
    """Escape the start of Typst markup which follows an embedded call.

    Typst ends an embedded expression such as `#raw("x")` with a ";", which it
    consumes, and continues it with "(" (arguments), "[" (a content argument,
    which `escape_typst` escapes) or "." and a name (a field). Text which
    starts with one of them right after a call is escaped, so it stays text.

    Args:
        text: rendered Typst markup which directly follows an embedded call.

    Returns:
        The markup, its first character escaped if Typst would read it as
        part of the call.
    """
    if text[:1] in (";", "(") or (
        text[:1] == "." and len(text) > 1 and text[1].isidentifier()
    ):
        return "\\" + text
    return text


def compile_pdf(typ_path: Path, pdf_path: Path | None = None) -> Path:
    """Compile a Typst file to PDF.

    Args:
        typ_path: Typst source; its folder is the root of the paths in it.
        pdf_path: PDF to write, by default next to the source.

    Returns:
        Path of the PDF.

    Raises:
        RuntimeError: when Typst cannot compile the source.
    """
    typ_path = Path(typ_path)
    out = Path(pdf_path) if pdf_path else typ_path.with_suffix(".pdf")
    try:
        typst.compile(str(typ_path), output=str(out), root=str(typ_path.parent))
    except Exception as err:
        raise RuntimeError(f"typst could not compile '{typ_path.name}': {err}") from err
    return out


class _Renderer:
    """Numbering of blocks and the dispatch on blocks and spans."""

    def __init__(self) -> None:
        self._numbers: dict[int, int] = {}
        self._labels: dict[str, tuple[str, int]] = {}

    def _index(self, blocks: tuple[Block, ...]) -> None:
        """Number the blocks per kind in document order and collect labels."""
        self._numbers = {}
        self._labels = {}
        counts: dict[str, int] = {}

        def visit(items: tuple[Block, ...]) -> None:
            for item in items:
                kind = _KIND.get(type(item))
                if kind is not None:
                    counts[kind] = counts.get(kind, 0) + 1
                    self._numbers[id(item)] = counts[kind]
                    label = getattr(item, "label", "")
                    if label:
                        if label in self._labels:
                            raise ValueError(f"duplicate label '{label}'")
                        self._labels[label] = (kind, counts[kind])
                if isinstance(item, Section):
                    visit(item.blocks)

        visit(blocks)

    def _prepare(self, block: Block) -> None:
        """Index a single block which is rendered outside of a document."""
        if id(block) not in self._numbers:
            self._index((block,))

    def _target(self, label: str) -> tuple[str, int]:
        if label not in self._labels:
            raise ValueError(f"reference to the unknown label '{label}'")
        return self._labels[label]

    def block(self, block: Block) -> str:
        """Render a block.

        Args:
            block: the block.

        Returns:
            The rendered block.
        """
        raise NotImplementedError

    def render(self, document: Document) -> str:
        """Render a document.

        Args:
            document: the document.

        Returns:
            The rendered document.
        """
        raise NotImplementedError


def _mathml(latex: str, display: str = "inline") -> str:
    """MathML of a LaTeX formula for HTML.

    The namespace of MathML is left out: HTML puts a `math` element into it,
    and a page which names no URL is evidently one which needs no network.
    """
    mathml = latex2mathml.converter.convert(latex, display=display)
    return mathml.replace(' xmlns="http://www.w3.org/1998/Math/MathML"', "", 1)


class HTMLRenderer(_Renderer):
    """Render as HTML with MathML.

    `render` writes a page of its own, `fragment` the content of a document to
    embed into another page, e.g. the report of a fit.
    """

    def __init__(self) -> None:
        """Create a renderer."""
        super().__init__()
        self._cites: dict[str, int] = {}

    @override
    def render(self, document: Document) -> str:
        body = self.fragment(document)
        title = escape_html(document.title)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f"<title>{title}</title>\n"
            f"<style>\n{_CSS}</style>\n</head>\n<body>\n"
            f"<h1>{title}</h1>\n{body}\n</body>\n</html>\n"
        )

    def fragment(self, document: Document) -> str:
        """Render the blocks and the references of a document without a page.

        The blocks are numbered and referenced across the whole document, so
        a reference into another section resolves; the title is left to the
        page the fragment is embedded in. The style of the page applies.

        Args:
            document: the document.

        Returns:
            The HTML of the sections and of the list of references.
        """
        self._index(document.blocks)
        self._cites = {}
        body = "\n".join(self._block(b) for b in document.blocks)
        return body + self._references(document)

    def _references(self, document: Document) -> str:
        """List of the references in the order of their first citation."""
        if not document.references:
            return ""
        order = {key: n for n, key in enumerate(self._cites)}
        refs = sorted(document.references, key=lambda r: order.get(r.key, len(order)))
        items = "\n".join(
            f'<li id="ref-{escape_html(r.key)}">{escape_html(r.text())}</li>'
            for r in refs
        )
        return f'\n<section id="references">\n<h2>References</h2>\n<ol>\n{items}\n</ol>\n</section>'

    @override
    def block(self, block: Block) -> str:
        self._prepare(block)
        return self._block(block)

    def _block(self, block: Block) -> str:
        number = self._numbers.get(id(block), 0)
        match block:
            case Paragraph():
                return f"<p>{self._inlines(block.content)}</p>"
            case Equation():
                ident = f' id="{escape_html(block.label)}"' if block.label else ""
                mathml = _mathml(block.latex, display="block")
                return (
                    f'<div class="equation"{ident}>{mathml}'
                    f'<span class="eqno">({number})</span></div>'
                )
            case Table():
                return self._table(block, number)
            case Figure():
                return (
                    f'<figure id="{escape_html(block.label)}">'
                    f'<img src="{escape_html(block.path)}" alt="Figure {number}">'
                    f"<figcaption>Figure {number}. "
                    f"{self._inlines(block.caption)}</figcaption></figure>"
                )
            case Section():
                ident = f' id="{escape_html(block.label)}"' if block.label else ""
                level = min(max(block.level, 1) + 1, 6)
                inner = "\n".join(self._block(b) for b in block.blocks)
                return (
                    f"<section{ident}>\n<h{level}>{escape_html(block.title)}"
                    f"</h{level}>\n{inner}\n</section>"
                )
        raise TypeError(f"unknown block {block!r}")

    def _table(self, table: Table, number: int) -> str:
        styles = {"l": "left", "r": "right", "c": "center"}
        align = [f' style="text-align: {styles[a]}"' for a in table.alignment]
        head = "".join(
            f"<th{a}>{escape_html(h)}</th>"
            for h, a in zip(table.header, align, strict=True)
        )
        rows = "\n".join(
            "<tr>"
            + "".join(
                f"<td{a}>{self._cell(c)}</td>" for c, a in zip(row, align, strict=True)
            )
            + "</tr>"
            for row in table.rows
        )
        return (
            f'<table id="{escape_html(table.label)}">\n'
            f"<caption>Table {number}. {self._inlines(table.caption)}</caption>\n"
            f"<thead><tr>{head}</tr></thead>\n<tbody>\n{rows}\n</tbody>\n</table>"
        )

    def _cell(self, cell: str | Code | Math) -> str:
        if isinstance(cell, str):
            return escape_html(cell)
        return self._inline(cell)

    def _inlines(self, content: tuple[Inline, ...]) -> str:
        return "".join(self._inline(i) for i in content)

    def _inline(self, span: Inline) -> str:
        match span:
            case Text():
                return escape_html(span.text)
            case Code():
                return f"<code>{escape_html(span.text)}</code>"
            case Math():
                return _mathml(span.latex)
            case Cite():
                links = []
                for key in span.keys:
                    n = self._cites.setdefault(key, len(self._cites) + 1)
                    links.append(f'<a href="#ref-{escape_html(key)}">{n}</a>')
                return "[" + ", ".join(links) + "]"
            case Ref():
                kind, n = self._target(span.label)
                label = f"{kind} {n}"
                return f'<a href="#{escape_html(span.label)}">{label}</a>'
        raise TypeError(f"unknown span {span!r}")


_CSS = """\
body { font-family: sans-serif; max-width: 52em; margin: 2em auto; padding: 0 1em; line-height: 1.5; }
table { border-collapse: collapse; margin: 1em 0; }
caption { text-align: left; font-weight: bold; margin-bottom: 0.5em; }
th, td { border-bottom: 1px solid #ccc; padding: 0.2em 0.8em; }
figure { margin: 1em 0; }
figure img { max-width: 100%; }
.equation { display: flex; justify-content: space-between; align-items: center; margin: 1em 0; }
code { font-size: 0.9em; }
"""


class TypstRenderer(_Renderer):
    """Render as Typst source."""

    @override
    def render(self, document: Document) -> str:
        self._index(document.blocks)
        body = "\n\n".join(self._block(b) for b in document.blocks)
        if document.references:
            body += f'\n\n#bibliography("{BIBLIOGRAPHY_FILE}", style: "{STYLE_FILE}")'
        return (
            '#set page(paper: "a4", margin: 2cm)\n'
            "#set text(size: 10pt)\n"
            '#set heading(numbering: "1.1")\n'
            '#set math.equation(numbering: "(1)")\n'
            "// a table longer than a page continues on the next page, its\n"
            "// caption above it and its header repeated\n"
            "#show figure.where(kind: table): set block(breakable: true)\n"
            "#show figure.where(kind: table): set figure.caption(position: top)\n\n"
            f'#align(center, text(size: 16pt, weight: "bold")'
            f"[{escape_typst(document.title)}])\n\n{body}\n"
        )

    @override
    def block(self, block: Block) -> str:
        self._prepare(block)
        return self._block(block)

    def _block(self, block: Block) -> str:
        match block:
            case Paragraph():
                return self._inlines(block.content)
            case Equation():
                label = f" <{block.label}>" if block.label else ""
                return f"$ {block.typst.strip()} ${label}"
            case Table():
                n = len(block.header)
                names = {"l": "left", "r": "right", "c": "center"}
                align = ", ".join(names[a] for a in block.alignment)
                cells = [f"[{escape_typst(h)}]" for h in block.header]
                head = f"table.header({', '.join(cells)})"
                body = ",\n  ".join(self._cell(c) for row in block.rows for c in row)
                rows = f",\n  {body}" if body else ""
                return (
                    f"#figure(\n"
                    f"  table(\n  columns: {n},\n  align: ({align},),\n  {head}{rows},\n  ),\n"
                    f"  caption: [{self._inlines(block.caption)}],\n) <{block.label}>"
                )
            case Figure():
                return (
                    f"#figure(\n  image({_typst_string(block.path)}, width: 100%),\n"
                    f"  caption: [{self._inlines(block.caption)}],\n) <{block.label}>"
                )
            case Section():
                label = f" <{block.label}>" if block.label else ""
                head = f"{'=' * max(block.level, 1)} {escape_typst(block.title)}{label}"
                return "\n\n".join([head, *(self._block(b) for b in block.blocks)])
        raise TypeError(f"unknown block {block!r}")

    def _cell(self, cell: str | Code | Math) -> str:
        if isinstance(cell, str):
            return f"[{escape_typst(cell)}]"
        if isinstance(cell, Code):
            return f"raw({_typst_string(cell.text)})"
        return f"[{self._inline(cell)}]"

    def _inlines(self, content: tuple[Inline, ...]) -> str:
        out: list[str] = []
        for span, previous in zip(content, (None, *content), strict=False):
            text = self._inline(span)
            if isinstance(previous, (Code, Cite, Ref)):
                text = _after_call(text)
            out.append(text)
        return "".join(out)

    def _inline(self, span: Inline) -> str:
        match span:
            case Text():
                return escape_typst(span.text)
            case Code():
                return f"#raw({_typst_string(span.text)})"
            case Math():
                return f"${span.typst.strip()}$"
            case Cite():
                return " ".join(f"#cite(<{key}>)" for key in span.keys)
            case Ref():
                self._target(span.label)
                return f"#ref(<{span.label}>)"
        raise TypeError(f"unknown span {span!r}")


class LaTeXRenderer(_Renderer):
    """Render as a LaTeX fragment or as a standalone document."""

    _SECTIONS = ("section", "subsection", "subsubsection", "paragraph")

    @override
    def render(self, document: Document) -> str:
        self._index(document.blocks)
        body = "\n\n".join(self._block(b) for b in document.blocks)
        if document.references:
            body += "\n\n\\bibliographystyle{unsrt}\n\\bibliography{references}"
        return body + "\n"

    def render_standalone(self, document: Document) -> str:
        """Render as a LaTeX document with preamble.

        Args:
            document: the document.

        Returns:
            The LaTeX document.
        """
        body = self.render(document)
        return (
            "\\documentclass{article}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{longtable}\n"
            "\\usepackage{hyperref}\n"
            f"\\title{{{escape_latex(document.title)}}}\n"
            "\\date{}\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
            f"{body}\n"
            "\\end{document}\n"
        )

    @override
    def block(self, block: Block) -> str:
        self._prepare(block)
        return self._block(block)

    def _block(self, block: Block) -> str:
        match block:
            case Paragraph():
                return self._inlines(block.content)
            case Equation():
                label = f"\\label{{{block.label}}}" if block.label else ""
                return (
                    f"\\begin{{equation}}{label}\n{block.latex.strip()}\n"
                    "\\end{equation}"
                )
            case Table():
                cols = block.alignment
                head = " & ".join(escape_latex(h) for h in block.header)
                rows = "".join(
                    " & ".join(self._cell(c) for c in row) + " \\\\\n"
                    for row in block.rows
                )
                # a longtable breaks across pages: the caption and the label
                # head the first page, the header repeats on every page
                n = len(cols)
                return (
                    f"\\begin{{longtable}}{{{cols}}}\n"
                    f"\\caption{{{self._inlines(block.caption)}}}"
                    f"\\label{{{block.label}}} \\\\\n"
                    f"\\toprule\n{head} \\\\\n\\midrule\n\\endfirsthead\n"
                    f"\\multicolumn{{{n}}}{{l}}{{\\small\\tablename~\\thetable{{}} "
                    "(continued)} \\\\\n"
                    f"\\toprule\n{head} \\\\\n\\midrule\n\\endhead\n"
                    f"\\bottomrule\n\\endlastfoot\n{rows}"
                    "\\end{longtable}"
                )
            case Figure():
                path = Path(block.path)
                if path.suffix.lower() == ".svg":
                    path = path.with_suffix(".png")
                return (
                    "\\begin{figure}[htbp]\n\\centering\n"
                    f"\\includegraphics[width=\\linewidth]{{{path.as_posix()}}}\n"
                    f"\\caption{{{self._inlines(block.caption)}}}\n"
                    f"\\label{{{block.label}}}\n\\end{{figure}}"
                )
            case Section():
                command = self._SECTIONS[min(max(block.level, 1), 4) - 1]
                label = f"\\label{{{block.label}}}" if block.label else ""
                head = f"\\{command}{{{escape_latex(block.title)}}}{label}"
                return "\n\n".join([head, *(self._block(b) for b in block.blocks)])
        raise TypeError(f"unknown block {block!r}")

    def _cell(self, cell: str | Code | Math) -> str:
        if isinstance(cell, str):
            return escape_latex(cell)
        return self._inline(cell)

    def _inlines(self, content: tuple[Inline, ...]) -> str:
        return "".join(self._inline(i) for i in content)

    def _inline(self, span: Inline) -> str:
        match span:
            case Text():
                return escape_latex(span.text)
            case Code():
                return f"\\texttt{{{escape_latex(span.text)}}}"
            case Math():
                return f"${span.latex}$"
            case Cite():
                return f"\\cite{{{','.join(span.keys)}}}"
            case Ref():
                kind, _ = self._target(span.label)
                if kind == "Equation":
                    return f"Equation~\\eqref{{{span.label}}}"
                return f"{kind}~\\ref{{{span.label}}}"
        raise TypeError(f"unknown span {span!r}")
