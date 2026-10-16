"""Tests of the document model and its renderers."""

from dataclasses import replace
from pathlib import Path

import pytest

from sbmlsim.report.document import (
    Cite,
    Code,
    Document,
    Equation,
    Figure,
    Math,
    Ref,
    Section,
    Table,
    Text,
    p,
)
from sbmlsim.report.references import bibliography, write_bibliography
from sbmlsim.report.render import (
    HTMLRenderer,
    LaTeXRenderer,
    TypstRenderer,
    compile_pdf,
    escape_latex,
    escape_typst,
)

NASTY = r"GU__HCTZABS_k 100% #1 & $x$ <a> \path"


def _doc() -> Document:
    return Document(
        title="Fit",
        blocks=(
            Section(
                "Methods",
                (
                    p(
                        "The residual ",
                        Math(r"r_{i,k}", "r_(i,k)"),
                        " of ",
                        Code("GU__HCTZABS_k"),
                        ", see ",
                        Ref("tab:params"),
                        ".",
                    ),
                    Equation(
                        r"F(\vec{p}) = \frac{1}{2} \sum_k w_k r_k^2",
                        "F(arrow(p)) = 1/2 sum_k w_k r_k^2",
                        label="eq:cost",
                    ),
                    Table(
                        caption=(Text("Parameters of the fit."),),
                        header=("Parameter", "Value"),
                        rows=((Code("GU__HCTZABS_k"), "1.5e-3"), (Code(NASTY), "2")),
                        label="tab:params",
                    ),
                    Figure(
                        "plots/waterfall.svg",
                        (Text("Waterfall."),),
                        label="fig:waterfall",
                    ),
                ),
            ),
        ),
    )


@pytest.mark.parametrize("renderer", [HTMLRenderer(), TypstRenderer(), LaTeXRenderer()])
def test_every_block_renders(renderer) -> None:
    out = renderer.render(_doc())
    assert "Methods" in out and "1.5e-3" in out


def test_html_math_is_mathml_and_refs_are_numbered() -> None:
    out = HTMLRenderer().render(_doc())
    assert "<math" in out and "mathjax" not in out.lower()
    assert 'href="#tab:params"' in out and "Table 1" in out


def test_special_characters_are_literal() -> None:
    assert escape_latex("a_b%c#d&e$f") == r"a\_b\%c\#d\&e\$f"
    assert escape_typst("a_b#c*d$e") == r"a\_b\#c\*d\$e"
    html = HTMLRenderer().render(_doc())
    assert "&lt;a&gt;" in html


def test_typst_compiles(tmp_path: Path) -> None:
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots" / "waterfall.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'
    )
    doc = _doc()
    doc = replace(
        doc,
        blocks=(*doc.blocks, p("cited", Cite(("Raue2009",)))),
    )
    doc = replace(doc, references=bibliography(doc))
    write_bibliography(doc.references, tmp_path)
    typ = tmp_path / "report.typ"
    typ.write_text(TypstRenderer().render(doc))
    pdf = compile_pdf(typ)
    assert pdf.read_bytes().startswith(b"%PDF")


def test_latex_is_balanced() -> None:
    tex = LaTeXRenderer().render(_doc())
    assert tex.count("\\begin{") == tex.count("\\end{")
    assert tex.count("{") == tex.count("}")


def test_citations_and_references_in_each_format() -> None:
    doc = Document("T", (p("see ", Cite(("a", "b")), " and ", Cite(("a",))),))
    html = HTMLRenderer().render(doc)
    assert '[<a href="#ref-a">1</a>, <a href="#ref-b">2</a>]' in html
    assert '[<a href="#ref-a">1</a>]' in html
    assert "#cite(<a>) #cite(<b>)" in TypstRenderer().render(doc)
    assert r"\cite{a,b}" in LaTeXRenderer().render(doc)


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        HTMLRenderer().render(Document("T", (p(Ref("nope")),)))


def test_latex_refs_and_standalone() -> None:
    renderer = LaTeXRenderer()
    tex = renderer.render(_doc())
    assert r"Table~\ref{tab:params}" in tex
    assert r"\includegraphics[width=\linewidth]{plots/waterfall.png}" in tex
    full = renderer.render_standalone(_doc())
    assert full.startswith(r"\documentclass{article}")
    assert full.count("\\begin{") == full.count("\\end{")


def test_table_alignment_is_checked() -> None:
    with pytest.raises(ValueError, match="align"):
        Table((), ("a", "b"), (), label="t", align="l")
    with pytest.raises(ValueError, match="cells"):
        Table((), ("a", "b"), (("1",),), label="t")


def test_compile_failure_raises_runtime_error(tmp_path: Path) -> None:
    typ = tmp_path / "bad.typ"
    typ.write_text("#unknown_function()")
    with pytest.raises(RuntimeError, match=r"could not compile 'bad\.typ'"):
        compile_pdf(typ)


@pytest.mark.parametrize("after", ["s", "-x", " and"])
def test_typst_ref_followed_by_text_compiles(tmp_path: Path, after: str) -> None:
    table = Table((Text("T."),), ("A",), (("1",),), label="tab:p")
    doc = Document("T", (p(Ref("tab:p"), after), table))
    out = TypstRenderer().render(doc)
    assert f"#ref(<tab:p>){after}" in out
    typ = tmp_path / "report.typ"
    typ.write_text(out)
    assert compile_pdf(typ).read_bytes().startswith(b"%PDF")


def test_typst_cites_in_a_row_are_separate_calls() -> None:
    out = TypstRenderer().render(Document("T", (p(Cite(("a",)), Cite(("b",)), "s"),)))
    assert "#cite(<a>)#cite(<b>)s" in out


def _long_table_doc(n: int = 150) -> Document:
    """A document with a table of several pages."""
    rows = tuple(
        (Code(f"data/study_{i:03d}/Fig{i}.tsv"), Code(f"sha{i:03d}abcdef"))
        for i in range(n)
    )
    table = Table(
        caption=p("Checksums of the files.").content,
        header=("File", "SHA-256"),
        rows=rows,
        label="tab:long",
    )
    return Document("T", (Section("S", (p("see ", Ref("tab:long")), table)),))


def test_typst_long_table_breaks_across_pages(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    typ = tmp_path / "long.typ"
    typ.write_text(TypstRenderer().render(_long_table_doc()))
    reader = pypdf.PdfReader(compile_pdf(typ))
    texts = [page.extract_text() for page in reader.pages]
    assert len(texts) > 2, "150 rows do not fit on two pages"
    assert all("study_" in text for text in texts[1:]), "no empty page"
    text = "\n".join(texts)
    for i in range(150):
        assert text.count(f"study_{i:03d}/") == 1, f"row {i}"
    assert text.count("Checksums of the files.") == 1


def test_latex_tables_are_longtables() -> None:
    renderer = LaTeXRenderer()
    tex = renderer.render(_long_table_doc())
    assert "\\begin{table}" not in tex and "\\begin{tabular}" not in tex
    begin = tex.index("\\begin{longtable}{lr}")
    end = tex.index("\\end{longtable}")
    body = tex[begin:end]
    assert body.index("\\caption{") < body.index("\\label{tab:long}")
    assert body.index("\\label{tab:long}") < body.index("\\toprule")
    assert "\\endfirsthead" in body and "\\endhead" in body
    assert body.count("study\\_") == 150
    full = renderer.render_standalone(_long_table_doc())
    assert "\\usepackage{longtable}" in full and "\\usepackage{booktabs}" in full
    assert full.count("\\begin{") == full.count("\\end{")


def test_latex_symbols_are_math_and_letters_stay() -> None:
    assert escape_latex("χ² at α, -∞ to ∞, ± 2") == (
        r"\ensuremath{\chi}\ensuremath{^{2}} at \ensuremath{\alpha}, "
        r"-\ensuremath{\infty} to \ensuremath{\infty}, \ensuremath{\pm} 2"
    )
    assert escape_latex("Kőnig, Müller, Ångström") == "Kőnig, Müller, Ångström"
    assert escape_latex("1–2 x y") == "1--2 x~y"
    table = Table(
        caption=p("Bounds.").content,
        header=("Parameter", "Upper"),
        rows=((Code("k_μ"), "∞"),),
        label="tab:b",
    )
    tex = LaTeXRenderer().render(Document("T", (table,)))
    assert r"\texttt{k\_\ensuremath{\mu}} & \ensuremath{\infty}" in tex


def test_typst_keeps_the_text_after_an_inline_call(tmp_path: Path) -> None:
    """Text right after `#raw`, `#ref` or `#cite` is text, not part of the call.

    Typst ends an embedded expression with a ";", which it consumes, and
    continues it with "(" (arguments) or "." and a name (a field); the
    renderer escapes them so the PDF has the text.
    """
    pypdf = pytest.importorskip("pypdf")
    doc = Document(
        "T",
        (
            Section(
                "S",
                (
                    p(Code("wt-models-review"), "; their SHA-256 checksums"),
                    p("see ", Ref("sec:s"), "; then ", Code("x"), "(y) z"),
                    p(Code("a"), ".b c and ", Code("d"), ". End"),
                    p("cited ", Cite(("Raue2009",)), "; next"),
                ),
                label="sec:s",
            ),
        ),
    )
    doc = replace(doc, references=bibliography(doc))
    write_bibliography(doc.references, tmp_path)
    typ = tmp_path / "report.typ"
    typ.write_text(TypstRenderer().render(doc))
    text = pypdf.PdfReader(compile_pdf(typ)).pages[0].extract_text()
    assert "wt-models-review; their SHA-256 checksums" in text
    assert "; then x(y) z" in text
    assert "a.b c and d. End" in text
    assert "; next" in text
