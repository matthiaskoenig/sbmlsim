"""Tests of the references of a report."""

from dataclasses import replace
from pathlib import Path

import pytest
from pypdf import PdfReader

from sbmlsim.report.document import Cite, Document, Figure, Section, Table, Text, p
from sbmlsim.report.references import (
    bibliography,
    cited_keys,
    load_references,
    to_bibtex,
    write_bibliography,
)
from sbmlsim.report.render import (
    HTMLRenderer,
    LaTeXRenderer,
    TypstRenderer,
    compile_pdf,
)


def test_bundled_references_parse() -> None:
    refs = load_references()
    assert {"Raue2009", "Branch1999", "Koenig2026_sbmlsim"} <= set(refs)
    assert refs["Raue2009"].fields["doi"] == "10.1093/bioinformatics/btp358"


def test_cited_in_order_and_versioned() -> None:
    doc = Document(
        "t",
        (
            Section(
                "s",
                (
                    p(
                        "x ",
                        Cite(("Raue2009",)),
                        Cite(("Koenig2026_sbmlsim", "Raue2009")),
                    ),
                ),
            ),
        ),
    )
    assert cited_keys(doc) == ["Raue2009", "Koenig2026_sbmlsim"]
    refs = bibliography(doc, versions={"sbmlsim": "0.10.0"})
    assert refs[1].fields["note"] == "Version 0.10.0"
    bib = to_bibtex(refs)
    assert "@misc{Koenig2026_sbmlsim" in bib
    assert "  note = {Version 0.10.0}," in bib
    assert "package" not in bib


def test_unknown_key_raises() -> None:
    doc = Document("t", (p(Cite(("Nobody1900",))),))
    with pytest.raises(KeyError):
        bibliography(doc)


def test_captions_are_scanned_and_bibtex_round_trips(tmp_path: Path) -> None:
    table = Table((Cite(("Storn1997",)),), ("A",), (("1",),), label="t")
    fig = Figure("f.svg", (Text("x"), Cite(("McKay1979",))), label="f")
    doc = Document("t", (Section("s", (table, fig)),))
    assert cited_keys(doc) == ["Storn1997", "McKay1979"]
    refs = bibliography(doc)
    path = tmp_path / "references.bib"
    path.write_text(to_bibtex(refs))
    assert tuple(load_references(path).values()) == refs


def test_text_and_nested_braces() -> None:
    ref = load_references()["Welsh2023"]
    assert ref.fields["title"].startswith("{libRoadRunner} 2.0")
    text = ref.text()
    assert text.startswith("Welsh C, Xu J")
    assert "(2023)" in text and "libRoadRunner 2.0" in text
    assert text.endswith("doi:10.1093/bioinformatics/btac770.")


def _cited() -> Document:
    doc = Document("T", (p("a", Cite(("Raue2009",)), Cite(("Branch1999",))),))
    return replace(doc, references=bibliography(doc))


def test_renderers_list_references_only_when_present() -> None:
    doc = _cited()
    html = HTMLRenderer().render(doc)
    assert '<section id="references">' in html
    assert html.index('id="ref-Raue2009"') < html.index('id="ref-Branch1999"')
    assert (
        '#bibliography("references.bib", style: "references.csl")'
        in TypstRenderer().render(doc)
    )
    tex = LaTeXRenderer().render(doc)
    assert "\\bibliographystyle{unsrt}\\n\\bibliography{references}" in tex.replace(
        "\n", "\\n"
    )
    bare = Document("T", doc.blocks)
    assert "references" not in HTMLRenderer().render(bare)
    assert "bibliography" not in TypstRenderer().render(bare)
    assert "bibliography" not in LaTeXRenderer().render(bare)


def test_every_bundled_reference_compiles_in_typst(tmp_path: Path) -> None:
    refs = load_references()
    doc = Document("T", (p("x", Cite(tuple(refs))),))
    doc = replace(doc, references=bibliography(doc, versions={"sbmlsim": "1.0"}))
    write_bibliography(doc.references, tmp_path)
    typ = tmp_path / "report.typ"
    typ.write_text(TypstRenderer().render(doc))
    assert compile_pdf(typ).read_bytes().startswith(b"%PDF")


def _pdf_text(path: Path) -> str:
    """Text of a PDF, its lines joined by spaces."""
    text = " ".join(page.extract_text() for page in PdfReader(path).pages)
    return " ".join(text.split())


def test_the_pdf_lists_the_versions_of_the_software(tmp_path: Path) -> None:
    """The version of a software is in the reference list of the PDF.

    The IEEE style of Typst drops the field `note` which carries the version,
    the bundled style prints it at the end of the entry.
    """
    doc = Document(
        "T", (p("x", Cite(("Koenig2026_sbmlsim", "Virtanen2020", "Raue2009"))),)
    )
    versions = {"sbmlsim": "0.10.0", "scipy": "1.18.1"}
    doc = replace(doc, references=bibliography(doc, versions=versions))
    bib, csl = write_bibliography(doc.references, tmp_path)
    assert (bib.name, csl.name) == ("references.bib", "references.csl")
    assert "note = {Version 0.10.0}" in bib.read_text()
    typ = tmp_path / "report.typ"
    typ.write_text(TypstRenderer().render(doc))

    text = _pdf_text(compile_pdf(typ))
    assert "SBML simulation made easy" in text
    assert "doi: 10.5281/zenodo.3597770. Version 0.10.0." in text
    assert "Version 1.18.1." in text
    # an entry without a note is the IEEE entry
    assert "Raue" in text and text.count("Version") == 2
