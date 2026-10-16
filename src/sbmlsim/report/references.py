"""References of a report: a bundled BibTeX file and the citations of a document.

The bundled `sbmlsim/resources/references.bib` holds the software and the
methods a report cites. `bibliography` selects the references a document cites,
in the order of their first citation, and adds the installed version to the
entries of software packages. The HTML report lists them itself, the Typst and
LaTeX reports read the file written by `to_bibtex`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from sbmlsim import RESOURCES_DIR
from sbmlsim.report.document import (
    Block,
    Cite,
    Document,
    Figure,
    Inline,
    Paragraph,
    Section,
    Table,
)

__all__ = [
    "REFERENCES_BIB",
    "Reference",
    "bibliography",
    "cited_keys",
    "load_references",
    "to_bibtex",
]

REFERENCES_BIB = RESOURCES_DIR / "references.bib"

_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,")
_NAME = re.compile(r"\s*(\w+)\s*=\s*")


@dataclass(frozen=True)
class Reference:
    """Entry of a BibTeX file.

    Attributes:
        key: citation key.
        entry_type: BibTeX entry type, e.g. `article` or `software`.
        fields: fields of the entry, braces of the value removed at the outer
            level only; `package` is read by `bibliography` and not written
            by `to_bibtex`.
    """

    key: str
    entry_type: str
    fields: dict[str, str]

    def text(self) -> str:
        """Plain citation text.

        Returns:
            "Authors (year). Title. Journal/Publisher. doi:..." with the
            volume, number and pages of an article and the note (the version)
            of software.
        """
        f = {name: _plain(value) for name, value in self.fields.items()}
        head = _authors(f.get("author", ""))
        if "year" in f:
            head = f"{head} ({f['year']})" if head else f"({f['year']})"
        parts = [head] if head else []
        if "title" in f:
            parts.append(f["title"])
        source = f.get("journal") or f.get("publisher") or ""
        if self.entry_type == "article":
            volume = f.get("volume", "")
            if f.get("number"):
                volume += f"({f['number']})"
            if f.get("pages"):
                volume += f":{f['pages']}"
            source = f"{source} {volume}".strip()
        elif "note" in f:
            source = f"{source} {f['note']}".strip()
        if source:
            parts.append(source)
        if "doi" in f:
            parts.append(f"doi:{f['doi']}")
        return ". ".join(p.rstrip(".") for p in parts) + "."


def _plain(value: str) -> str:
    """Value without the braces which protect the case and page ranges as dashes."""
    return value.replace("{", "").replace("}", "").replace("--", "-")


def _authors(value: str) -> str:
    """Authors "Last, First and ..." as "Last F, ..."."""
    names = []
    for author in (a.strip() for a in value.split(" and ") if a.strip()):
        if author == "others":
            names.append("et al.")
            continue
        last, _, first = author.partition(",")
        initials = "".join(w[0] for w in re.split(r"[\s-]+", first.strip()) if w)
        names.append(f"{last.strip()} {initials}".strip())
    return ", ".join(names)


def _braced(text: str, start: int) -> tuple[str, int]:
    """Value of the braces opening at `start`, with the position behind them."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    raise ValueError("unbalanced braces in BibTeX")


def _parse(text: str) -> Iterator[Reference]:
    """Parse the entries `@type{key, name = {value}, ...}` of BibTeX text."""
    pos = 0
    while (m := _ENTRY.search(text, pos)) is not None:
        fields: dict[str, str] = {}
        pos = m.end()
        while True:
            while pos < len(text) and text[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(text):
                raise ValueError(f"unterminated BibTeX entry '{m.group(2)}'")
            if text[pos] == "}":
                pos += 1
                break
            name = _NAME.match(text, pos)
            if name is None:
                raise ValueError(f"cannot parse BibTeX entry '{m.group(2)}' at {pos}")
            pos = name.end()
            if text[pos] == "{":
                value, pos = _braced(text, pos)
            else:
                bare = re.compile(r"[^,}\s]+").match(text, pos)
                if bare is None:
                    raise ValueError(f"empty field '{name.group(1)}' in '{m.group(2)}'")
                value, pos = bare.group(), bare.end()
            fields[name.group(1).lower()] = " ".join(value.split())
        yield Reference(m.group(2), m.group(1).lower(), fields)


def load_references(path: Path | None = None) -> dict[str, Reference]:
    """Load the references of a BibTeX file.

    Args:
        path: BibTeX file, by default the bundled `references.bib`.

    Returns:
        The references by key.

    Raises:
        ValueError: on a key which occurs twice or text which is not BibTeX.
    """
    source = Path(path) if path else REFERENCES_BIB
    refs: dict[str, Reference] = {}
    for ref in _parse(source.read_text(encoding="utf-8")):
        if ref.key in refs:
            raise ValueError(f"duplicate reference key '{ref.key}' in '{source}'")
        refs[ref.key] = ref
    return refs


def _cites(content: Iterable[Inline]) -> Iterator[str]:
    for span in content:
        if isinstance(span, Cite):
            yield from span.keys


def _block_cites(blocks: Iterable[Block]) -> Iterator[str]:
    for block in blocks:
        if isinstance(block, Paragraph):
            yield from _cites(block.content)
        elif isinstance(block, Table | Figure):
            yield from _cites(block.caption)
        elif isinstance(block, Section):
            yield from _block_cites(block.blocks)


def cited_keys(document: Document) -> list[str]:
    """Keys cited by a document.

    Args:
        document: the document.

    Returns:
        The keys in the order of their first citation, without repetition.
    """
    return list(dict.fromkeys(_block_cites(document.blocks)))


def bibliography(
    document: Document, versions: Mapping[str, str] | None = None
) -> tuple[Reference, ...]:
    """References cited by a document.

    Args:
        document: the document.
        versions: installed versions by package name; an entry with the field
            `package` gets the field `note` "Version x" of its package.

    Returns:
        The cited references in the order of their first citation.

    Raises:
        KeyError: when a cited key is not in the bundled references.
    """
    refs = load_references()
    versions = versions or {}
    out = []
    for key in cited_keys(document):
        if key not in refs:
            raise KeyError(f"cited reference '{key}' is not in the bundled references")
        ref = refs[key]
        package = ref.fields.get("package")
        if package and package in versions:
            ref = Reference(
                ref.key,
                ref.entry_type,
                {**ref.fields, "note": f"Version {versions[package]}"},
            )
        out.append(ref)
    return tuple(out)


def to_bibtex(references: Iterable[Reference]) -> str:
    """BibTeX text of references.

    Args:
        references: the references.

    Returns:
        The entries, readable by `load_references`, without the field
        `package`.
    """
    entries = []
    for ref in references:
        lines = [f"@{ref.entry_type}{{{ref.key},"]
        lines += [
            f"  {name} = {{{value}}},"
            for name, value in ref.fields.items()
            if name != "package"
        ]
        entries.append("\n".join([*lines, "}"]))
    return "\n\n".join(entries) + "\n" if entries else ""
