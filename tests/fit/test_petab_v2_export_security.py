"""Exports cannot write through untrusted filenames or destination symlinks."""

from pathlib import Path
from tempfile import TemporaryDirectory

import libsbml
import pytest

from sbmlsim.fit.petab_v2.export import to_petab
from sbmlsim.fit.petab_v2.reader import PetabReader
from sbmlsim.model.provenance import Derivation
from tests.fit.test_petab_v2_reader import SETTINGS, write_problem


@pytest.mark.parametrize("source_kind", ["parent", "absolute", "nested_parent"])
def test_provenance_cannot_escape_the_export_directory(source_kind: str) -> None:
    """A malicious source annotation cannot overwrite a sibling sentinel."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        sentinel = root / "sentinel.xml"
        sentinel.write_text("Keep this harmless sentinel.")
        source_name = {
            "parent": "../sentinel.xml",
            "absolute": str(sentinel),
            "nested_parent": "sub/../../sentinel.xml",
        }[source_kind]
        path = write_problem(root / "input", {"prey_o": "prey"}, {"e1": None})
        source = root / "input" / "lv.xml"
        document = libsbml.readSBMLFromFile(str(source))
        document.getModel().appendAnnotation(Derivation(source=source_name).xml())
        assert libsbml.writeSBMLToFile(document, str(source))
        reader = PetabReader.from_yaml(path)
        problem = reader.to_optimization_problem()
        problem.initialize(SETTINGS)
        with pytest.raises(ValueError, match="filename"):
            to_petab(problem, root / "exported")
        assert sentinel.read_text() == "Keep this harmless sentinel."


@pytest.mark.parametrize("filename", ["lv.xml", "measurements.tsv", "problem.yaml"])
def test_export_rejects_destination_symlinks_outside_directory(filename: str) -> None:
    """Every export destination is checked before any files are written."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        sentinel = root / "sentinel.txt"
        sentinel.write_text("Keep this harmless sentinel.")
        output = root / "exported"
        output.mkdir()
        (output / filename).symlink_to(sentinel)
        path = write_problem(root / "input", {"prey_o": "prey"}, {"e1": None})
        reader = PetabReader.from_yaml(path)
        problem = reader.to_optimization_problem()
        problem.initialize(SETTINGS)
        with pytest.raises(ValueError, match="outside the export directory"):
            to_petab(problem, output)
        assert sentinel.read_text() == "Keep this harmless sentinel."
        assert list(output.iterdir()) == [output / filename]
