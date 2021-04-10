from pathlib import Path

from sbmlsim.combine.sedml.io import SEDMLInputType, SEDMLReader
from sbmlsim.test import DATA_DIR


def test_read_sedml_file1(tmp_path: Path) -> None:
    """Read SED-ML from file str."""
    repressilator_l1v4_sedml = (
        DATA_DIR / "sedml" / "l1v4" / "repressilator" / "repressilator_sedml.xml"
    )
    reader = SEDMLReader(source=str(repressilator_l1v4_sedml), working_dir=tmp_path)
    assert reader
    assert reader.sed_doc
    assert reader.input_type == SEDMLInputType.SEDML_FILE


def test_read_sedml_file2(tmp_path: Path) -> None:
    """Read SED-ML from file path."""
    repressilator_l1v4_sedml = (
        DATA_DIR / "sedml" / "l1v4" / "repressilator" / "repressilator_sedml.xml"
    )
    reader = SEDMLReader(source=repressilator_l1v4_sedml, working_dir=tmp_path)
    assert reader
    assert reader.sed_doc
    assert reader.input_type == SEDMLInputType.SEDML_FILE


def test_read_sedml_str(tmp_path: Path) -> None:
    """Read SED-ML from file"""
    with open(
        DATA_DIR / "sedml" / "l1v4" / "repressilator" / "repressilator_sedml.xml", "r"
    ) as f_in:
        sedml_str = f_in.read()
        reader = SEDMLReader(source=sedml_str, working_dir=tmp_path)
        assert reader
        assert reader.sed_doc
        assert reader.input_type == SEDMLInputType.SEDML_STRING


def test_read_sedml_omex(tmp_path: Path) -> None:
    """Read SED-ML from file"""
    source = DATA_DIR / "omex" / "tellurium" / "repressilator.omex"
    reader = SEDMLReader(source=source, working_dir=tmp_path)
    assert reader
    assert reader.sed_doc
    assert reader.input_type == SEDMLInputType.OMEX
