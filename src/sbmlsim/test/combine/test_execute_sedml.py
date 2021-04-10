from pathlib import Path

import pytest

from sbmlsim.examples.sedml.execute_sedml import base_path, execute_sedml


@pytest.mark.parametrize(
    "name, sedml_file",
    [
        ("Repressilator", "repressilator_sedml.xml"),
        ("TestFile1", "test_file_1.sedml"),
        ("TestLineFill", "test_line_fill.sedml"),
        ("MarkerType", "markertype.sedml"),
        ("StackedBar", "stacked_bar.sedml"),
        ("HBarStacked", "test_3hbarstacked.sedml"),
        ("Bar", "test_bar.sedml"),
        ("Bar", "test_bar3stacked.sedml"),
        ("StackedBar", "test_file.sedml"),
        ("StackedBar", "test_hbar_stacked.sedml"),
        ("StackedBar", "test_shaded_area.sedml"),
    ],
)
def test_execute_sedml(tmp_path: Path, name: str, sedml_file: str) -> None:
    working_dir = base_path / "experiments"
    execute_sedml(working_dir=working_dir, name=name, path=working_dir / sedml_file)
